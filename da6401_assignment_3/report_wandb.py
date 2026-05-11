from __future__ import annotations

import argparse
import contextlib
import math
import random
from types import MethodType
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb

import model as model_module
from dataset import create_dataloaders
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask
from train import LabelSmoothingLoss, evaluate_bleu, run_epoch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> str:
    if name == "cpu":
        return "cpu"
    if name == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class LearnedPositionalTransformer(Transformer):
    def __init__(self, *args, learned_max_len: int = 512, **kwargs) -> None:
        kwargs["checkpoint_path"] = None
        super().__init__(*args, **kwargs)
        self.learned_max_len = learned_max_len
        self.learned_pos_embed = nn.Embedding(learned_max_len, self.d_model)
        self.learned_pos_dropout = nn.Dropout(self.dropout_p)

    def _learned_pos(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if seq_len > self.learned_max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds learned positional max {self.learned_max_len}."
            )
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        return self.learned_pos_dropout(x + self.learned_pos_embed(pos_ids))

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        src_emb = self._learned_pos(src_emb)
        return self.encoder(src_emb, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_emb = self._learned_pos(tgt_emb)
        dec_out = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        return self.generator(dec_out)


@contextlib.contextmanager
def attention_scaling(enabled: bool):
    original_fn = model_module.scaled_dot_product_attention

    def scaled(Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.bool(), float("-inf"))
        attn_w = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_w, V), attn_w

    def unscaled(Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask.bool(), float("-inf"))
        attn_w = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_w, V), attn_w

    model_module.scaled_dot_product_attention = scaled if enabled else unscaled
    try:
        yield
    finally:
        model_module.scaled_dot_product_attention = original_fn


def create_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    args,
    learned_positional: bool = False,
):
    model_cls = LearnedPositionalTransformer if learned_positional else Transformer
    model = model_cls(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        N=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        checkpoint_path=None,
    )
    return model


def get_dataloaders(args, device: str):
    return create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device == "cuda",
        dataset_name=args.dataset_name,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        cache_dir=args.cache_dir,
    )


def train_and_eval(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    tgt_vocab,
    args,
    scheduler_mode: str,
    label_smoothing: float,
    run_prefix: str,
    wandb_run,
) -> Dict[str, float]:
    device = args.device_resolved
    model = model.to(device)
    effective_lr = args.lr if scheduler_mode == "noam" else args.fixed_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = (
        NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)
        if scheduler_mode == "noam"
        else None
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=tgt_vocab.stoi["<pad>"],
        smoothing=label_smoothing,
    )

    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
        )
        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
        )
        lr_now = optimizer.param_groups[0]["lr"]
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        wandb_run.log(
            {
                f"{run_prefix}/epoch": epoch + 1,
                f"{run_prefix}/train_loss": train_loss,
                f"{run_prefix}/val_loss": val_loss,
                f"{run_prefix}/lr": lr_now,
                f"{run_prefix}/effective_base_lr": effective_lr,
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=args.bleu_max_len)
    wandb_run.log({f"{run_prefix}/test_bleu": test_bleu, f"{run_prefix}/best_val_loss": best_val})
    return {"best_val_loss": best_val, "test_bleu": test_bleu}


def experiment_noam_vs_fixed(args, wandb_run) -> None:
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(args, args.device_resolved)

    model_noam = create_model(len(src_vocab), len(tgt_vocab), args)
    stats_noam = train_and_eval(
        model_noam,
        train_loader,
        val_loader,
        test_loader,
        tgt_vocab,
        args,
        scheduler_mode="noam",
        label_smoothing=args.label_smoothing,
        run_prefix="noam",
        wandb_run=wandb_run,
    )

    model_fixed = create_model(len(src_vocab), len(tgt_vocab), args)
    stats_fixed = train_and_eval(
        model_fixed,
        train_loader,
        val_loader,
        test_loader,
        tgt_vocab,
        args,
        scheduler_mode="fixed",
        label_smoothing=args.label_smoothing,
        run_prefix="fixed_lr",
        wandb_run=wandb_run,
    )

    wandb_run.summary["noam_test_bleu"] = stats_noam["test_bleu"]
    wandb_run.summary["fixed_test_bleu"] = stats_fixed["test_bleu"]


def run_steps_with_grad_logging(
    model: nn.Module,
    fixed_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    tgt_vocab,
    args,
    label: str,
    wandb_run,
) -> Dict[str, List[float]]:
    device = args.device_resolved
    model = model.to(device)
    effective_lr = args.lr if args.scaling_scheduler == "noam" else args.scaling_fixed_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = (
        NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.scaling_warmup_steps)
        if args.scaling_scheduler == "noam"
        else None
    )
    loss_fn = LabelSmoothingLoss(len(tgt_vocab), pad_idx=tgt_vocab.stoi["<pad>"], smoothing=args.label_smoothing)

    model.train()
    q_series: List[float] = []
    k_series: List[float] = []
    for step, (src, tgt) in enumerate(fixed_batches, start=1):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_in)

        optimizer.zero_grad(set_to_none=True)
        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()

        w_q = model.encoder.layers[0].self_attn.w_q.weight.grad
        w_k = model.encoder.layers[0].self_attn.w_k.weight.grad
        q_norm = w_q.norm().item() if w_q is not None else 0.0
        k_norm = w_k.norm().item() if w_k is not None else 0.0
        q_series.append(float(q_norm))
        k_series.append(float(k_norm))

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step % args.log_every_steps == 0 or step == 1:
            wandb_run.log(
                {
                    f"{label}/step": step,
                    f"{label}/loss": float(loss.item()),
                    f"{label}/q_grad_norm": q_norm,
                    f"{label}/k_grad_norm": k_norm,
                    f"{label}/lr": optimizer.param_groups[0]["lr"],
                    f"{label}/effective_base_lr": effective_lr,
                }
            )
    return {"q": q_series, "k": k_series}


def summarize_series(prefix: str, values: List[float], wandb_run) -> None:
    if not values:
        return
    arr = np.asarray(values, dtype=np.float64)
    wandb_run.summary[f"{prefix}/mean"] = float(np.mean(arr))
    wandb_run.summary[f"{prefix}/std"] = float(np.std(arr))
    wandb_run.summary[f"{prefix}/p95"] = float(np.percentile(arr, 95))
    wandb_run.summary[f"{prefix}/max"] = float(np.max(arr))
    wandb_run.summary[f"{prefix}/min"] = float(np.min(arr))
    wandb_run.summary[f"{prefix}/last"] = float(arr[-1])


def collect_fixed_batches(train_loader, total_steps: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    data_iter = iter(train_loader)
    while len(batches) < total_steps:
        try:
            src, tgt = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            src, tgt = next(data_iter)
        batches.append((src.detach().cpu().clone(), tgt.detach().cpu().clone()))
    return batches


def experiment_scaling_ablation(args, wandb_run) -> None:
    train_loader, _, _, src_vocab, tgt_vocab = get_dataloaders(args, args.device_resolved)
    fixed_batches = collect_fixed_batches(train_loader, args.scaling_steps)

    with attention_scaling(True):
        model_scaled = create_model(len(src_vocab), len(tgt_vocab), args)
    init_state = {k: v.detach().cpu().clone() for k, v in model_scaled.state_dict().items()}

    with attention_scaling(True):
        model_scaled.load_state_dict(init_state, strict=True)
        scaled_stats = run_steps_with_grad_logging(model_scaled, fixed_batches, tgt_vocab, args, "scaled", wandb_run)

    with attention_scaling(False):
        model_unscaled = create_model(len(src_vocab), len(tgt_vocab), args)
        model_unscaled.load_state_dict(init_state, strict=True)
        unscaled_stats = run_steps_with_grad_logging(
            model_unscaled, fixed_batches, tgt_vocab, args, "unscaled", wandb_run
        )

    summarize_series("scaled/q_grad_norm", scaled_stats["q"], wandb_run)
    summarize_series("scaled/k_grad_norm", scaled_stats["k"], wandb_run)
    summarize_series("unscaled/q_grad_norm", unscaled_stats["q"], wandb_run)
    summarize_series("unscaled/k_grad_norm", unscaled_stats["k"], wandb_run)

    eps = 1e-12
    if scaled_stats["q"] and unscaled_stats["q"]:
        wandb_run.summary["scaling_ablation/q_mean_ratio_unscaled_over_scaled"] = float(
            (np.mean(unscaled_stats["q"]) + eps) / (np.mean(scaled_stats["q"]) + eps)
        )
    if scaled_stats["k"] and unscaled_stats["k"]:
        wandb_run.summary["scaling_ablation/k_mean_ratio_unscaled_over_scaled"] = float(
            (np.mean(unscaled_stats["k"]) + eps) / (np.mean(scaled_stats["k"]) + eps)
        )


def attach_last_encoder_attention_capture(model: Transformer) -> None:
    target = model.encoder.layers[-1].self_attn

    def forward_with_capture(self, query, key, value, mask=None):
        bsz = query.size(0)

        def split_heads(x):
            return x.view(bsz, -1, self.num_heads, self.d_k).transpose(1, 2)

        q = split_heads(self.w_q(query))
        k = split_heads(self.w_k(key))
        v = split_heads(self.w_v(value))
        out, attn_w = model_module.scaled_dot_product_attention(q, k, v, mask)
        self.last_attn = attn_w.detach().cpu()
        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)
        return self.w_o(out)

    target.forward = MethodType(forward_with_capture, target)


def experiment_head_specialization(args, wandb_run) -> None:
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(args, args.device_resolved)
    model = create_model(len(src_vocab), len(tgt_vocab), args)
    model = model.to(args.device_resolved)

    # Quick training if not loading external checkpoint
    _ = train_and_eval(
        model,
        train_loader,
        val_loader,
        test_loader,
        tgt_vocab,
        args,
        scheduler_mode="noam",
        label_smoothing=args.label_smoothing,
        run_prefix="head_exp",
        wandb_run=wandb_run,
    )

    attach_last_encoder_attention_capture(model)
    model.eval()
    src_batch, _ = next(iter(test_loader))
    src = src_batch[0:1].to(args.device_resolved)
    src_mask = make_src_mask(src)
    with torch.no_grad():
        _ = model.encode(src, src_mask)

    attn = model.encoder.layers[-1].self_attn.last_attn  # [1, heads, q, k]
    if attn is None:
        return
    attn = attn[0]
    token_ids = src[0].detach().cpu().tolist()
    src_tokens = [src_vocab.lookup_token(i) if i < len(src_vocab.itos) else "<unk>" for i in token_ids]

    for h in range(attn.size(0)):
        mat = attn[h].numpy()
        values = mat.reshape(-1).tolist()
        table = wandb.Table(data=[[v] for v in values], columns=["weight"])
        histogram = wandb.plot.histogram(table, value="weight", title=f"Head {h} attention weights")
        wandb_run.log({f"head_{h}_attn_hist": histogram})

        if args.log_heatmaps:
            try:
                heatmap = wandb.plots.HeatMap(src_tokens, src_tokens, mat.tolist(), show_text=False)
                wandb_run.log({f"head_{h}_attn_heatmap": heatmap})
            except Exception:
                pass


def collect_prediction_confidences(model, data_loader, tgt_vocab, device: str, max_batches: int = 20) -> List[float]:
    model.eval()
    pad_idx = tgt_vocab.stoi["<pad>"]
    confidences = []
    batches = 0
    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            src_mask = make_src_mask(src)
            tgt_mask = make_tgt_mask(tgt_in)
            logits = model(src, tgt_in, src_mask, tgt_mask)
            probs = torch.softmax(logits, dim=-1)
            chosen = probs.gather(dim=-1, index=tgt_out.unsqueeze(-1)).squeeze(-1)
            mask = tgt_out != pad_idx
            confidences.extend(chosen[mask].detach().cpu().tolist())
            batches += 1
            if batches >= max_batches:
                break
    return confidences


def experiment_positional_vs_learned(args, wandb_run) -> None:
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(args, args.device_resolved)

    model_sin = create_model(len(src_vocab), len(tgt_vocab), args, learned_positional=False)
    stats_sin = train_and_eval(
        model_sin,
        train_loader,
        val_loader,
        test_loader,
        tgt_vocab,
        args,
        scheduler_mode="noam",
        label_smoothing=args.label_smoothing,
        run_prefix="sinusoidal",
        wandb_run=wandb_run,
    )

    model_learned = create_model(len(src_vocab), len(tgt_vocab), args, learned_positional=True)
    stats_learned = train_and_eval(
        model_learned,
        train_loader,
        val_loader,
        test_loader,
        tgt_vocab,
        args,
        scheduler_mode="noam",
        label_smoothing=args.label_smoothing,
        run_prefix="learned_pos",
        wandb_run=wandb_run,
    )

    wandb_run.summary["sinusoidal_test_bleu"] = stats_sin["test_bleu"]
    wandb_run.summary["learned_pos_test_bleu"] = stats_learned["test_bleu"]


def experiment_label_smoothing(args, wandb_run) -> None:
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(args, args.device_resolved)

    model_ls = create_model(len(src_vocab), len(tgt_vocab), args)
    _ = train_and_eval(
        model_ls,
        train_loader,
        val_loader,
        test_loader,
        tgt_vocab,
        args,
        scheduler_mode="noam",
        label_smoothing=0.1,
        run_prefix="ls_0_1",
        wandb_run=wandb_run,
    )
    conf_ls = collect_prediction_confidences(model_ls, val_loader, tgt_vocab, args.device_resolved, args.confidence_batches)
    table_ls = wandb.Table(data=[[v] for v in conf_ls], columns=["confidence"])
    hist_ls = wandb.plot.histogram(table_ls, value="confidence", title="Prediction confidence (label smoothing 0.1)")
    wandb_run.log({"ls_0_1/confidence_hist": hist_ls})

    model_ce = create_model(len(src_vocab), len(tgt_vocab), args)
    _ = train_and_eval(
        model_ce,
        train_loader,
        val_loader,
        test_loader,
        tgt_vocab,
        args,
        scheduler_mode="noam",
        label_smoothing=0.0,
        run_prefix="ls_0_0",
        wandb_run=wandb_run,
    )
    conf_ce = collect_prediction_confidences(model_ce, val_loader, tgt_vocab, args.device_resolved, args.confidence_batches)
    table_ce = wandb.Table(data=[[v] for v in conf_ce], columns=["confidence"])
    hist_ce = wandb.plot.histogram(table_ce, value="confidence", title="Prediction confidence (label smoothing 0.0)")
    wandb_run.log({"ls_0_0/confidence_hist": hist_ce})


def parse_args():
    parser = argparse.ArgumentParser(description="Run W&B report experiments for DA6401 A3.")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=[
            "noam_vs_fixed",
            "scaling_ablation",
            "head_specialization",
            "positional_vs_learned",
            "label_smoothing",
            "all",
        ],
    )
    parser.add_argument("--project", type=str, default="DA6401-Assignment_3")
    parser.add_argument("--entity", type=str, default="da25s003-indian-institute-of-technology-madras")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--group", type=str, default="wandb_report")

    parser.add_argument("--dataset-name", type=str, default="bentrevett/multi30k")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--fixed-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--bleu-max-len", type=int, default=120)

    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=None)
    parser.add_argument("--max-src-len", type=int, default=None)
    parser.add_argument("--max-tgt-len", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)

    parser.add_argument("--scaling-steps", type=int, default=1000)
    parser.add_argument("--scaling-scheduler", type=str, default="noam", choices=["noam", "fixed"])
    parser.add_argument("--scaling-warmup-steps", type=int, default=4000)
    parser.add_argument("--scaling-fixed-lr", type=float, default=1e-4)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--confidence-batches", type=int, default=20)
    parser.add_argument("--log-heatmaps", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.device_resolved = resolve_device(args.device)

    wandb_run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        group=args.group,
        config=vars(args),
    )

    if args.experiment == "noam_vs_fixed":
        experiment_noam_vs_fixed(args, wandb_run)
    elif args.experiment == "scaling_ablation":
        experiment_scaling_ablation(args, wandb_run)
    elif args.experiment == "head_specialization":
        experiment_head_specialization(args, wandb_run)
    elif args.experiment == "positional_vs_learned":
        experiment_positional_vs_learned(args, wandb_run)
    elif args.experiment == "label_smoothing":
        experiment_label_smoothing(args, wandb_run)
    elif args.experiment == "all":
        experiment_noam_vs_fixed(args, wandb_run)
        experiment_scaling_ablation(args, wandb_run)
        experiment_head_specialization(args, wandb_run)
        experiment_positional_vs_learned(args, wandb_run)
        experiment_label_smoothing(args, wandb_run)

    wandb.finish()


if __name__ == "__main__":
    main()
