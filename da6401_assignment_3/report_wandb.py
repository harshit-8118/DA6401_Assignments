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
    checkpoint_path: Optional[str] = None,
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
        checkpoint_path=checkpoint_path,
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
    val_bleu = evaluate_bleu(model, val_loader, tgt_vocab, device=device, max_len=args.bleu_max_len)
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=args.bleu_max_len)
    wandb_run.log(
        {
            f"{run_prefix}/val_bleu": val_bleu,
            f"{run_prefix}/test_bleu": test_bleu,
            f"{run_prefix}/best_val_loss": best_val,
        }
    )
    return {"best_val_loss": best_val, "val_bleu": val_bleu, "test_bleu": test_bleu}


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
    target_self = model.encoder.layers[-1].self_attn

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

    target_self.forward = MethodType(forward_with_capture, target_self)


def experiment_head_specialization(args, wandb_run) -> None:
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(args, args.device_resolved)
    ckpt_path = args.head_checkpoint_path if args.head_checkpoint_path else None
    if ckpt_path is not None and not ckpt_path.strip():
        ckpt_path = None
    model = create_model(len(src_vocab), len(tgt_vocab), args, checkpoint_path=ckpt_path)
    model = model.to(args.device_resolved)

    if args.head_train_epochs > 0:
        original_epochs = args.epochs
        args.epochs = args.head_train_epochs
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
        args.epochs = original_epochs

    attach_last_encoder_attention_capture(model)
    model.eval()

    src = select_probe_german_sentence(test_loader, model, args)
    src = src.to(args.device_resolved)
    src_mask = make_src_mask(src)
    with torch.no_grad():
        _ = model.encode(src, src_mask)

    self_attn = model.encoder.layers[-1].self_attn.last_attn  # [1, heads, src_len, src_len]
    if self_attn is None:
        return
    self_attn = self_attn[0]

    src_ids = src[0].detach().cpu().tolist()
    src_ids = trim_pad(src_ids, model.pad_idx)

    src_tokens = [src_vocab.lookup_token(i) if i < len(src_vocab.itos) else "<unk>" for i in src_ids]
    src_labels = positional_labels(src_tokens)
    self_attn = self_attn[:, : len(src_tokens), : len(src_tokens)]

    for h in range(self_attn.size(0)):
        self_mat = self_attn[h].numpy()
        self_values = self_mat.reshape(-1).tolist()
        self_table = wandb.Table(data=[[v] for v in self_values], columns=["weight"])
        self_hist = wandb.plot.histogram(self_table, value="weight", title=f"Encoder head {h} attention weights")
        wandb_run.log({f"encoder_head_{h}_attn_hist": self_hist})
        log_attention_heatmap_custom_chart(
            wandb_run=wandb_run,
            key=f"encoder_head_{h}_attn_heatmap",
            matrix=self_mat,
            x_labels=src_labels,
            y_labels=src_labels,
            vega_spec_name=args.heatmap_spec_name,
            title=f"Encoder self-attn head {h}",
        )


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


def summarize_confidence(prefix: str, conf: List[float], wandb_run) -> None:
    if not conf:
        return
    arr = np.asarray(conf, dtype=np.float64)
    wandb_run.summary[f"{prefix}/mean"] = float(np.mean(arr))
    wandb_run.summary[f"{prefix}/std"] = float(np.std(arr))
    wandb_run.summary[f"{prefix}/p90"] = float(np.percentile(arr, 90))
    wandb_run.summary[f"{prefix}/p95"] = float(np.percentile(arr, 95))
    wandb_run.summary[f"{prefix}/gt_095_frac"] = float(np.mean(arr > 0.95))
    wandb_run.summary[f"{prefix}/lt_010_frac"] = float(np.mean(arr < 0.10))


def log_normalized_confidence_distribution(
    key: str,
    title: str,
    conf: List[float],
    wandb_run,
    bins: int = 20,
) -> None:
    if not conf:
        return
    arr = np.asarray(conf, dtype=np.float64)
    hist, edges = np.histogram(arr, bins=bins, range=(0.0, 1.0), density=False)
    frac = hist / max(hist.sum(), 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    rows = [[float(c), float(f)] for c, f in zip(centers, frac)]
    table = wandb.Table(data=rows, columns=["bin_center", "fraction"])
    chart = wandb.plot.bar(table, "bin_center", "fraction", title=title)
    wandb_run.log({key: chart})


def trim_pad(token_ids: List[int], pad_idx: int) -> List[int]:
    if pad_idx in token_ids:
        return token_ids[: token_ids.index(pad_idx)]
    return token_ids


def positional_labels(tokens: List[str]) -> List[str]:
    return [f"{i:02d}:{tok}" for i, tok in enumerate(tokens)]


def select_probe_german_sentence(test_loader, model: Transformer, args):
    if args.head_sentence is not None:
        de_tokens = [tok.text.lower() for tok in model.src_tokenizer(args.head_sentence.strip()) if tok.text.strip()]
        if len(de_tokens) != args.head_words:
            print(
                f"[warn] Provided head sentence has {len(de_tokens)} tokens, expected {args.head_words}."
            )
        de_tokens = ["<sos>"] + de_tokens + ["<eos>"]
        src_ids = [model.src_vocab.get(t, model.src_vocab["<unk>"]) for t in de_tokens]
        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0)
        return src

    target_words = args.head_words
    best = None
    best_gap = 10**9
    for src_batch, tgt_batch in test_loader:
        for i in range(src_batch.size(0)):
            src_ids = src_batch[i].tolist()
            src_ids_trim = trim_pad(src_ids, model.pad_idx)
            # remove <sos>, <eos> if present for counting words
            core = [t for t in src_ids_trim if t not in (model.sos_idx, model.eos_idx, model.pad_idx)]
            gap = abs(len(core) - target_words)
            if gap < best_gap:
                best_gap = gap
                best = torch.tensor(src_ids_trim, dtype=torch.long).unsqueeze(0)
            if gap == 0:
                return best
    if best is None:
        src_batch, tgt_batch = next(iter(test_loader))
        return src_batch[0:1].cpu()
    return best


def log_attention_heatmap_custom_chart(
    wandb_run,
    key: str,
    matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    vega_spec_name: str,
    title: str,
) -> None:
    rows = []
    for i, y in enumerate(y_labels):
        for j, x in enumerate(x_labels):
            val = float(matrix[i, j])
            rows.append([x, y, val])

    table = wandb.Table(data=rows, columns=["x", "y", "value"])
    chart = wandb.plot_table(
        vega_spec_name=vega_spec_name,
        data_table=table,
        fields={"x": "x", "y": "y", "value": "value"},
        string_fields={"title": title},
    )
    wandb_run.log({key: chart})


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

    wandb_run.summary["sinusoidal_val_bleu"] = stats_sin["val_bleu"]
    wandb_run.summary["learned_pos_val_bleu"] = stats_learned["val_bleu"]
    wandb_run.summary["sinusoidal_test_bleu"] = stats_sin["test_bleu"]
    wandb_run.summary["learned_pos_test_bleu"] = stats_learned["test_bleu"]
    wandb_run.summary["positional_val_bleu_delta_learned_minus_sinusoidal"] = (
        stats_learned["val_bleu"] - stats_sin["val_bleu"]
    )


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
    log_normalized_confidence_distribution(
        key="ls_0_1/confidence_fraction_bins",
        title="Prediction confidence fraction by bin (label smoothing 0.1)",
        conf=conf_ls,
        wandb_run=wandb_run,
        bins=args.confidence_bins,
    )
    summarize_confidence("ls_0_1/confidence", conf_ls, wandb_run)

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
    log_normalized_confidence_distribution(
        key="ls_0_0/confidence_fraction_bins",
        title="Prediction confidence fraction by bin (label smoothing 0.0)",
        conf=conf_ce,
        wandb_run=wandb_run,
        bins=args.confidence_bins,
    )
    summarize_confidence("ls_0_0/confidence", conf_ce, wandb_run)


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
    parser.add_argument("--confidence-bins", type=int, default=20)
    parser.add_argument("--head-sentence", type=str, default=None)
    parser.add_argument("--head-words", type=int, default=10)
    parser.add_argument("--head-checkpoint-path", type=str, default="checkpoint.pt")
    parser.add_argument("--head-train-epochs", type=int, default=0)
    parser.add_argument(
        "--heatmap-spec-name",
        type=str,
        default="da25s003-indian-institute-of-technology-madras/my_heatmap",
    )
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
