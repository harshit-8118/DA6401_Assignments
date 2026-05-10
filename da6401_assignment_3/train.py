"""
train.py - Training, decoding, BLEU evaluation and checkpoint utilities.

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  greedy_decode(model, src, src_mask, max_len, start_symbol) -> torch.Tensor
  evaluate_bleu(model, test_dataloader, tgt_vocab, device) -> float
  save_checkpoint(model, optimizer, scheduler, epoch, path) -> None
  load_checkpoint(path, model, optimizer, scheduler) -> int
"""

from __future__ import annotations

import argparse
import math
import os
import json
import random
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import create_dataloaders
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask

_RUN_EPOCH_GRAD_CLIP: Optional[float] = None


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing from "Attention Is All You Need".
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0
            pad_mask = target == self.pad_idx
            true_dist[pad_mask] = 0

        loss = -(true_dist * log_probs).sum(dim=1)
        denom = (~(target == self.pad_idx)).sum().clamp_min(1)
        return loss.sum() / denom


def run_epoch(
    data_iter: DataLoader,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.
    """
    del epoch_num
    model.train() if is_train else model.eval()
    total_loss, total_batches = 0.0, 0

    for src, tgt in data_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_in)

        if is_train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            if is_train and optimizer is not None:
                loss.backward()
                if _RUN_EPOCH_GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), _RUN_EPOCH_GRAD_CLIP)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Greedy token-by-token decoding for a single source sentence (batch size 1).
    """
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        next_word = torch.argmax(out[:, -1, :], dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_word]], dtype=torch.long, device=device)], dim=1)
        if next_word == end_symbol:
            break

    return ys


def _extract_ngrams(tokens: List[str], n: int) -> Counter[Tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _corpus_bleu(references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4) -> float:
    if len(references) == 0 or len(hypotheses) == 0:
        return 0.0

    clipped_counts = [0 for _ in range(max_n)]
    total_counts = [0 for _ in range(max_n)]
    ref_len_sum = 0
    hyp_len_sum = 0

    for ref_tokens, hyp_tokens in zip(references, hypotheses):
        ref_len_sum += len(ref_tokens)
        hyp_len_sum += len(hyp_tokens)

        for n in range(1, max_n + 1):
            hyp_ngrams = _extract_ngrams(hyp_tokens, n)
            ref_ngrams = _extract_ngrams(ref_tokens, n)
            total_counts[n - 1] += sum(hyp_ngrams.values())
            for gram, count in hyp_ngrams.items():
                clipped_counts[n - 1] += min(count, ref_ngrams.get(gram, 0))

    if hyp_len_sum == 0:
        return 0.0

    precisions = []
    for clip, total in zip(clipped_counts, total_counts):
        if total == 0 or clip == 0:
            return 0.0
        precisions.append(clip / total)

    log_precision = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0 if hyp_len_sum > ref_len_sum else math.exp(1.0 - (ref_len_sum / hyp_len_sum))
    bleu = bp * math.exp(log_precision)
    return float(bleu * 100.0)


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Corpus BLEU over test_dataloader.
    """
    if hasattr(tgt_vocab, "stoi"):
        stoi = tgt_vocab.stoi
    else:
        stoi = {}

    sos = stoi.get("<sos>", 2)
    eos = stoi.get("<eos>", 3)
    pad = stoi.get("<pad>", 1)
    model.eos_idx = eos

    def lookup_token(idx: int) -> str:
        if hasattr(tgt_vocab, "itos"):
            return tgt_vocab.itos[idx]
        if hasattr(tgt_vocab, "lookup_token"):
            return tgt_vocab.lookup_token(idx)
        return str(idx)

    references: List[List[str]] = []
    hypotheses: List[List[str]] = []

    model.eval()
    with torch.no_grad():
        for src_batch, tgt_batch in test_dataloader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            for i in range(src_batch.size(0)):
                src = src_batch[i : i + 1]
                tgt = tgt_batch[i].tolist()
                src_mask = make_src_mask(src)
                pred = greedy_decode(model, src, src_mask, max_len, sos, eos, device=device).squeeze(0).tolist()

                if eos in pred:
                    pred = pred[: pred.index(eos) + 1]

                pred_tokens = [lookup_token(x) for x in pred if x not in (sos, eos, pad)]
                ref_tokens = [lookup_token(x) for x in tgt if x not in (sos, eos, pad)]

                references.append(ref_tokens)
                hypotheses.append(pred_tokens)

    return _corpus_bleu(references, hypotheses)


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimizer + scheduler state to disk.
    """
    model_config = {
        "src_vocab_size": model.src_vocab_size,
        "tgt_vocab_size": model.tgt_vocab_size,
        "d_model": model.d_model,
        "N": model.N,
        "num_heads": model.num_heads,
        "d_ff": model.d_ff,
        "dropout": model.dropout_p,
    }

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": model_config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model and optionally optimizer/scheduler state from disk.
    """
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])


    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return int(checkpoint.get("epoch", 0))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer on Multi30k (de->en).")

    parser.add_argument("--dataset-name", type=str, default="bentrevett/multi30k")
    parser.add_argument("--cache-dir", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")

    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["noam", "fixed"], default="noam")
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=None)

    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=None)
    parser.add_argument("--max-src-len", type=int, default=None)
    parser.add_argument("--max-tgt-len", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--checkpoint-path", type=str, default="checkpoint.pt")
    parser.add_argument("--vocab-save-path", type=str, default="vocab.json")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--bleu-max-len", type=int, default=100)

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="DA6401-Assignment_3")
    parser.add_argument("--wandb-entity", type=str, default="da25s003-indian-institute-of-technology-madras")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    return parser.parse_args()


def _resolve_device(name: str) -> str:
    if name == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_training_experiment() -> None:
    """
    Full experiment entry point with CLI configuration.
    """
    args = parse_args()
    set_seed(args.seed)
    device = _resolve_device(args.device)
    global _RUN_EPOCH_GRAD_CLIP
    _RUN_EPOCH_GRAD_CLIP = args.grad_clip

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_dataloaders(
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
    with open(args.vocab_save_path, "w", encoding="utf-8") as f:
        json.dump({"src_vocab": src_vocab.stoi, "tgt_vocab": tgt_vocab.stoi}, f, ensure_ascii=False)

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        N=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        checkpoint_path=None,
    ).to(device)
    model.eos_idx = tgt_vocab.stoi["<eos>"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = (
        NoamScheduler(optimizer, d_model=args.d_model, warmup_steps=args.warmup_steps)
        if args.scheduler == "noam"
        else None
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=tgt_vocab.stoi["<pad>"],
        smoothing=args.label_smoothing,
    )

    start_epoch = 0
    if args.resume_from is not None:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")
        last_epoch = load_checkpoint(args.resume_from, model, optimizer, scheduler)
        start_epoch = last_epoch + 1
        print(f"Resumed from {args.resume_from} at epoch {last_epoch}.")

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=vars(args))
        except Exception as exc:
            print(f"[warn] W&B disabled (init failed): {exc}")
            wandb_run = None

    if args.eval_only:
        bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=args.bleu_max_len)
        print(f"Test BLEU: {bleu:.4f}")
        if wandb_run is not None:
            wandb_run.log({"test_bleu": bleu})
            wandb_run.finish()
        return

    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
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
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr_now:.6e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(model, optimizer, scheduler, epoch, path=args.checkpoint_path)
            print(f"Saved best checkpoint at epoch {epoch + 1} (val_loss={val_loss:.4f}).")
        else:
            bad_epochs += 1

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr_now,
                    "best_val_loss": best_val_loss,
                }
            )

        if ((epoch + 1) % args.save_every) == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, path=f"{args.checkpoint_path}.latest")

        if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1}: "
                f"no val_loss improvement for {args.early_stop_patience} epochs."
            )
            break

    if best_epoch >= 0:
        print(f"Best checkpoint: epoch {best_epoch + 1}, val_loss={best_val_loss:.4f}, path={args.checkpoint_path}")

    if os.path.exists(args.checkpoint_path):
        load_checkpoint(args.checkpoint_path, model, optimizer=None, scheduler=None)
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=args.bleu_max_len)
    print(f"Final test BLEU: {test_bleu:.4f}")

    if wandb_run is not None:
        wandb_run.log({"test_bleu": test_bleu})
        wandb_run.finish()


if __name__ == "__main__":
    run_training_experiment()
