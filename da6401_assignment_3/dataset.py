from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import spacy
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def build(
        cls,
        token_sequences: Sequence[Sequence[str]],
        min_freq: int = 1,
        max_size: Optional[int] = None,
    ) -> "Vocab":
        counter = Counter()
        for tokens in token_sequences:
            counter.update(tokens)

        itos = list(SPECIAL_TOKENS)
        for token, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            if token in SPECIAL_TOKENS:
                continue
            if freq < min_freq:
                continue
            if max_size is not None and (len(itos) - len(SPECIAL_TOKENS)) >= max_size:
                break
            itos.append(token)

        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def __len__(self) -> int:
        return len(self.itos)

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]

    def lookup_index(self, token: str) -> int:
        return self.stoi.get(token, self.stoi["<unk>"])


def build_spacy_tokenizer(lang: str) -> Callable[[str], list[str]]:
    nlp = spacy.blank(lang)

    def tokenize(text: str) -> list[str]:
        return [tok.text.lower() for tok in nlp.tokenizer(text.strip()) if tok.text.strip()]

    return tokenize


def _add_special_tokens(tokens: Sequence[str]) -> list[str]:
    return ["<sos>", *tokens, "<eos>"]


def _maybe_slice(split, max_samples: Optional[int]):
    if max_samples is None:
        return split
    max_samples = max(0, min(max_samples, len(split)))
    return split.select(range(max_samples))


class Multi30kTorchDataset(Dataset):
    def __init__(
        self,
        hf_split,
        src_tokenizer: Callable[[str], list[str]],
        tgt_tokenizer: Callable[[str], list[str]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_src_len: Optional[int] = None,
        max_tgt_len: Optional[int] = None,
    ) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for example in hf_split:
            src_tokens = _add_special_tokens(src_tokenizer(example["de"]))
            tgt_tokens = _add_special_tokens(tgt_tokenizer(example["en"]))

            if max_src_len is not None:
                src_tokens = src_tokens[:max_src_len]
                if src_tokens[-1] != "<eos>":
                    src_tokens[-1] = "<eos>"
            if max_tgt_len is not None:
                tgt_tokens = tgt_tokens[:max_tgt_len]
                if tgt_tokens[-1] != "<eos>":
                    tgt_tokens[-1] = "<eos>"

            src_ids = [src_vocab.lookup_index(token) for token in src_tokens]
            tgt_ids = [tgt_vocab.lookup_index(token) for token in tgt_tokens]
            self.samples.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def build_multi30k_splits(
    dataset_name: str = "bentrevett/multi30k",
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    max_src_len: Optional[int] = None,
    max_tgt_len: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
):
    ds = load_dataset(dataset_name, cache_dir=cache_dir)
    train_split = _maybe_slice(ds["train"], max_train_samples)
    val_split = _maybe_slice(ds["validation"], max_val_samples)
    test_split = _maybe_slice(ds["test"], max_test_samples)

    src_tokenizer = build_spacy_tokenizer("de")
    tgt_tokenizer = build_spacy_tokenizer("en")

    src_train_tokens = [src_tokenizer(example["de"]) for example in train_split]
    tgt_train_tokens = [tgt_tokenizer(example["en"]) for example in train_split]
    src_vocab = Vocab.build(src_train_tokens, min_freq=min_freq, max_size=max_vocab_size)
    tgt_vocab = Vocab.build(tgt_train_tokens, min_freq=min_freq, max_size=max_vocab_size)

    train_data = Multi30kTorchDataset(
        train_split,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )
    val_data = Multi30kTorchDataset(
        val_split,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )
    test_data = Multi30kTorchDataset(
        test_split,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
    )
    return train_data, val_data, test_data, src_vocab, tgt_vocab


def make_collate_fn(pad_idx: int):
    def collate(batch):
        src_batch = [src for src, _ in batch]
        tgt_batch = [tgt for _, tgt in batch]
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
        return src_padded, tgt_padded

    return collate


def create_dataloaders(
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    dataset_name: str = "bentrevett/multi30k",
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    max_src_len: Optional[int] = None,
    max_tgt_len: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
):
    train_data, val_data, test_data, src_vocab, tgt_vocab = build_multi30k_splits(
        dataset_name=dataset_name,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        cache_dir=cache_dir,
    )
    pad_idx = src_vocab.stoi["<pad>"]
    collate_fn = make_collate_fn(pad_idx)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


class Multi30kDataset:
    """
    Template-compatible dataset wrapper kept for autograder/skeleton parity.
    """

    def __init__(self, split="train"):
        self.split = split
        self.dataset_name = "bentrevett/multi30k"
        self.src_tokenizer = build_spacy_tokenizer("de")
        self.tgt_tokenizer = build_spacy_tokenizer("en")
        split_map = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}
        if split not in split_map:
            raise ValueError(f"Unsupported split '{split}'. Use train/validation(test)/val.")
        ds = load_dataset(self.dataset_name)
        self.samples = list(ds[split_map[split]])
        self.src_vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.tgt_vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}

    def build_vocab(self):
        """
        Build vocabulary using only the training split to avoid data leakage.
        """
        ds_train = load_dataset(self.dataset_name, split="train")
        src_tokens = [self.src_tokenizer(ex["de"]) for ex in ds_train]
        tgt_tokens = [self.tgt_tokenizer(ex["en"]) for ex in ds_train]

        src_vocab_obj = Vocab.build(src_tokens)
        tgt_vocab_obj = Vocab.build(tgt_tokens)
        self.src_vocab = dict(src_vocab_obj.stoi)
        self.tgt_vocab = dict(tgt_vocab_obj.stoi)
        return self.src_vocab, self.tgt_vocab

    def process_data(self):
        """
        Tokenize and convert current split into index sequences with <sos>/<eos>.
        """
        if len(self.src_vocab) <= len(SPECIAL_TOKENS) or len(self.tgt_vocab) <= len(SPECIAL_TOKENS):
            self.build_vocab()

        data = []
        for ex in self.samples:
            src_tokens = _add_special_tokens(self.src_tokenizer(ex["de"]))
            tgt_tokens = _add_special_tokens(self.tgt_tokenizer(ex["en"]))
            src_ids = [self.src_vocab.get(t, self.src_vocab["<unk>"]) for t in src_tokens]
            tgt_ids = [self.tgt_vocab.get(t, self.tgt_vocab["<unk>"]) for t in tgt_tokens]
            data.append((src_ids, tgt_ids))
        return data
