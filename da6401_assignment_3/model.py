import math
import copy
import os
import json
import re
from collections import Counter
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask.bool(), float("-inf"))
    attn_w = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w


def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    bsz, tgt_len = tgt.size()
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool), diagonal=1
    ).unsqueeze(0).unsqueeze(1)
    return pad_mask | causal_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = query.size(0)

        def split_heads(x):
            return x.view(bsz, -1, self.num_heads, self.d_k).transpose(1, 2)

        q = split_heads(self.w_q(query))
        k = split_heads(self.w_k(key))
        v = split_heads(self.w_v(value))

        out, attn_w = scaled_dot_product_attention(q, k, v, mask)
        out = self.dropout(out)

        out = out.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)
        return self.w_o(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn))
        ff = self.ffn(x)
        x = self.norm2(x + self.dropout2(ff))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn))
        cross_attn = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn))
        ff = self.ffn(x)
        x = self.norm3(x + self.dropout3(ff))
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    _shared_tokenizers = None
    _shared_vocabs = None

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: str = "checkpoint.pt",
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self._init_nlp_resources()

        if src_vocab_size is None:
            src_vocab_size = len(self.src_vocab)
        if tgt_vocab_size is None:
            tgt_vocab_size = len(self.tgt_vocab)

        self.src_vocab_size = int(src_vocab_size)
        self.tgt_vocab_size = int(tgt_vocab_size)
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self.pad_idx = self.src_vocab.get("<pad>", 1)
        self.sos_idx = self.tgt_vocab.get("<sos>", 2)
        self.eos_idx = self.tgt_vocab.get("<eos>", 3)

        self._maybe_download_artifacts()
        self._load_weights_from_checkpoint()

    def _init_nlp_resources(self) -> None:
        if Transformer._shared_tokenizers is None:
            import spacy

            Transformer._shared_tokenizers = {
                "de": spacy.blank("de").tokenizer,
                "en": spacy.blank("en").tokenizer,
            }
        self.src_tokenizer = Transformer._shared_tokenizers["de"]
        self.tgt_tokenizer = Transformer._shared_tokenizers["en"]

        if Transformer._shared_vocabs is None:
            src_vocab, tgt_vocab = self._load_or_build_vocabs()
            Transformer._shared_vocabs = {
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "src_itos": self._invert_vocab(src_vocab),
                "tgt_itos": self._invert_vocab(tgt_vocab),
            }

        shared = Transformer._shared_vocabs
        self.src_vocab = shared["src_vocab"]
        self.tgt_vocab = shared["tgt_vocab"]
        self.src_itos = shared["src_itos"]
        self.tgt_itos = shared["tgt_itos"]

    def _invert_vocab(self, stoi):
        size = max(stoi.values()) + 1
        itos = ["<unk>"] * size
        for token, idx in stoi.items():
            if 0 <= idx < size:
                itos[idx] = token
        return itos

    def _load_or_build_vocabs(self):
        vocab_path = os.getenv("A3_VOCAB_PATH", "vocab.json")
        vocab_drive_id = os.getenv("A3_VOCAB_DRIVE_ID", "1ysSSFaEk0Kh1YaC_blO_CRE78WTIV-Qn")
        if os.path.exists(vocab_path):
            return self._load_vocab_file(vocab_path)

        if vocab_drive_id:
            try:
                import gdown

                gdown.download(id=vocab_drive_id, output=vocab_path, quiet=True)
                if os.path.exists(vocab_path):
                    return self._load_vocab_file(vocab_path)
            except Exception:
                pass

        src_vocab, tgt_vocab = self._build_vocab_from_multi30k()
        try:
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump({"src_vocab": src_vocab, "tgt_vocab": tgt_vocab}, f, ensure_ascii=False)
        except Exception:
            pass
        return src_vocab, tgt_vocab

    def _load_vocab_file(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "src_vocab" not in data or "tgt_vocab" not in data:
            raise ValueError("vocab.json must contain 'src_vocab' and 'tgt_vocab'.")
        src_vocab = {k: int(v) for k, v in data["src_vocab"].items()}
        tgt_vocab = {k: int(v) for k, v in data["tgt_vocab"].items()}
        return src_vocab, tgt_vocab

    def _build_vocab_from_multi30k(self):
        from datasets import load_dataset

        specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
        src_counter = Counter()
        tgt_counter = Counter()
        ds_train = load_dataset("bentrevett/multi30k", split="train")

        for ex in ds_train:
            src_tokens = [tok.text.lower() for tok in self.src_tokenizer(ex["de"]) if tok.text.strip()]
            tgt_tokens = [tok.text.lower() for tok in self.tgt_tokenizer(ex["en"]) if tok.text.strip()]
            src_counter.update(src_tokens)
            tgt_counter.update(tgt_tokens)

        src_vocab = {tok: idx for idx, tok in enumerate(specials)}
        tgt_vocab = {tok: idx for idx, tok in enumerate(specials)}
        for token, _ in sorted(src_counter.items(), key=lambda x: (-x[1], x[0])):
            if token not in src_vocab:
                src_vocab[token] = len(src_vocab)
        for token, _ in sorted(tgt_counter.items(), key=lambda x: (-x[1], x[0])):
            if token not in tgt_vocab:
                tgt_vocab[token] = len(tgt_vocab)
        return src_vocab, tgt_vocab

    def _maybe_download_artifacts(self) -> None:
        if self.checkpoint_path is None:
            return
        if os.path.exists(self.checkpoint_path):
            return
        ckpt_drive_id = os.getenv("A3_CKPT_DRIVE_ID", "1EjAiHVs1JzynUQ5eciAamwsgb5rWZyri") # checkpoint.pt
        if not ckpt_drive_id:
            return
        try:
            import gdown

            gdown.download(id=ckpt_drive_id, output=self.checkpoint_path, quiet=False)
        except Exception:
            pass

    def _load_weights_from_checkpoint(self) -> None:
        if self.checkpoint_path is None or not os.path.exists(self.checkpoint_path):
            return
        ckpt = torch.load(self.checkpoint_path, weights_only=True, map_location="cpu")
        loaded_state = ckpt.get("model_state_dict", ckpt)
        current_state = self.state_dict()
        compatible_state = {}
        for k, v in loaded_state.items():
            if k in current_state and current_state[k].shape == v.shape:
                compatible_state[k] = v
        current_state.update(compatible_state)
        self.load_state_dict(current_state, strict=False)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(src_emb, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        dec_out = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        return self.generator(dec_out)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        self.eval()
        if not isinstance(src_sentence, str):
            raise TypeError("src_sentence must be a string.")

        src_tokens = [tok.text.lower() for tok in self.src_tokenizer(src_sentence.strip()) if tok.text.strip()]
        src_tokens = ["<sos>"] + src_tokens + ["<eos>"]
        src_ids = [self.src_vocab.get(tok, self.src_vocab["<unk>"]) for tok in src_tokens]

        device = next(self.parameters()).device
        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src, pad_idx=self.pad_idx)

        with torch.no_grad():
            memory = self.encode(src, src_mask)
            src_len = src.size(1)
            max_len = min(int(os.getenv("A3_INFER_MAX_LEN", "120")), 2 * src_len + 10)
            beam_size = max(1, int(os.getenv("A3_BEAM_SIZE", "4")))
            alpha = float(os.getenv("A3_BEAM_ALPHA", "0.6"))
            min_len = max(1, int(os.getenv("A3_INFER_MIN_LEN", "2")))

            if beam_size == 1:
                ys = torch.full((1, 1), self.sos_idx, dtype=torch.long, device=device)
                for step in range(max_len - 1):
                    tgt_mask = make_tgt_mask(ys, pad_idx=self.pad_idx)
                    out = self.decode(memory, src_mask, ys, tgt_mask)
                    logits = out[:, -1, :]
                    if step + 1 < min_len:
                        logits[:, self.eos_idx] = -1e9
                    next_word = torch.argmax(logits, dim=-1).item()
                    ys = torch.cat([ys, torch.tensor([[next_word]], dtype=torch.long, device=device)], dim=1)
                    if next_word == self.eos_idx:
                        break
                pred_ids = ys.squeeze(0).tolist()
            else:
                beams = [([self.sos_idx], 0.0, False)]
                finished = []

                def length_penalty(length):
                    return ((5.0 + float(length)) / 6.0) ** alpha

                for step in range(max_len - 1):
                    candidates = []
                    for tokens, score, done in beams:
                        if done:
                            candidates.append((tokens, score, True))
                            continue
                        ys = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                        tgt_mask = make_tgt_mask(ys, pad_idx=self.pad_idx)
                        out = self.decode(memory, src_mask, ys, tgt_mask)
                        log_probs = torch.log_softmax(out[:, -1, :], dim=-1).squeeze(0)
                        if step + 1 < min_len:
                            log_probs[self.eos_idx] = -1e9
                        topk_log_probs, topk_ids = torch.topk(log_probs, k=beam_size)

                        for lp, tid in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                            new_tokens = tokens + [int(tid)]
                            new_score = score + float(lp)
                            done_now = int(tid) == self.eos_idx
                            candidates.append((new_tokens, new_score, done_now))

                    candidates.sort(
                        key=lambda x: x[1] / length_penalty(len(x[0])),
                        reverse=True,
                    )
                    beams = candidates[:beam_size]
                    for b in beams:
                        if b[2]:
                            finished.append(b)
                    if len(finished) >= beam_size:
                        break

                final_pool = finished if finished else beams
                best = max(final_pool, key=lambda x: x[1] / (((5.0 + len(x[0])) / 6.0) ** alpha))
                pred_ids = best[0]

        pred_tokens = []
        for idx in pred_ids:
            if idx in (self.sos_idx, self.pad_idx):
                continue
            if idx == self.eos_idx:
                break
            if 0 <= idx < len(self.tgt_itos):
                pred_tokens.append(self.tgt_itos[idx])
            else:
                pred_tokens.append("<unk>")
        return self._detokenize_en(pred_tokens)

    def _detokenize_en(self, tokens) -> str:
        if not tokens:
            return ""
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r"\s+'\s*", "'", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
