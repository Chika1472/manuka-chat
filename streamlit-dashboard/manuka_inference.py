from __future__ import annotations

import json
import math
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("\u200b", " ").replace("\ufeff", " ").strip())


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    rope_freqs: torch.Tensor,
    start_pos: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")

    original_dim = q.shape[-1]
    head_dim = original_dim + original_dim % 2
    if original_dim % 2 != 0:
        q = F.pad(q, (0, 1))
        k = F.pad(k, (0, 1))

    seq_len = q.shape[-2]
    freqs = rope_freqs[start_pos : start_pos + seq_len, :head_dim].to(
        device=q.device,
        dtype=torch.float32,
    )
    cos = freqs.cos().to(dtype=q.dtype).view(1, 1, seq_len, head_dim)
    sin = freqs.sin().to(dtype=q.dtype).view(1, 1, seq_len, head_dim)

    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    if original_dim % 2 != 0:
        q_out = q_out[..., :original_dim]
        k_out = k_out[..., :original_dim]
    return q_out.contiguous(), k_out.contiguous()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        max_len: int,
        n_kv_heads: int | None = None,
        qk_norm: bool = True,
        rope_theta: float = 10000.0,
        rope_scale: float = 1.0,
        rope_scaling_type: str = "linear",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        if n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if rope_scale <= 0:
            raise ValueError("rope_scale must be positive")
        if rope_scaling_type not in {"linear", "ntk"}:
            raise ValueError("rope_scaling_type must be 'linear' or 'ntk'")

        self.head_dim = d_model // n_heads
        self.kv_repeat = n_heads // self.n_kv_heads
        self.rope_theta = rope_theta
        self.rope_scale = rope_scale
        self.rope_scaling_type = rope_scaling_type

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("rope_freqs", self._build_rope_freqs(max_len), persistent=False)

    def _build_rope_freqs(self, max_len: int) -> torch.Tensor:
        head_dim = self.head_dim + self.head_dim % 2
        fractions = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        inv_freq = 1.0 / (self.rope_theta**fractions)
        if self.rope_scaling_type == "ntk":
            inv_freq = inv_freq / (self.rope_scale**fractions)
            positions = torch.arange(max_len, dtype=torch.float32)
        else:
            positions = torch.arange(max_len, dtype=torch.float32) / self.rope_scale
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        return torch.repeat_interleave(freqs, 2, dim=1)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.kv_repeat == 1:
            return x
        return x.repeat_interleave(self.kv_repeat, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch_size, seq_len, channels = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        past_len = past_kv[0].shape[-2] if past_kv is not None else 0
        q, k = apply_rope(q, k, rope_freqs=self.rope_freqs, start_pos=past_len)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        present = (k, v) if use_cache else None

        k_full = self._repeat_kv(k)
        v_full = self._repeat_kv(v)
        key_len = k_full.shape[-2]

        causal_needed = seq_len > 1 or past_kv is None
        attn_bias = None
        if attn_mask is not None:
            if attn_mask.dim() != 2:
                raise ValueError("attn_mask must have shape (batch, key_length)")
            if attn_mask.shape[-1] != key_len:
                attn_mask = attn_mask[:, -key_len:]
            pad_mask = (attn_mask == 0).unsqueeze(1).unsqueeze(2)
            attn_bias = torch.zeros((batch_size, 1, seq_len, key_len), device=x.device, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(pad_mask, torch.finfo(q.dtype).min)
            if causal_needed:
                q_pos = torch.arange(past_len, past_len + seq_len, device=x.device).view(seq_len, 1)
                k_pos = torch.arange(key_len, device=x.device).view(1, key_len)
                causal_mask = k_pos > q_pos
                attn_bias = attn_bias.masked_fill(
                    causal_mask.view(1, 1, seq_len, key_len),
                    torch.finfo(q.dtype).min,
                )
                causal_needed = False

        y = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=causal_needed,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        y = self.resid_drop(self.out_proj(y))
        return y, present


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_len: int,
        mlp_ratio: float = 8 / 3,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
        qk_norm: bool = True,
        residual_scale: float = 1.0,
        rope_theta: float = 10000.0,
        rope_scale: float = 1.0,
        rope_scaling_type: str = "linear",
    ) -> None:
        super().__init__()
        hidden_dim = int(mlp_ratio * d_model)
        hidden_dim = 256 * math.ceil(hidden_dim / 256)
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
            dropout,
            max_len,
            n_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
            rope_theta=rope_theta,
            rope_scale=rope_scale,
            rope_scaling_type=rope_scaling_type,
        )
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, hidden_dim, dropout)
        self.attn_res_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.mlp_res_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present = self.attn(
            self.ln1(x),
            attn_mask=attn_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        x = x + self.attn_res_scale * attn_out
        x = x + self.mlp_res_scale * self.mlp(self.ln2(x))
        return x, present


class GPTScratch(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int | None,
        max_len: int,
        dropout: float,
        qk_norm: bool,
        residual_scale: float,
        rope_theta: float,
        rope_scale: float,
        rope_scaling_type: str,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.n_layers = n_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.emb_norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    max_len,
                    mlp_ratio=8 / 3,
                    dropout=dropout,
                    n_kv_heads=n_kv_heads,
                    qk_norm=qk_norm,
                    residual_scale=residual_scale,
                    rope_theta=rope_theta,
                    rope_scale=rope_scale,
                    rope_scaling_type=rope_scaling_type,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init)
        self._scale_residual_projections()
        self.head.weight = self.tok_emb.weight

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        scale = 0.02 / math.sqrt(2 * self.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=scale)
            nn.init.normal_(block.mlp.w3.weight, mean=0.0, std=scale)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        _, seq_len = x.shape
        if past_kv is None and seq_len > self.max_len:
            x = x[:, -self.max_len :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_len :]

        h = self.drop(self.emb_norm(self.tok_emb(x)))
        presents: list[tuple[torch.Tensor, torch.Tensor] | None] | None = [] if use_cache else None
        if past_kv is None:
            past_kv = [None] * len(self.blocks)

        for block, layer_past in zip(self.blocks, past_kv):
            if self.gradient_checkpointing and self.training and not use_cache:
                h, present = checkpoint.checkpoint(
                    block,
                    h,
                    attention_mask,
                    layer_past,
                    False,
                    use_reentrant=False,
                )
            else:
                h, present = block(h, attn_mask=attention_mask, past_kv=layer_past, use_cache=use_cache)
            if use_cache and presents is not None:
                presents.append(present)

        logits = self.head(self.ln_f(h))
        if use_cache and presents is not None:
            return logits, presents
        return logits


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_count: int
    prompt_token_count: int
    elapsed_seconds: float
    device: str


class ManukaGenerator:
    def __init__(self, model_dir: str | Path, device: str | None = None) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        with open(self.model_dir / "config.json", encoding="utf-8") as handle:
            self.cfg: dict[str, Any] = json.load(handle)

        requested_device = device or "auto"
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected, but torch.cuda.is_available() is false.")
        self.device = torch.device(requested_device)

        tokenizer_name = self.cfg.get("tokenizer_model", "spm16k.model")
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_dir / tokenizer_name))

        specials = self.cfg.get("special_tokens", {})
        self.pad = self._special_id("<pad>", "PAD", specials, self.sp.pad_id())
        self.bos = self._special_id("<s>", "BOS", specials, self.sp.bos_id())
        self.sep = self._special_id("<sep>", "SEP", specials, self.sp.piece_to_id("<sep>"))
        self.eos = self._special_id("</s>", "EOS", specials, self.sp.eos_id())
        self.unk = self._special_id("<unk>", "UNK", specials, self.sp.unk_id())
        self.max_len = int(self.cfg["max_len"])
        self.generation_defaults = self.cfg.get("generation", {})

        self.model = GPTScratch(
            vocab_size=int(self.cfg["vocab_size"]),
            d_model=int(self.cfg["d_model"]),
            n_layers=int(self.cfg["n_layers"]),
            n_heads=int(self.cfg["n_heads"]),
            n_kv_heads=int(self.cfg.get("n_kv_heads", self.cfg["n_heads"])),
            max_len=self.max_len,
            dropout=float(self.cfg.get("dropout", 0.0)),
            qk_norm=bool(self.cfg.get("qk_norm", True)),
            residual_scale=float(self.cfg.get("residual_scale", 1.0)),
            rope_theta=float(self.cfg.get("rope_theta", 10000.0)),
            rope_scale=float(self.cfg.get("rope_scale", 1.0)),
            rope_scaling_type=str(self.cfg.get("rope_scaling_type", "linear")),
            gradient_checkpointing=False,
        ).to(self.device)
        self.model.eval()

        state_dict = self._load_state_dict(self.model_dir / "model.pt")
        self.model.load_state_dict(state_dict, strict=True)

    def _special_id(self, piece: str, key: str, specials: dict[str, int], detected_id: int) -> int:
        if detected_id is not None and detected_id >= 0:
            return int(detected_id)
        return int(specials[key])

    def _load_state_dict(self, path: Path) -> dict[str, torch.Tensor]:
        if path.exists() and path.stat().st_size <= 1024:
            checkpoint_header = path.read_text(encoding="utf-8", errors="ignore")
            if checkpoint_header.startswith(LFS_POINTER_PREFIX):
                raise RuntimeError(
                    f"{path} is a Git LFS pointer, not the real checkpoint. "
                    "Run `git lfs pull` or redeploy after enabling Git LFS for this file."
                )

        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            loaded = torch.load(path, map_location="cpu")
        except pickle.UnpicklingError:
            trusted_checkpoint = (
                Path(__file__).resolve().parent.parent / "model" / "manuka-model-0527" / "model.pt"
            ).resolve()
            if path.resolve() != trusted_checkpoint:
                raise
            loaded = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(loaded, dict):
            for key in ("model_state_dict", "state_dict"):
                nested = loaded.get(key)
                if isinstance(nested, dict):
                    loaded = nested
                    break
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected a state dict in {path}")

        if any(str(key).startswith("_orig_mod.") for key in loaded):
            loaded = {str(key).removeprefix("_orig_mod."): value for key, value in loaded.items()}
        return loaded

    def encode_text(self, value: str) -> list[int]:
        return self.sp.encode(clean_text(value), out_type=int)

    def decode_text(self, ids: list[int]) -> str:
        drop = {self.pad, self.bos, self.sep, self.eos, self.unk}
        return self.sp.decode([int(token_id) for token_id in ids if int(token_id) not in drop])

    def banned_tokens_for_no_repeat_ngram(self, generated: list[int], n: int) -> set[int]:
        if n <= 0 or len(generated) < n - 1:
            return set()
        prefix = tuple(generated[-(n - 1) :]) if n > 1 else tuple()
        banned = set()
        for index in range(0, len(generated) - n + 1):
            if tuple(generated[index : index + n - 1]) == prefix:
                banned.add(generated[index + n - 1])
        return banned

    def apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: list[int],
        penalty: float = 1.0,
    ) -> torch.Tensor:
        if penalty <= 1.0:
            return logits
        for token_id in set(generated):
            if 0 <= token_id < logits.numel():
                logits[token_id] = logits[token_id] * penalty if logits[token_id] < 0 else logits[token_id] / penalty
        return logits

    @torch.no_grad()
    def sample_next_token(
        self,
        logits_row: torch.Tensor,
        generated: list[int],
        *,
        top_p: float = 0.92,
        top_k: int = 80,
        temperature: float = 0.8,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 4,
        min_new_tokens: int = 12,
        rng: torch.Generator | None = None,
        **_: Any,
    ) -> int:
        logits = logits_row.float().clone()
        for token_id in (self.pad, self.bos, self.sep, self.unk):
            if token_id is not None and token_id >= 0:
                logits[token_id] = -float("inf")
        if len(generated) < min_new_tokens and self.eos >= 0:
            logits[self.eos] = -float("inf")
        for token_id in self.banned_tokens_for_no_repeat_ngram(generated, int(no_repeat_ngram_size)):
            if 0 <= token_id < logits.numel():
                logits[token_id] = -float("inf")

        logits = self.apply_repetition_penalty(logits, generated, float(repetition_penalty))
        logits = logits / max(1e-6, float(temperature))

        if top_k and top_k > 0 and top_k < logits.numel():
            kth = torch.topk(logits, int(top_k)).values[-1]
            logits[logits < kth] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or probs.sum() <= 0:
            finite = torch.isfinite(logits)
            if finite.any():
                return int(torch.argmax(logits).item())
            return int(self.eos)

        top_p = float(top_p)
        if 0 < top_p < 1:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > top_p
            remove[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(remove, 0)
            if sorted_probs.sum() <= 0 or torch.isnan(sorted_probs.sum()):
                return int(torch.argmax(probs).item())
            sorted_probs = sorted_probs / sorted_probs.sum()
            return int(sorted_idx[torch.multinomial(sorted_probs, 1, generator=rng)].item())

        return int(torch.multinomial(probs, 1, generator=rng).item())

    @torch.no_grad()
    def generate_reply(
        self,
        prompt: str,
        *,
        max_ctx: int | None = None,
        max_new_tokens: int | None = None,
        seed: int | None = None,
        **sampling_kwargs: Any,
    ) -> GenerationResult:
        started_at = time.perf_counter()
        max_ctx = max(4, min(int(max_ctx or self.max_len), self.max_len))
        default_max_new = int(self.generation_defaults.get("max_new_tokens", 96))
        max_new_tokens = max(1, min(int(max_new_tokens or default_max_new), max_ctx - 3))
        prompt_budget = max(1, max_ctx - max_new_tokens - 2)
        prompt_piece_ids = self.encode_text(prompt)[-prompt_budget:]
        prompt_ids = [self.bos] + prompt_piece_ids + [self.sep]

        rng = None
        if seed is not None:
            rng = torch.Generator(device=self.device)
            rng.manual_seed(int(seed))

        x = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        attn = torch.ones_like(x)
        logits, past_kv = self.model(x, attention_mask=attn, use_cache=True)
        generated: list[int] = []
        sampling = {**self.generation_defaults, **sampling_kwargs, "rng": rng}

        for _ in range(max_new_tokens):
            next_token = self.sample_next_token(logits[0, -1], generated, **sampling)
            generated.append(next_token)
            if next_token == self.eos:
                break
            x = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            logits, past_kv = self.model(x, past_kv=past_kv, use_cache=True)

        if self.eos in generated:
            generated = generated[: generated.index(self.eos)]
        elapsed = time.perf_counter() - started_at
        return GenerationResult(
            text=self.decode_text(generated).strip(),
            token_count=len(generated),
            prompt_token_count=len(prompt_piece_ids),
            elapsed_seconds=elapsed,
            device=str(self.device),
        )

    def info(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "vocab_size": int(self.cfg["vocab_size"]),
            "d_model": int(self.cfg["d_model"]),
            "n_layers": int(self.cfg["n_layers"]),
            "n_heads": int(self.cfg["n_heads"]),
            "n_kv_heads": int(self.cfg.get("n_kv_heads", self.cfg["n_heads"])),
            "max_len": self.max_len,
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
        }
