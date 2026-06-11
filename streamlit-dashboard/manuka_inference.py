from __future__ import annotations

import importlib.util
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch


CORE_BACKEND_PATH = Path(__file__).resolve().parent.parent / "streamlit-dashboard" / "manuka_inference.py"
DEPLOY_MODEL_DIR = Path("/mount/src/manuka-chat/model/manuka-model-0611")
LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "manuka-model-0611"
SENTENCE_END_PATTERN = r'[.!?\u3002\uff01\uff1f\u2026~]+[)\]\}"\']*$'
DEFAULT_GENERATION = {
    "max_new_tokens": 192,
    "min_new_tokens": 24,
    "temperature": 0.72,
    "top_p": 0.88,
    "top_k": 60,
    "repetition_penalty": 1.12,
    "no_repeat_ngram_size": 4,
    "eos_requires_sentence_end": True,
    "sentence_end_pattern": SENTENCE_END_PATTERN,
}


def _load_core_backend():
    spec = importlib.util.spec_from_file_location("_manuka_inference_core", CORE_BACKEND_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load core backend from {CORE_BACKEND_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_core = _load_core_backend()
GenerationResult = _core.GenerationResult
GPTScratch = _core.GPTScratch
LFS_POINTER_PREFIX = _core.LFS_POINTER_PREFIX
clean_text = _core.clean_text


class ManukaGenerator(_core.ManukaGenerator):
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

        tokenizer_name = self.cfg.get("tokenizer_model", "spm32k.model")
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_dir / tokenizer_name))

        specials = self.cfg.get("special_tokens", {})
        self.pad = self._special_id("<pad>", "PAD", specials, self.sp.pad_id())
        self.bos = self._special_id("<s>", "BOS", specials, self.sp.bos_id())
        self.sep = self._special_id("<sep>", "SEP", specials, self.sp.piece_to_id("<sep>"))
        self.eos = self._special_id("</s>", "EOS", specials, self.sp.eos_id())
        self.unk = self._special_id("<unk>", "UNK", specials, self.sp.unk_id())
        if self.sep < 0:
            raise RuntimeError("Tokenizer/config must define a valid <sep> token.")
        if self.sp.get_piece_size() != int(self.cfg["vocab_size"]):
            raise RuntimeError(
                f"Tokenizer vocab size {self.sp.get_piece_size()} does not match config vocab_size {self.cfg['vocab_size']}."
            )

        self.max_len = int(self.cfg["max_len"])
        self.generation_defaults = {
            **DEFAULT_GENERATION,
            **dict(self.cfg.get("generation", {})),
        }
        sentence_pattern = str(self.generation_defaults.get("sentence_end_pattern", SENTENCE_END_PATTERN))
        self.sentence_end_regex = re.compile(sentence_pattern)

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

    def _is_trusted_checkpoint(self, path: Path) -> bool:
        trusted_paths = {
            (DEPLOY_MODEL_DIR / "model.pt").resolve(),
            (LOCAL_MODEL_DIR / "model.pt").resolve(),
        }
        return path.resolve() in trusted_paths

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
            if not self._is_trusted_checkpoint(path):
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

    def can_stop_on_eos(
        self,
        generated: list[int],
        min_new_tokens: int,
        eos_requires_sentence_end: bool,
    ) -> bool:
        if len(generated) < min_new_tokens:
            return False
        if not eos_requires_sentence_end:
            return True
        text = self.decode_text(generated).strip()
        return bool(self.sentence_end_regex.search(text))

    @torch.no_grad()
    def sample_next_token(
        self,
        logits_row: torch.Tensor,
        generated: list[int],
        *,
        top_p: float = 0.88,
        top_k: int = 60,
        temperature: float = 0.72,
        repetition_penalty: float = 1.12,
        no_repeat_ngram_size: int = 4,
        min_new_tokens: int = 24,
        allow_eos: bool = False,
        rng: torch.Generator | None = None,
        **_: Any,
    ) -> int:
        logits = logits_row.float().clone()
        for token_id in (self.pad, self.bos, self.sep, self.unk):
            if token_id is not None and token_id >= 0:
                logits[token_id] = -float("inf")
        if (not allow_eos or len(generated) < min_new_tokens) and self.eos >= 0:
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
        default_max_new = int(self.generation_defaults.get("max_new_tokens", DEFAULT_GENERATION["max_new_tokens"]))
        max_new_tokens = max(1, min(int(max_new_tokens or default_max_new), max_ctx - 2))
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
        sampling["min_new_tokens"] = min(
            int(sampling.get("min_new_tokens", DEFAULT_GENERATION["min_new_tokens"])),
            max_new_tokens,
        )
        eos_requires_sentence_end = bool(
            sampling.get("eos_requires_sentence_end", DEFAULT_GENERATION["eos_requires_sentence_end"])
        )

        for _ in range(max_new_tokens):
            sampling["allow_eos"] = self.can_stop_on_eos(
                generated,
                int(sampling["min_new_tokens"]),
                eos_requires_sentence_end,
            )
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
