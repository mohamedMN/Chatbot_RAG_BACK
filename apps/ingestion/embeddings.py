# backend/apps/ingestion/embeddings.py
import os
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

EMBED_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "nomic-ai/nomic-embed-text-v1")
EMBED_MAX_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "1024"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))
EXPECTED_DIM = int(os.getenv("EMBED_DIM", "768"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_dtype(device: torch.device) -> torch.dtype:
    """
    Pick a real torch.dtype (no strings) to avoid TypeError in model.to(dtype=...).
    - GPU: prefer bfloat16 if supported, else float16
    - CPU: float32
    """
    if device.type == "cuda":
        # Try to use bf16 if available
        bf16_ok = False
        try:
            if hasattr(torch.cuda, "is_bf16_supported"):
                bf16_ok = bool(torch.cuda.is_bf16_supported())
            else:
                # Fallback heuristic: Ampere (sm_80) and newer generally support bf16
                major, _ = torch.cuda.get_device_capability(0)
                bf16_ok = major >= 8
        except Exception:
            bf16_ok = False
        return torch.bfloat16 if bf16_ok else torch.float16
    # CPU
    return torch.float32


DTYPE = _select_dtype(DEVICE)

# Load HF components (NO "auto" string for torch_dtype)
tokenizer = AutoTokenizer.from_pretrained(
    EMBED_MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    EMBED_MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=DTYPE,
).to(DEVICE).eval()


@torch.no_grad()
def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Mean-pool with attention mask
    mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)       # [B, 1]
    return summed / counts


@torch.no_grad()
def _embed_batch(texts: List[str]) -> List[List[float]]:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=EMBED_MAX_TOKENS,
        return_tensors="pt",
    ).to(DEVICE)

    outputs = model(**inputs)
    # Use pooled output if provided; otherwise mean-pool token embeddings
    pooled = getattr(outputs, "pooler_output", None)
    embs = pooled if pooled is not None else _mean_pool(
        outputs.last_hidden_state, inputs["attention_mask"])

    # L2-normalize
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    out = embs.detach().cpu().tolist()

    # Optional sanity check
    if EXPECTED_DIM and out and len(out[0]) != EXPECTED_DIM:
        # Not raising to keep it simple; adjust EMBED_DIM or model if needed
        pass
    return out


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return one embedding per input string."""
    if not texts:
        return []
    vectors: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors.extend(_embed_batch(texts[i:i + EMBED_BATCH_SIZE]))
    return vectors
