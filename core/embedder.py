# core/embedder.py
from __future__ import annotations
from typing import List, Dict, Any, Optional

# Use your config if available; otherwise fallback
try:
    from config.settings import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_DIMENSION
except Exception:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE = 64
    EMBEDDING_DIMENSION = 384  # default MiniLM

# One unified selector: LM Studio or Sentence-Transformers
try:
    from core.embedder_selector import get_embedder
except Exception as e:
    raise RuntimeError("Missing core.embedder_selector.get_embedder()") from e


class EmbeddingModel:
    """
    Unified embedding wrapper.
    Always call backend.embed(List[str]) -> List[List[float]]
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.backend = get_embedder()   # LMStudioEmbedder or STEmbedder
        self.embedding_dim: Optional[int] = None

        # Optional predefined dim
        try:
            if EMBEDDING_DIMENSION and EMBEDDING_DIMENSION > 0:
                self.embedding_dim = int(EMBEDDING_DIMENSION)
        except Exception:
            pass

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        out = self.backend.embed(list(texts))
        # normalize output type
        if hasattr(out, "tolist"):
            out = out.tolist()
        if len(out) != len(texts):
            raise RuntimeError(
                f"Encoder returned {len(out)} vectors for {len(texts)} texts")
        # fix or validate dimension
        if self.embedding_dim is None:
            if not out or not out[0]:
                raise RuntimeError("Embedding backend returned empty vectors")
            self.embedding_dim = len(out[0])
        else:
            for v in out:
                if len(v) != self.embedding_dim:
                    raise RuntimeError(
                        f"Inconsistent embedding dimension: got {len(v)} expected {self.embedding_dim}")
        return out

    def get_embedding_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dimension": int(self.embedding_dim or 0),
            "backend": type(self.backend).__name__,
            "available": True,
        }


class ChunkEmbedder:
    """
    Batch embedding for chunks.
    **Guarantees**: 1 chunk -> 1 embedding (no skipping).
    Embedding record uses key 'embedding' (FAISSIndexer expects this).
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = EMBEDDING_BATCH_SIZE):
        self.embedding_model = EmbeddingModel(model_name)
        try:
            self.batch_size = int(batch_size)
            if self.batch_size <= 0:
                self.batch_size = 64
        except Exception:
            self.batch_size = 64

    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not chunks:
            return {}

        print(f"Génération d'embeddings pour {len(chunks)} chunks...")

        # Diagnostics
        empty_content_count = 0
        duplicate_ids = set()
        seen_ids = set()

        result: Dict[str, Dict[str, Any]] = {}
        ids: List[int] = []
        texts: List[str] = []

        for i, ch in enumerate(chunks):
            cid = int(ch.get("id", i + 1))

            # Détecter doublons d'IDs
            if cid in seen_ids:
                duplicate_ids.add(cid)
            seen_ids.add(cid)

            txt = (ch.get("content") or "").strip()

            if not txt:
                empty_content_count += 1
                # Placeholder obligatoire pour préserver l'alignement
                txt = f"[Empty chunk - Section: {ch.get('section', 'N/A')}]"

            ids.append(cid)
            texts.append(txt)

        # Rapport de diagnostic
        if empty_content_count > 0:
            print(
                f"⚠️  {empty_content_count} chunks avec contenu vide détectés (placeholders utilisés)")
        if duplicate_ids:
            print(
                f"⚠️  {len(duplicate_ids)} IDs dupliqués détectés: {sorted(duplicate_ids)[:10]}")

        n = len(texts)
        B = self.batch_size

        for start in range(0, n, B):
            end = min(start + B, n)
            batch_texts = texts[start:end]
            vecs = self.embedding_model.embed(batch_texts)
            if len(vecs) != len(batch_texts):
                raise RuntimeError(
                    f"Batch embed mismatch: got {len(vecs)} != {len(batch_texts)}")

            for j, v in enumerate(vecs):
                cid = ids[start + j]
                ch = chunks[start + j]
                result[str(cid)] = {
                    "embedding": v,                        # <— key expected by your FAISSIndexer
                    "content": ch.get("content", ""),
                    "source": ch.get("source", ""),
                    "section": ch.get("section", "Section Principale"),
                }

        print(f"Embeddings générés avec succès pour {len(result)} chunks")
        return result

    # Optional helpers (stats/validation)

    def get_embedding_stats(self, embeddings_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        import math as _math
        if not embeddings_dict:
            return {"total_embeddings": 0, "embedding_dimension": 0, "model_info": self.embedding_model.get_embedding_info()}
        first = next(iter(embeddings_dict.values()))["embedding"]
        return {
            "total_embeddings": len(embeddings_dict),
            "embedding_dimension": len(first),
            "model_info": self.embedding_model.get_embedding_info(),
            "sample_embedding_norm": _math.sqrt(sum(x*x for x in first)),
        }

    def validate_embeddings(self, embeddings_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        import math as _math
        if not embeddings_dict:
            return {"valid": False, "error": "Aucun embedding trouvé"}
        issues = []
        valid_count = 0
        expected_dim = int(self.embedding_model.embedding_dim or 0)
        for cid, rec in embeddings_dict.items():
            vec = rec.get("embedding") or []
            if expected_dim and len(vec) != expected_dim:
                issues.append(f"{cid}: dim {len(vec)} != {expected_dim}")
                continue
            if not vec or all(x == 0 for x in vec):
                issues.append(f"{cid}: empty/null vector")
                continue
            nrm = _math.sqrt(sum(x*x for x in vec))
            if not (0.5 <= nrm <= 2.0):
                issues.append(f"{cid}: odd norm {nrm:.3f}")
            valid_count += 1
        return {
            "valid": len(issues) == 0,
            "valid_count": valid_count,
            "total_count": len(embeddings_dict),
            "issues": issues[:10],
            "success_rate": valid_count / max(1, len(embeddings_dict)),
        }
