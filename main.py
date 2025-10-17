# main.py
from __future__ import annotations
"""
FastAPI entrypoint for Chatbot RAG — auto-build FAISS on first request if missing.

Env (optional)
- AUTO_BUILD_INDEX=1       (default) build FAISS if missing at first request
- WARMUP_EMB=0|1           (default 0) cache all-MiniLM-L6-v2
- LLM_AUTOSTART=0|1        (default 0) start provider on first request
- LLM_PROVIDER=groq|ollama (when LLM_AUTOSTART=1)
- CORS_ORIGINS="http://localhost:5173,https://example.com"
"""

import os
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, Callable, Optional, List
from fastapi.middleware.cors import CORSMiddleware
from api.runtime_llm import router as llm_router, _start_provider

# Optional .env
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(override=False)
except Exception:
    pass

import orjson
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from config.settings import settings

# ---- Import API router (fail fast with fallback)
try:
    from api.main import api_router as _router
except Exception as e1:
    try:
        from api.main import router as _router
    except Exception as e2:
        raise RuntimeError(
            "Failed to import API router from api/main.py "
            "(looked for `api_router` then `router`). "
            f"Errors: {e1!r} | {e2!r}"
        )
api_router = _router

# ---- LLM starter (optional)
try:
    from api.runtime_llm import _start_provider
    _HAS_RUNTIME_LLM = True
except Exception:
    _HAS_RUNTIME_LLM = False

# ---- Pipeline import (robust: supports both module layouts)
_HAS_PIPELINE = False
_run_faiss: Optional[Callable[..., Dict[str, Any]]] = None
try:
    # Your earlier file name
    from rag.pipeline_faiss import run_rag_pipeline_with_faiss as _run_faiss  # type: ignore
    _HAS_PIPELINE = True
except Exception:
    try:
        # If you moved it to pipeline/rag_pipeline.py
        from pipeline.rag_pipeline import run_rag_pipeline_with_faiss as _run_faiss  # type: ignore
        _HAS_PIPELINE = True
    except Exception:
        _HAS_PIPELINE = False

# ---------- Logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("chatbot_rag.main")

# ---------- ORJSON (pretty/safe) ----------


def _orjson_dumps_pretty(v: Any) -> bytes:
    return orjson.dumps(
        v,
        option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY,
        default=str,
    )


class ORJSONResponsePretty(ORJSONResponse):
    def render(self, content: Any) -> bytes:
        return _orjson_dumps_pretty(content)


# ---------- App ----------
APP_NAME = os.getenv("APP_NAME", "Chatbot RAG")
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")

# First-request warmup state
_WARMED_UP = False
_WARMUP_LOCK = asyncio.Lock()

SUPPORTED_EXTS: List[str] = ['.txt', '.json', '.docx', '.pdf', '.doc']


def _ensure_dirs() -> None:
    for p in (
        settings.documents_path,
        settings.output_path,
        settings.index_path,
        settings.runtime_path,
    ):
        Path(p).mkdir(parents=True, exist_ok=True)


def _faiss_all_files_present() -> bool:
    return all([
        Path(settings.faiss_index_path).exists(),
        Path(settings.idmap_path).exists(),
        Path(settings.metadata_path).exists(),
    ])


def _docs_exist(doc_dir: Path) -> bool:
    if not Path(doc_dir).exists():
        return False
    for p in Path(doc_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and p.stat().st_size > 0:
            return True
    return False


def _build_faiss_if_missing() -> Dict[str, Any]:
    """
    If FAISS files are missing and AUTO_BUILD_INDEX=1, build them now
    using your pipeline runner.
    """
    if _faiss_all_files_present():
        return {"skipped": True, "reason": "index already present"}

    if os.getenv("AUTO_BUILD_INDEX", "1") != "1":
        return {"ok": False, "error": "AUTO_BUILD_INDEX=0 and index missing"}

    if not _HAS_PIPELINE or _run_faiss is None:
        return {"ok": False, "error": "pipeline module not importable (rag.pipeline_faiss or pipeline.rag_pipeline)"}

    # Require at least one document to index
    if not _docs_exist(settings.documents_path):
        return {
            "ok": False,
            "error": f"No source documents in {settings.documents_path}. "
                     f"Add files with one of {SUPPORTED_EXTS} and retry."
        }

    try:
        log.info("FAISS index missing — building via pipeline (this can take a bit)…")
        res = _run_faiss(rebuild_index=True)
        if res.get("success"):
            log.info("FAISS build done: %s",
                     {k: res.get(k) for k in ("faiss_index_path", "idmap_path", "metadata_path")})
            return {
                "ok": True,
                "stats": res.get("stats", {}),
                "paths": {
                    "faiss_index_path": res.get("faiss_index_path"),
                    "idmap_path": res.get("idmap_path"),
                    "metadata_path": res.get("metadata_path"),
                },
            }
        return {"ok": False, "error": res.get("error", "unknown error from pipeline")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _do_warmup_first_request() -> Dict[str, Any]:
    """
    One-time heavy work on first request:
      1) Ensure/Build FAISS index
      2) (optional) Cache sentence-transformers
      3) (optional) Start LLM (Groq/Ollama)
    """
    res: Dict[str, Any] = {}

    # 1) FAISS
    res["faiss"] = _build_faiss_if_missing()

    # 2) Embedding cache
    if os.getenv("WARMUP_EMB", "0") == "1":
        try:
            from sentence_transformers import SentenceTransformer
            SentenceTransformer("all-MiniLM-L6-v2")
            res["embeddings"] = {"ok": True, "model": "all-MiniLM-L6-v2"}
        except Exception as e:
            res["embeddings"] = {"ok": False, "error": str(e)}
    else:
        res["embeddings"] = {"skipped": True}

    # 3) LLM auto-start
    if os.getenv("LLM_AUTOSTART", "0") == "1" and _HAS_RUNTIME_LLM:
        provider = (os.getenv("LLM_PROVIDER", "groq") or "groq").lower()
        if provider not in ("groq", "ollama"):
            res["llm"] = {"ok": False,
                          "error": f"Unsupported provider: {provider}"}
        else:
            try:
                _start_provider(provider)
                res["llm"] = {"ok": True, "provider": provider}
            except Exception as e:
                res["llm"] = {"ok": False, "error": str(e)}
    else:
        res["llm"] = {"skipped": True}

    return res

# ---------- Lifespan ----------


@asynccontextmanager
async def lifespan(app: FastAPI):
    prov = (os.getenv("LLM_PROVIDER") or "").lower().strip()
    if prov in ("groq", "ollama"):
        try:
            _start_provider(prov)
            print(f"[startup] LLM provider started: {prov}")
        except Exception as e:
            print(f"[startup] Could not start LLM provider '{prov}': {e}")

    _ensure_dirs()
    
    log.info("Data dirs: %s", [str(p) for p in (
        settings.documents_path, settings.output_path, settings.index_path, settings.runtime_path
    )])

    if Path(settings.faiss_index_path).exists():
        log.info("FAISS index present at %s.", settings.faiss_index_path)
    else:
        log.info("No FAISS index yet at %s.", settings.faiss_index_path)

    yield
    log.info("Shutting down app.")

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    default_response_class=ORJSONResponsePretty,
    lifespan=lifespan,
)

# ---------- CORS ----------
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
origins_env = os.getenv("CORS_ORIGINS", "").strip()
origins = [o.strip() for o in origins_env.split(",")
           if o.strip()] or default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,                         # <= utilise bien 'origins'
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):5173$",
    allow_credentials=True,                        # cookies cross-origin
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Admin-Key",
        "X-Requested-With",
    ],
    expose_headers=["set-cookie"],
)
# ---------- First-request warmup (includes FAISS auto-build if missing) ----------


@app.middleware("http")
async def first_request_warmup(request: Request, call_next):
    global _WARMED_UP
    if not _WARMED_UP:
        async with _WARMUP_LOCK:
            if not _WARMED_UP:
                log.info(
                    "Running first-request warmup (FAISS ensure/build, optional emb+LLM)…")
                res = await _do_warmup_first_request()
                log.info("Warmup result: %s", res)
                _WARMED_UP = True
    return await call_next(request)

# ---------- API Router ----------
app.include_router(api_router, prefix="/api")

# ---------- Meta ----------


@app.get("/", tags=["meta"])
def root() -> Dict[str, Any]:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "docs": ["/docs", "/redoc"],
        "api_base": "/api",
        "warmed_up": _WARMED_UP,
    }


@app.get("/health", tags=["meta"])
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["meta"])
def ready() -> Dict[str, Any]:
    return {
        "status": "ready",
        "faiss_index_present": _faiss_all_files_present(),
        "dirs": {
            "documents": str(settings.documents_path),
            "processed": str(settings.output_path),
            "index": str(settings.index_path),
            "runtime": str(settings.runtime_path),
        },
        "warmed_up": _WARMED_UP,
    }


@app.get("/status/faiss", tags=["meta"])
def status_faiss() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "present": _faiss_all_files_present(),
        "paths": {
            "faiss_index_path": str(settings.faiss_index_path),
            "idmap_path": str(settings.idmap_path),
            "metadata_path": str(settings.metadata_path),
        },
    }
    if Path(settings.faiss_index_path).exists():
        try:
            import faiss  # type: ignore
            index = faiss.read_index(str(settings.faiss_index_path))
            out.update({"readable": True, "dimension": int(index.d)})
        except Exception as e:
            out.update({"readable": False, "error": str(e)})
    return out


# ---------- Entrypoint (dev) ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "1") == "1",
        workers=int(os.getenv("WORKERS", "1")),
    )
