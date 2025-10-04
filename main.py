from __future__ import annotations
# RAGSettings instance (with runtime_path, etc.)
from config.settings import settings

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict

import orjson
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

# ---------- Logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("chatbot_rag.main")

# ---------- Settings ----------

# ---------- ORJSON (safe/pretty) ----------


def _orjson_dumps_pretty(v: Any) -> bytes:
    return orjson.dumps(
        v,
        option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY,
        default=str,  # never crash on unknown types
    )


class ORJSONResponsePretty(ORJSONResponse):
    def render(self, content: Any) -> bytes:
        return _orjson_dumps_pretty(content)


# ---------- Import API router (fail fast if missing) ----------
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

# ---------- Lifespan ----------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure directories exist
    for p in (settings.documents_path, settings.output_path, settings.index_path, settings.runtime_path):
        Path(p).mkdir(parents=True, exist_ok=True)
    log.info("Data dirs OK: %s", [str(p) for p in (
        settings.documents_path, settings.output_path, settings.index_path, settings.runtime_path)])

    # Optional FAISS check
    idx = settings.faiss_index_path
    if Path(idx).exists():
        try:
            import faiss  # type: ignore
            _ = faiss.read_index(str(idx))
            log.info("FAISS readable at %s", idx)
        except Exception as e:
            log.warning("FAISS read failed (%s): %s", idx, e)
    else:
        log.info("No FAISS index yet at %s.", idx)

    yield
    log.info("Shutting down app.")

# ---------- App ----------
APP_NAME = os.getenv("APP_NAME", "Chatbot RAG")
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    default_response_class=ORJSONResponsePretty,
    lifespan=lifespan,
)

# CORS
# --- CORS ---

# 1) default to your Vite origin in dev
default_origins = ["http://localhost:5173"]

# 2) allow override via env (comma separated)
origins_env = os.getenv("CORS_ORIGINS", "").strip()
if origins_env:
    origins = [o.strip() for o in origins_env.split(",") if o.strip()]
else:
    origins = default_origins

# 3) DO NOT use "*" with allow_credentials=True
#    FastAPI/Starlette will set the explicit origin automatically.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

# ---------- Meta ----------


@app.get("/", tags=["meta"])
def root() -> Dict[str, Any]:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "docs": ["/docs", "/redoc"],
        "api_base": "/api",
    }


@app.get("/health", tags=["meta"])
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["meta"])
def ready() -> Dict[str, Any]:
    return {
        "status": "ready",
        "faiss_index_present": Path(settings.faiss_index_path).exists(),
        "dirs": {
            "documents": str(settings.documents_path),
            "processed": str(settings.output_path),
            "index": str(settings.index_path),
            "runtime": str(settings.runtime_path),
        },
    }


@app.get("/status/faiss", tags=["meta"])
def status_faiss() -> Dict[str, Any]:
    idx = settings.faiss_index_path
    out: Dict[str, Any] = {"present": Path(idx).exists(), "path": str(idx)}
    if out["present"]:
        try:
            import faiss  # type: ignore
            index = faiss.read_index(str(idx))
            out.update({"readable": True, "dimension": int(index.d)})
        except Exception as e:
            out.update({"readable": False, "error": str(e)})
    return out


# ---------- Entrypoint ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "1") == "1",
        workers=int(os.getenv("WORKERS", "1")),
    )
