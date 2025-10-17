# api/admin.py
from __future__ import annotations

import os
import hmac
import hashlib
import json
import base64
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
from pipeline.rag_pipeline import run_rag_pipeline_with_faiss

from fastapi import APIRouter, Header, HTTPException, Request , Query

# project settings (paths for flush-index)
from config.settings import settings

router = APIRouter()

# ---------- session helpers (mirror auth.py; ideally factor to utils/session.py) ----------
_SESSION_COOKIE = "session"
_SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret-change-me").encode()


def _sign(b: bytes) -> bytes:
    return hmac.new(_SESSION_SECRET, b, hashlib.sha256).digest()


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64u(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _parse_session(token: Optional[str]) -> Optional[dict]:
    if not token or "." not in token:
        return None
    body, sig = token.split(".", 2)
    good = _b64u(_sign(body.encode("ascii")))
    if not hmac.compare_digest(sig, good):
        return None
    try:
        return json.loads(_unb64u(body))
    except Exception:
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------- authorization ----------
def _require_admin(request: Request, x_admin_key: Optional[str]) -> dict:
    sess = _parse_session(request.cookies.get(_SESSION_COOKIE))
    # 1) prefer cookie role
    if sess and sess.get("role") == "admin":
        return sess
    # 2) fallback header key if configured
    want = os.getenv("ADMIN_API_KEY")
    if want and x_admin_key == want:
        return {"user_id": "admin-key", "email": "admin-key@local", "role": "admin"}
    # otherwise reject
    raise HTTPException(status_code=403, detail="Admin privileges required")


# ---------- safe Supabase import ----------
def _get_supabase_safe():
    """
    Importe et renvoie le client Supabase ou None si indisponible.
    Évite que /admin/stats lève une 500 en cas de problème d'import/connexion.
    """
    try:
        # ajuste si ton util est ailleurs (ex: from api.database import get_supabase)
        from .database import get_supabase  # type: ignore
        return get_supabase()
    except Exception:
        return None


def _current_user(request: Request) -> Optional[dict]:
    """Return {'user_id','email','role'} or None from signed cookie."""
    return _parse_session(request.cookies.get(_SESSION_COOKIE))

# ---------- endpoints ----------
class QuestionRow(BaseModel):
    id: str
    user_id: str | None = None
    email: str | None = None
    question: str
    workspace: str | None = None
    created_at: datetime


@router.get(
    "/admin/ask/questions",
    response_model=list[QuestionRow],
    summary="Admin: list all questions",
    tags=["admin"],
)
def admin_list_questions(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    email: str | None = None,
    user_id: str | None = None,
    workspace: str | None = None,
    start: str | None = None,
    end: str | None = None,
):
    # simple admin check via cookie (role=admin)
    user = _current_user(request)
    if not (user and user.get("role") == "admin"):
        raise HTTPException(
            status_code=403, detail="Admin privileges required")

    sb = _get_supabase_safe()
    q = sb.table("chat_questions").select("*")
    if email:
        q = q.eq("email", email)
    if user_id:
        q = q.eq("user_id", user_id)
    if workspace:
        q = q.eq("workspace", workspace)
    if start:
        q = q.gte("created_at", f"{start}T00:00:00Z")
    if end:
        q = q.lt("created_at", f"{end}T23:59:59Z")
    q = q.order("created_at", desc=True).range(offset, offset + limit - 1)
    res = q.execute()
    return res.data or []



@router.get("/ping", summary="Admin ping")
def admin_ping(request: Request, x_admin_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    _require_admin(request, x_admin_key)
    return {"ok": True, "msg": "Welcome to admin dashboard"}


@router.get("/stats", summary="Admin stats")
def admin_stats(request: Request, x_admin_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Renvoie des métriques du dashboard. Résilient : renvoie des valeurs par défaut
    si la base n'est pas accessible plutôt que 500.
    """
    # _require_admin(request, x_admin_key)

    # Fenêtres de temps
    now = _utc_now()
    t24h = now - timedelta(hours=24)
    t48h = now - timedelta(hours=48)
    t7d = now - timedelta(days=7)

    # Valeurs par défaut (fallback si pas de DB/tables)
    out: Dict[str, Any] = {
        "conversations_24h": 0,
        "delta_24h": 0.0,
        "documents_total": 0,
        "documents_7d": 0,
        "retrieval_topk_avg": None,
        "similarity_avg": None,
        "latency_ms_avg": None,
        "safe_answer_rate": None,
        "total_messages_24h": 0,             # NEW
        "unique_customers_24h": 0,           # NEW
        "avg_session_time_seconds_24h": 0.0  # NEW
    }

    supabase = _get_supabase_safe()
    if supabase is None:
        # DB indispo => retourne les defaults (200 OK)
        return out

    # Toutes les requêtes sont protégées par try/except pour rester "fail-safe"
    try:
        # --- Conversations (sessions) ---
        conv_24h = supabase.table("sessions").select(
            "id").gte("started_at", t24h.isoformat()).execute()
        conversations_24h = len(conv_24h.data or [])
        conv_prev = (
            supabase.table("sessions")
            .select("id")
            .gte("started_at", t48h.isoformat())
            .lt("started_at", t24h.isoformat())
            .execute()
        )
        conversations_prev_24h = len(conv_prev.data or [])
        if conversations_prev_24h == 0:
            delta_24h = 100.0 if conversations_24h > 0 else 0.0
        else:
            delta_24h = round(
                ((conversations_24h - conversations_prev_24h) / conversations_prev_24h) * 100.0, 1)

        out["conversations_24h"] = conversations_24h
        out["delta_24h"] = delta_24h
    except Exception:
        pass

    # --- Documents (si table "documents") ---
    try:
        docs_total = supabase.table("documents").select(
            "id", count="exact").execute()
        out["documents_total"] = docs_total.count or 0
        docs_7d = supabase.table("documents").select(
            "id").gte("created_at", t7d.isoformat()).execute()
        out["documents_7d"] = len(docs_7d.data or [])
    except Exception:
        pass  # table absente => on laisse les defaults

    # --- Total messages (24h) (table "messages") ---
    try:
        msgs_24h = supabase.table("messages").select(
            "id").gte("created_at", t24h.isoformat()).execute()
        out["total_messages_24h"] = len(msgs_24h.data or [])
    except Exception:
        pass

    # --- Unique customers (24h) via sessions.user_id ---
    try:
        users_24h = supabase.table("sessions").select(
            "user_id").gte("started_at", t24h.isoformat()).execute()
        out["unique_customers_24h"] = len(
            {r["user_id"] for r in (users_24h.data or []) if r.get("user_id")})
    except Exception:
        pass

    # --- Durée moyenne des sessions (24h) ---
    try:
        dur_rows = (
            supabase.table("sessions")
            .select("started_at,ended_at")
            .gte("started_at", t24h.isoformat())
            .execute()
        )
        total_secs = 0.0
        n = 0
        for r in dur_rows.data or []:
            try:
                s_raw = r.get("started_at")
                e_raw = r.get("ended_at")
                if not s_raw or not e_raw:
                    continue
                s = datetime.fromisoformat(str(s_raw).replace("Z", "+00:00"))
                e = datetime.fromisoformat(str(e_raw).replace("Z", "+00:00"))
                if e > s:
                    total_secs += (e - s).total_seconds()
                    n += 1
            except Exception:
                pass
        out["avg_session_time_seconds_24h"] = round(
            total_secs / n, 1) if n > 0 else 0.0
    except Exception:
        pass

    # --- Qualité (si table "answers") ---
    try:
        lat = supabase.table("answers").select("latency_ms").gte(
            "created_at", t24h.isoformat()).execute()
        lat_vals = [row["latency_ms"]
                    for row in (lat.data or []) if "latency_ms" in row]
        out["latency_ms_avg"] = round(
            sum(lat_vals) / len(lat_vals), 1) if lat_vals else None
    except Exception:
        pass

    try:
        sim = supabase.table("answers").select("similarity").gte(
            "created_at", t24h.isoformat()).execute()
        sim_vals = [row["similarity"]
                    for row in (sim.data or []) if row.get("similarity") is not None]
        out["similarity_avg"] = round(
            sum(sim_vals) / len(sim_vals), 3) if sim_vals else None
    except Exception:
        pass

    try:
        topk = supabase.table("answers").select("top_k").gte(
            "created_at", t24h.isoformat()).execute()
        topk_vals = [row["top_k"]
                     for row in (topk.data or []) if row.get("top_k") is not None]
        out["retrieval_topk_avg"] = round(
            sum(topk_vals) / len(topk_vals), 1) if topk_vals else None
    except Exception:
        pass

    try:
        safe = supabase.table("answers").select("safe_answer").gte(
            "created_at", t24h.isoformat()).execute()
        total_rows = len(safe.data or [])
        ok_rows = len([1 for row in (safe.data or [])
                      if row.get("safe_answer") is True])
        out["safe_answer_rate"] = round(
            100.0 * ok_rows / total_rows, 1) if total_rows > 0 else None
    except Exception:
        pass

    return out


@router.get("/config", summary="Admin config (.env)")
def admin_config(request: Request, x_admin_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    # _require_admin(request, x_admin_key)
    return {
        "INGEST_CHUNK_SIZE": int(os.getenv("INGEST_CHUNK_SIZE", "350")),
        "INGEST_OVERLAP": int(os.getenv("INGEST_OVERLAP", "120")),
        "INGEST_DIR": os.getenv("INGEST_DIR", "./data/uploads"),
        "EMBED_DIM": int(os.getenv("EMBED_DIM", "768")),
        "OLLAMA_NUM_CTX": int(os.getenv("OLLAMA_NUM_CTX", "2048")),
        "OLLAMA_NUM_PREDICT": int(os.getenv("OLLAMA_NUM_PREDICT", "256")),
        "DATA_DIR": os.getenv(
            "DATA_DIR",
            "C:/Users/lenovo/Desktop/LA_FAC/PFE-orange/ChatBot/backend/apps/data",
        ),
    }


# ---------- filesystem helpers ----------
def _safe_rm(path: Path, removed: list[str]) -> None:
    try:
        if path.is_file():
            path.unlink(missing_ok=True)
            removed.append(str(path))
        elif path.is_dir():
            # remove entire directory recursively
            import shutil
            shutil.rmtree(path, ignore_errors=True)
            removed.append(str(path) + "/")
    except Exception as e:
        # don’t block – just report what failed to delete
        removed.append(f"{path} (failed: {e})")

# ---------- endpoints ----------


@router.post("/flush-index", summary="Delete FAISS + processed artifacts (+ optional DB)")
def flush_index(
    request: Request,
    x_admin_key: Optional[str] = Header(None),
    wipe_db: bool = Query(
        False, description="Also purge DB tables used by RAG"),
) -> Dict[str, Any]:
    """
    Purge all index-related artifacts:
    - FAISS index + idmap + metadata
    - processed artifacts (chunks.json, embeddings.json, pipeline_stats.json)
    - index/runtime folders (if configured)
    Optionally purge Supabase tables when wipe_db=true.
    """
    _require_admin(request, x_admin_key)
    removed: list[str] = []

    # Files/dirs (paths depend on your settings)
    _safe_rm(Path(settings.faiss_index_path), removed)
    _safe_rm(Path(settings.idmap_path), removed)
    _safe_rm(Path(settings.metadata_path), removed)

    out_dir = Path(settings.output_path)
    _safe_rm(out_dir / "chunks.json", removed)
    _safe_rm(out_dir / "embeddings.json", removed)
    _safe_rm(out_dir / "pipeline_stats.json", removed)

    # Optional folders if you expose them in settings
    for maybe in ["index_path", "runtime_path"]:
        p = Path(getattr(settings, maybe, out_dir))
        if p and p.exists():
            _safe_rm(p, removed)

    # Defensive: remove stray *.faiss / *.index
    for root in {out_dir, Path(getattr(settings, "index_path", out_dir))}:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix in (".faiss", ".index"):
                    _safe_rm(p, removed)

    db_cleared: Dict[str, Any] = {}
    if wipe_db:
        try:
            sb = get_supabase()
            # ⚠️ Use the exact table names you actually created.
            # Examples (comment out those you don't have):
            for tbl in [
                "answers",          # KPIs per answer
                "messages",         # chat turns
                "sessions",         # chat sessions
                "chat_questions",   # optional logging of questions
                "processed_chunks",  # if you persist chunks in DB
                "embeddings",       # if you persist vectors in DB
                "faiss_meta",       # if you persist meta in DB
            ]:
                try:
                    sb.table(tbl).delete().neq(
                        "id", 0).execute()  # delete all rows
                    db_cleared[tbl] = "ok"
                except Exception as e:
                    db_cleared[tbl] = f"skip/err: {e}"
        except Exception as e:
            db_cleared["error"] = str(e)

    return {"ok": True, "removed": removed, "db": db_cleared}

def _index_artifacts_exist() -> bool:
    """Return True iff FAISS + sidecars already exist."""
    return (
        Path(settings.faiss_index_path).exists()
        and Path(settings.idmap_path).exists()
        and Path(settings.metadata_path).exists()
    )


@router.post("/reindex", summary="Rebuild FAISS index (run full pipeline)")
def reindex(
    request: Request,
    x_admin_key: Optional[str] = Header(None),
    force: bool = Query(
        False, description="Force rebuild even if index exists"),
) -> Dict[str, Any]:
    """
    Runs the full RAG pipeline (docs→chunks→embeddings→FAISS).
    - If artifacts already exist and force=false, we skip the rebuild.
    - Pass ?force=true to actually rebuild.
    """
    _require_admin(request, x_admin_key)

    if _index_artifacts_exist() and not force:
        return {
            "ok": True,
            "skipped": True,
            "reason": "FAISS artifacts already exist. Use ?force=true to rebuild.",
            "paths": {
                "faiss_index_path": str(settings.faiss_index_path),
                "idmap_path": str(settings.idmap_path),
                "metadata_path": str(settings.metadata_path),
            },
        }

    # Run pipeline. If force=True we rebuild; if False pipeline will skip if present.
    result = run_rag_pipeline_with_faiss(
        documents_path=Path(settings.documents_path),
        output_path=Path(settings.output_path),
        rebuild_index=force,
    )
    return {
        "ok": bool(result.get("success")),
        "skipped": bool(result.get("skip_build")),
        "result": result,
    }
