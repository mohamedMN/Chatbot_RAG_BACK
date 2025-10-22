# api/admin_metrics.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
from fastapi import APIRouter, HTTPException, Request, Query

from .database import get_supabase
from .auth import _parse_session, _SESSION_COOKIE

metrics_router = APIRouter(prefix="/admin/metrics", tags=["admin"])


# ------------------------------ Period helpers ------------------------------ #
def _this_week_window(now: datetime) -> Tuple[datetime, datetime]:
    """Lundi 00:00:00 -> Dimanche 23:59:59 (UTC)"""
    dow = (now.weekday() + 7) % 7  # 0 = lundi
    start = (now - timedelta(days=dow)).replace(hour=0,
                                                minute=0, second=0, microsecond=0)
    end = (start + timedelta(days=7)) - timedelta(seconds=1)
    return start, end


def _period_window(period: str, now: datetime) -> tuple[datetime, datetime]:
    p = (period or "last_7d").lower()

    if p in ("today", "aujourd'hui", "aujourdhui"):
        # Today from 00:00:00 UTC to now
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
        return start, end

    if p in ("this_week", "week"):
        return _this_week_window(now)

    if p in ("last_7d", "7d"):
        end = now
        start = (now - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, end

    if p in ("this_month", "month"):
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        nextm = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
        end = nextm - timedelta(seconds=1)
        return start, end

    if p in ("last_30d", "30d"):  # kept for backward compatibility (not used by UI)
        end = now
        start = (now - timedelta(days=29)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, end

    # Fallback
    return _this_week_window(now)



# ------------------------------ Math helpers -------------------------------- #
def _smoothed_growth(cur: int, prev: int, alpha: int = 2, decimals: int = 0) -> float:
    """
    Growth = 100 * (cur - prev) / (prev + alpha)
    - Lisse les extrêmes, évite -100% et +inf sur petits volumes
    - cur=prev=0 => 0%
    - prev=0, cur>0 => 100%
    """
    if prev == 0 and cur == 0:
        return 0.0
    if prev == 0 and cur > 0:
        return 100.0
    g = 100.0 * (cur - prev) / (prev + alpha)
    return round(g, decimals)


def _days_span(start: datetime, end: datetime) -> List[datetime]:
    out: List[datetime] = []
    cur = start.replace(hour=0, minute=0, second=0, microsecond=0)
    last = end.replace(hour=0, minute=0, second=0, microsecond=0)
    while cur <= last:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def _labelfr(d: datetime) -> str:
    return ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"][d.weekday()]


# --------------------------------- Route ------------------------------------ #
@metrics_router.get("/timeseries")
def usage_timeseries(
    request: Request,
    period: str = Query("last_7d"),   # ⬅️ défaut changé ici
):
    # --- Auth: admin only
    sess = _parse_session(request.cookies.get(_SESSION_COOKIE))
    if not (sess and sess.get("role") == "admin"):
        raise HTTPException(
            status_code=403, detail="Admin privileges required")

    sb = get_supabase()
    now = datetime.now(timezone.utc)
    start, end = _period_window(period, now)

    # --- 1) Fetch raw rows (current period)
    sessions = (
        sb.table("sessions")
        .select("id, started_at")
        .gte("started_at", start.isoformat())
        .lte("started_at", end.isoformat())
        .execute()
    ).data or []

    messages = (
        sb.table("messages")
        .select("id, created_at")
        .gte("created_at", start.isoformat())
        .lte("created_at", end.isoformat())
        .execute()
    ).data or []

    # --- 2) Bucket per day
    days = _days_span(start, end)
    conv_map = {d.date(): 0 for d in days}
    msg_map = {d.date(): 0 for d in days}

    for r in sessions:
        try:
            dt = datetime.fromisoformat(
                str(r["started_at"]).replace("Z", "+00:00")).date()
            if dt in conv_map:
                conv_map[dt] += 1
        except Exception:
            pass

    for r in messages:
        try:
            dt = datetime.fromisoformat(
                str(r["created_at"]).replace("Z", "+00:00")).date()
            if dt in msg_map:
                msg_map[dt] += 1
        except Exception:
            pass

    series = []
    conv_vals, msg_vals = [], []
    for d in days:
        dd = d.date()
        c = conv_map[dd]
        m = msg_map[dd]
        conv_vals.append(c)
        msg_vals.append(m)
        series.append(
            {
                "label": _labelfr(d),
                "date": d.isoformat(),
                "conversations": c,
                "messages": m,
            }
        )

    cur_conv_total = sum(conv_vals)
    cur_msg_total = sum(msg_vals)

    # --- 3) KPIs (peak, avg/day, growth vs previous period)
    peak_conversations = max(conv_vals) if conv_vals else 0
    avg_messages = round(sum(msg_vals) / max(1, len(msg_vals)), 0)

    prev_start, prev_end = _period_window(period, start - timedelta(seconds=1))

    prev_sessions = (
        sb.table("sessions")
        .select("id, started_at")
        .gte("started_at", prev_start.isoformat())
        .lte("started_at", prev_end.isoformat())
        .execute()
    ).data or []

    prev_messages = (
        sb.table("messages")
        .select("id, created_at")
        .gte("created_at", prev_start.isoformat())
        .lte("created_at", prev_end.isoformat())
        .execute()
    ).data or []

    prev_conv_total = len(prev_sessions)
    prev_msg_total = len(prev_messages)

    # Règle business pour éviter -100%:
    # - si 0 conv dans les deux périodes -> 0%
    # - sinon on calcule sur les conversations
    # - si conv courantes = 0 mais messages > 0 -> fallback sur messages
    if cur_conv_total == 0 and prev_conv_total == 0:
        growth = 0.0
    elif cur_conv_total == 0 and cur_msg_total > 0:
        growth = _smoothed_growth(
            cur_msg_total, prev_msg_total, alpha=2, decimals=0)
    else:
        growth = _smoothed_growth(
            cur_conv_total, prev_conv_total, alpha=2, decimals=0)

    return {
        "period": {"key": period, "start": start.isoformat(), "end": end.isoformat()},
        "kpi": {
            "peak_conversations": peak_conversations,
            "avg_messages_per_day": avg_messages,
            "growth_percent": growth,
        },
        "series": series,
    }
