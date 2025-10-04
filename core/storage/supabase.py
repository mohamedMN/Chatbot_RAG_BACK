# backend/core/storage/supabase.py
from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional
from supabase import create_client, Client
from pydantic_settings import BaseSettings


class _Settings(BaseSettings):
    supabase_url: Optional[str] = os.getenv("SUPABASE_URL")
    supabase_service_role_key: Optional[str] = os.getenv(
        "SUPABASE_SERVICE_ROLE_KEY")

    class Config:
        env_file = ".env"


@lru_cache(maxsize=1)
def _get_settings() -> _Settings:
    return _Settings()


_client_service: Optional[Client] = None


def supa_client() -> Client:
    s = _get_settings()
    if not s.supabase_url:
        raise RuntimeError("Missing SUPABASE_URL")
    if not s.supabase_service_role_key:
        raise RuntimeError("Missing SUPABASE_SERVICE_ROLE_KEY")
    global _client_service
    if _client_service is None:
        _client_service = create_client(
            s.supabase_url, s.supabase_service_role_key)
    return _client_service
