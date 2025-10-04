# api/database.py
from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional
from supabase import create_client, Client  # type: ignore
from dotenv import load_dotenv  



load_dotenv(override=False)

def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing env var: {name}")
    return v


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    return create_client(_env("supabaseUrl"), _env("supabaseKey"))
