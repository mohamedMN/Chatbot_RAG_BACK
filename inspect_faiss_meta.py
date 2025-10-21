from pathlib import Path
import json, sys
import numpy as np

root = Path("data/index")  # <- change if needed
idx_path = root/"faiss.index"
meta_path = root/"metadata.json"
idmap_path = root/"idmap.json"

# ---- read FAISS index ----
import faiss  # pip install faiss-cpu
index = faiss.read_index(str(idx_path))

# some indexes expose metric_type; most details we keep in metadata.json
metric_type = getattr(index, "metric_type", None)  # 0=L2, 1=IP (if available)

# ---- read JSON sidecars ----
meta = {}
if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

idmap = {}
if idmap_path.exists():
    idmap = json.loads(idmap_path.read_text(encoding="utf-8"))

# ---- basic consistency ----
ntotal = index.ntotal
dim    = index.d

ids     = idmap.get("ids", []) or []
ordinals= idmap.get("ordinal", []) or []
subjects= idmap.get("subject", []) or []
sources = idmap.get("source", []) or []

print("=== FAISS INDEX ===")
print(f"type           : {type(index).__name__}")
print(f"dimension (d)  : {dim}")
print(f"ntotal vectors : {ntotal}")
print(f"is_trained     : {getattr(index, 'is_trained', True)}")
print(f"metric_type    : {metric_type}  (0=L2, 1=IP; may be None for some index types)")

print("\n=== metadata.json ===")
for k in ["metric","normalized","count","version","embedding_dim","built_at","embedding_model"]:
    if k in meta: print(f"{k:14s}: {meta[k]}")

print("\n=== idmap.json (sizes) ===")
print(f"ids            : {len(ids)}")
print(f"ordinal        : {len(ordinals)}")
print(f"subject        : {len(subjects)}")
print(f"source         : {len(sources)}")

# quick checks
problems = []
if "count" in meta and int(meta["count"]) != ntotal:
    problems.append(f"metadata.count ({meta['count']}) != index.ntotal ({ntotal})")
if ids and len(ids) != ntotal:
    problems.append(f"len(ids) ({len(ids)}) != index.ntotal ({ntotal})")
if problems:
    print("\nWARNINGS:")
    for p in problems: print(" -", p)

# small peek at label distribution
from collections import Counter
if sources:
    top_src = Counter(sources).most_common(5)
    print("\nTop sources:")
    for s, c in top_src: print(f" - {s} : {c}")

print("\nDone.")
