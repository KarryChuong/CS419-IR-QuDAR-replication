"""Check the six Kaggle datasets before Step 2.  python check_datasets.py /kaggle/input"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/kaggle/input")
EXPECT_CORPUS = {"fiqa": 57638, "scidocs": 25657, "climate-fever": 5416593, "hotpotqa": 5233329}
EXPECT_EXP = {("fiqa", "test"): 648, ("scidocs", "test"): 1000, ("climate-fever", "test"): 1535,
              ("hotpotqa", "test"): 7405, ("fiqa", "train"): 5500, ("hotpotqa", "train"): 10000}
ok = True


def report(good: bool, msg: str) -> None:
    global ok
    ok &= good
    print(("  OK   " if good else "  FAIL ") + msg)


def find_one(pattern: str):
    hits = sorted(root.rglob(pattern))
    return hits


print("1. qudar-beir-data")
for ds, n in EXPECT_CORPUS.items():
    corpus = [p for p in find_one("corpus.jsonl") if p.parent.name == ds]
    if not corpus:
        report(False, f"{ds}/corpus.jsonl not found"); continue
    d = corpus[0].parent
    need = ["queries.jsonl", "qrels/test.tsv"] + (["qrels/train.tsv"] if ds in ("fiqa", "hotpotqa") else [])
    missing = [f for f in need if not (d / f).exists()]
    report(not missing, f"{ds}: {d}  (missing: {missing or 'none'})")
print("   -> DATA =", corpus[0].parent.parent if corpus else "?")

print("2. qudar-patches")
for f in ["apply_patch.py", "bm25_fast.py", "run_retrieval.py", "encode_shards.py", "expand_queries.py"]:
    hits = find_one(f)
    report(bool(hits), f"{f}: {hits[0] if hits else 'missing'}")
src = find_one("bm25_fast.py")
report(bool(src) and "from_texts" in src[0].read_text(), "bm25_fast.py is the newest version (has BM25sSparseRetriever.from_texts)")
print("   -> PATCH =", src[0].parent if src else "?")

print("3. qudar-index-small")
for ds in ("fiqa", "scidocs"):
    idx = [p for p in find_one("index.faiss") if ds in p.parts]
    if not idx:
        report(False, f"{ds}: index.faiss not found"); continue
    import faiss
    n = faiss.read_index(str(idx[0])).ntotal
    meta = pickle.load(open(idx[0].with_name("index_meta.faiss"), "rb"))
    report(n == len(meta) == EXPECT_CORPUS[ds], f"{ds}: {n} vectors, {len(meta)} ids  ({idx[0].parent})")

print("4-5. qudar-shards-*")
for ds in ("climate-fever", "hotpotqa"):
    shards = sorted(p for p in find_one("shard_*[0-9].npy") if ds in str(p))
    rows = sum(np.load(p, mmap_mode="r").shape[0] for p in shards)
    ids = sum(len(json.loads(p.with_name(p.name.replace(".npy", ".ids.json")).read_text())) for p in shards)
    dims = {np.load(p, mmap_mode="r").shape[1] for p in shards}
    report(len(shards) == 11 and rows == ids == EXPECT_CORPUS[ds] and dims == {768},
           f"{ds}: {len(shards)} shards, {rows} vectors, {ids} ids, dim {dims}  ({shards[0].parent if shards else '?'})")

print("6. qudar-expansions")
for (ds, split), n in EXPECT_EXP.items():
    f = find_one(f"expanded_queries_{ds}_{split}_expansion.json")
    if not f:
        report(False, f"{ds}/{split}: file missing"); continue
    d = json.loads(f[0].read_text())
    empty = sum(1 for v in d.values() if not v.strip())
    report(len(d) == n and empty == 0, f"{ds}/{split}: {len(d)} queries (expected {n}), {empty} empty")
print("   -> EXP =", f[0].parent if f else "?")
print("\nALL GOOD" if ok else "\nSOMETHING IS WRONG - send this output")
