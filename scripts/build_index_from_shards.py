"""Assemble encode_shards.py output into a FAISS index readable by qudar.index.DenseIndexer.

Writes <out>/index.faiss and <out>/index_meta.faiss (pickled list of doc ids), the same
two files qudar.build_index produces.

  --index-type flat : faiss.IndexFlatIP, float32 (exactly what qudar.build_index builds)
  --index-type fp16 : faiss.IndexScalarQuantizer fp16, inner product (half the RAM)
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--index-type", choices=["flat", "fp16"], default="flat")
    args = p.parse_args()

    shard_dir = Path(args.shards)
    emb_files = sorted(shard_dir.glob("shard_*[0-9].npy"))
    if not emb_files:
        raise SystemExit(f"No shards in {shard_dir}")
    dim = np.load(emb_files[0], mmap_mode="r").shape[1]
    if args.index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)

    all_ids: list[str] = []
    for emb_file in emb_files:
        ids = json.loads(emb_file.with_name(emb_file.name.replace(".npy", ".ids.json")).read_text())
        emb = np.load(emb_file).astype(np.float32)
        assert len(ids) == emb.shape[0], emb_file
        index.add(emb)
        all_ids.extend(ids)
        print(f"added {emb_file.name}: {len(ids)} (total {len(all_ids)})", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with (out_dir / "index_meta.faiss").open("wb") as handle:
        pickle.dump(all_ids, handle)
    print(f"Saved {index.ntotal} vectors ({args.index_type}) to {out_dir}")


if __name__ == "__main__":
    main()
