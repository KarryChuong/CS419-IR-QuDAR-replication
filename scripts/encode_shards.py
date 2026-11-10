"""Encode a BEIR corpus.jsonl into resumable fp16 embedding shards.

Uses the same text format, pooling and normalisation as qudar.retrievers.DenseEncoder
(passage prefix "passage: ", title + text, mean pooling, L2-normalise, max_length 512),
but streams the corpus, sorts each shard by length for speed, runs in fp16 and saves
every shard to disk so an interrupted run resumes where it stopped.

Usage:
  python encode_shards.py --corpus /path/corpus.jsonl --out /kaggle/working/shards \
      --model facebook/contriever --shard-size 500000 --batch-size 256
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PASSAGE_PREFIX = "passage: "  # identical to qudar DenseEncoder default


def merged_text(title, text) -> str:
    return PASSAGE_PREFIX + (" ".join(part for part in [title, text] if part)).strip()


def iter_shards(corpus_path: Path, shard_size: int):
    ids, texts = [], []
    seen = set()
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            doc_id = str(row.get("_id"))
            if doc_id in seen:  # GenericDataLoader keeps one entry per id
                continue
            seen.add(doc_id)
            ids.append(doc_id)
            texts.append(merged_text(row.get("title"), row.get("text")))
            if len(ids) == shard_size:
                yield ids, texts
                ids, texts = [], []
    if ids:
        yield ids, texts


@torch.no_grad()
def encode(model, tokenizer, texts, batch_size, max_length, device, fp16) -> np.ndarray:
    order = np.argsort([-len(t) for t in texts], kind="stable")  # longest first: OOM shows up early
    out = np.zeros((len(texts), model_dim(model)), dtype=np.float16)
    for start in tqdm(range(0, len(texts), batch_size), desc="encode", mininterval=30):
        idx = order[start:start + batch_size]
        batch = [texts[i] for i in idx]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1).float() / mask.sum(dim=1).float().clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out[idx] = pooled.cpu().numpy().astype(np.float16)
    return out


def model_dim(model) -> int:
    base = model.module if hasattr(model, "module") else model
    return base.config.hidden_size


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="facebook/contriever")
    p.add_argument("--shard-size", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--limit-shards", type=int, default=None, help="stop after N shards (for a quick test)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda" and not args.no_fp16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for shard_no, (ids, texts) in enumerate(iter_shards(Path(args.corpus), args.shard_size)):
        if args.limit_shards is not None and shard_no >= args.limit_shards:
            break
        emb_path = out_dir / f"shard_{shard_no:03d}.npy"
        ids_path = out_dir / f"shard_{shard_no:03d}.ids.json"
        total += len(ids)
        if emb_path.exists() and ids_path.exists():
            print(f"shard {shard_no}: already done, skipping")
            continue
        t0 = time.time()
        emb = encode(model, tokenizer, texts, args.batch_size, args.max_length, device, fp16)
        np.save(emb_path.with_suffix(".tmp.npy"), emb)
        emb_path.with_suffix(".tmp.npy").rename(emb_path)  # write-then-rename: no half-written shards
        ids_path.write_text(json.dumps(ids))
        print(f"shard {shard_no}: {len(ids)} docs in {time.time() - t0:.0f}s (total {total})", flush=True)
    (out_dir / "DONE").write_text(str(total))
    print(f"Finished: {total} documents")


if __name__ == "__main__":
    main()
