"""Step 10b: build FEVER Contriever shards from the Climate-FEVER shards (CPU is enough).

Step 10a found that FEVER and Climate-FEVER share 5,416,536 documents with identical text,
32 ids exist only in FEVER and 57 only in Climate-FEVER. So instead of a 3-hour re-encode:
  1. copy every Climate-FEVER shard, keeping only rows whose id is a FEVER id;
  2. encode the FEVER-only documents with encode_shards.py's own text format and pooling
     and write them as one extra shard;
  3. check that the shard ids equal the FEVER corpus ids, then write DONE.
It also reports whether the FEVER-only ids are Climate-FEVER ids in another Unicode
normalisation (NFC vs NFD), which would explain the 32/57 difference.

  python build_fever_shards.py \
      --cf-shards  /kaggle/input/.../qudar-shards-climate-fever/<folder with shard_000.npy> \
      --fever-corpus /kaggle/input/.../qudar-beir-data-2/data/fever/corpus.jsonl \
      --out /kaggle/working/shards/fever
"""
from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cf_shard_files(cf_dir: Path) -> list[Path]:
    files = sorted(cf_dir.glob("shard_*[0-9].npy"))
    if not files:
        raise SystemExit(f"no shard_XXX.npy in {cf_dir}")
    return files


def read_fever(corpus: Path, cf_ids: set[str]):
    """FEVER ids in corpus order (first copy of an id, as encode_shards.py) + rows missing from Climate-FEVER."""
    ids, seen, missing = [], set(), []
    with corpus.open(encoding="utf8") as fh:
        for line in fh:
            row = json.loads(line)
            did = str(row.get("_id"))
            if did in seen:
                continue
            seen.add(did)
            ids.append(did)
            if did not in cf_ids:
                missing.append(row)
    return ids, missing


def unicode_report(missing_ids: list[str], cf_only: set[str]) -> dict:
    by_form = {form: {unicodedata.normalize(form, i): i for i in cf_only} for form in ("NFC", "NFD")}
    matches = {}
    for i in missing_ids:
        for form, table in by_form.items():
            hit = table.get(unicodedata.normalize(form, i))
            if hit is not None and hit != i:
                matches[i] = {"climate_fever_id": hit, "form": form}
                break
    return {"fever_only": len(missing_ids), "same_title_other_normalisation": len(matches),
            "examples": dict(list(matches.items())[:5])}


def encode_rows(rows: list[dict], model_name: str, batch_size: int) -> np.ndarray:
    """Same text format, pooling and normalisation as encode_shards.py (imported from it)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    from encode_shards import encode, merged_text

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    texts = [merged_text(r.get("title"), r.get("text")) for r in rows]
    return encode(model, tokenizer, texts, batch_size, 512, device, fp16=device == "cuda")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cf-shards", required=True, help="folder with Climate-FEVER shard_XXX.npy + .ids.json")
    p.add_argument("--fever-corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="facebook/contriever")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    cf_dir, out = Path(args.cf_shards), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = cf_shard_files(cf_dir)
    cf_ids: set[str] = set()
    for f in files:
        cf_ids.update(json.loads(f.with_name(f.name.replace(".npy", ".ids.json")).read_text()))
    log(f"Climate-FEVER shards: {len(files)} files, {len(cf_ids)} ids")

    fever_ids, missing = read_fever(Path(args.fever_corpus), cf_ids)
    fever_set = set(fever_ids)
    cf_only = cf_ids - fever_set
    log(f"FEVER corpus: {len(fever_ids)} ids; {len(missing)} not in Climate-FEVER; {len(cf_only)} Climate-FEVER ids to drop")
    report = {"fever_ids": len(fever_ids), "cf_ids": len(cf_ids), "dropped_cf_only": len(cf_only),
              "unicode": unicode_report([r["_id"] for r in missing], cf_only)}

    kept_total = 0
    for f in files:
        ids = json.loads(f.with_name(f.name.replace(".npy", ".ids.json")).read_text())
        keep = np.fromiter((i in fever_set for i in ids), dtype=bool, count=len(ids))
        emb_out, ids_out = out / f.name, out / f.name.replace(".npy", ".ids.json")
        if not (emb_out.exists() and ids_out.exists()):
            emb = np.load(f, mmap_mode="r")
            if emb.shape[0] != len(ids):
                raise SystemExit(f"{f.name}: {emb.shape[0]} vectors but {len(ids)} ids")
            np.save(emb_out.with_suffix(".tmp.npy"), np.ascontiguousarray(emb[keep]))
            emb_out.with_suffix(".tmp.npy").rename(emb_out)
            ids_out.write_text(json.dumps([i for i, k in zip(ids, keep) if k]))
        kept_total += int(keep.sum())
        log(f"  {f.name}: kept {int(keep.sum())}/{len(ids)}")

    if missing:
        extra_no = int(files[-1].stem.split("_")[1]) + 1
        emb_out = out / f"shard_{extra_no:03d}.npy"
        if not emb_out.exists():
            log(f"Encoding {len(missing)} FEVER-only documents with {args.model}")
            vecs = encode_rows(missing, args.model, args.batch_size)
            np.save(emb_out.with_suffix(".tmp.npy"), vecs)
            emb_out.with_suffix(".tmp.npy").rename(emb_out)
            (out / f"shard_{extra_no:03d}.ids.json").write_text(json.dumps([str(r["_id"]) for r in missing]))
        kept_total += len(missing)

    shard_ids: list[str] = []
    for f in sorted(out.glob("shard_*[0-9].ids.json")):
        shard_ids.extend(json.loads(f.read_text()))
    ok = len(shard_ids) == len(fever_ids) and set(shard_ids) == fever_set
    report.update({"shard_ids": len(shard_ids), "match_fever_corpus": ok})
    (out / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if not ok:
        raise SystemExit(f"shard ids ({len(shard_ids)}) do not match FEVER corpus ({len(fever_ids)}); see build_report.json")
    (out / "DONE").write_text(str(len(shard_ids)))
    print("\n===== Step 10b FEVER shards (send this block back) =====")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
