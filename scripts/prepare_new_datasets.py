"""Step 10a: download and check the four new BEIR datasets (CPU only).

  SciFact (5K docs, has train), NFCorpus (3.6K, has train), FEVER (5.4M, has train),
  DBPedia-entity (4.6M, no train) -- two small and two multi-million corpora, as in QuDAR.

For every dataset it reports: unique / duplicate corpus ids, queries, and per qrels split the
number of queries, pairs, relevance levels, positives per query and qrels ids missing from the
corpus. With --cf-corpus it also compares FEVER's corpus with Climate-FEVER's, document by
document, using the exact text encode_shards.py embeds ("title text", first copy of an id):
if every FEVER document is in Climate-FEVER with the same text, FEVER can reuse the
Climate-FEVER Contriever shards (drop the extra ids) instead of a 3-hour re-encode.

  python prepare_new_datasets.py --out /kaggle/working \
      --cf-corpus /kaggle/input/datasets/karrychuong/qudar-beir-data/data/climate-fever/corpus.jsonl

Writes <out>/data/<dataset>/ (BEIR layout), <out>/step10a_report.json, and, for FEVER,
<out>/fever_vs_cf/{summary.json, fever_only_ids.txt, text_mismatch_ids.txt}.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
import urllib.request
from collections import Counter
from pathlib import Path

DATASETS = ["scifact", "nfcorpus", "fever", "dbpedia-entity"]
# Corpus size and test queries as listed in the BEIR paper (Thakur et al., 2021, Table 1); a mismatch is a warning.
EXPECT = {"scifact": (5183, 300), "nfcorpus": (3633, 323), "fever": (5416568, 6666), "dbpedia-entity": (4635922, 400)}
TRAIN_CAP = {"fever": 10000}  # first N train queries, as for HotpotQA; others: all train queries
EXPANSION_RATE = 5.5          # queries / s, LLaMA 3.1-8B on T4 x2 in Step 1C
HF = "https://huggingface.co"
# Mirror on Hugging Face (the UKP server times out from Kaggle): BeIR/<name> holds corpus + queries as parquet,
# BeIR/<name>-qrels holds <split>.tsv. Converted here to the BEIR layout (corpus.jsonl, queries.jsonl, qrels/<split>.tsv).


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 22)
    tmp.rename(dest)


def parquet_to_jsonl(parquet_files: list, dest: Path, keep: list) -> int:
    import pyarrow.parquet as pq
    n, tmp = 0, dest.with_suffix(".jsonl.part")
    with tmp.open("w", encoding="utf8") as out:
        for pf in parquet_files:
            for batch in pq.ParquetFile(pf).iter_batches(batch_size=50000):
                cols = {c: batch.column(c).to_pylist() for c in keep if c in batch.schema.names}
                for i in range(batch.num_rows):
                    out.write(json.dumps({c: cols[c][i] for c in cols}, ensure_ascii=False) + "\n")
                    n += 1
    tmp.rename(dest)
    return n


def hf_files(repo: str, folder: str = "") -> list:
    items = http_json(f"{HF}/api/datasets/{repo}/tree/main/{folder}".rstrip("/"))
    return [i["path"] for i in items if i["type"] == "file"]


def download(name: str, data_dir: Path) -> Path:
    target = data_dir / name
    if (target / "corpus.jsonl").exists() and (target / "queries.jsonl").exists() and list((target / "qrels").glob("*.tsv")):
        log(f"{name}: already present, skip download")
        return target
    (target / "qrels").mkdir(parents=True, exist_ok=True)
    tmp = target / "_parquet"
    tmp.mkdir(exist_ok=True)
    for kind, keep in (("queries", ["_id", "text"]), ("corpus", ["_id", "title", "text"])):
        files = []
        for path in sorted(f for f in hf_files(f"BeIR/{name}", kind) if f.endswith(".parquet")):
            log(f"{name}: downloading {path}")
            dest = tmp / path.replace("/", "_")
            fetch(f"{HF}/datasets/BeIR/{name}/resolve/main/{path}", dest)
            files.append(dest)
        if not files:
            raise SystemExit(f"{name}: no parquet under BeIR/{name}/{kind}")
        n = parquet_to_jsonl(files, target / f"{kind}.jsonl", keep)
        log(f"{name}: wrote {n} rows to {kind}.jsonl")
        for f in files:
            f.unlink()
    for path in hf_files(f"BeIR/{name}-qrels"):
        if path.endswith(".tsv"):
            fetch(f"{HF}/datasets/BeIR/{name}-qrels/resolve/main/{path}", target / "qrels" / path)
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"{name}: qrels splits {[f.stem for f in (target / 'qrels').glob('*.tsv')]}")
    return target


def merged_text(row: dict) -> str:
    """Exactly what encode_shards.py embeds (without its constant 'passage: ' prefix)."""
    return (" ".join(part for part in [row.get("title"), row.get("text")] if part)).strip()


def corpus_stats(path: Path):
    ids, dup, words = set(), 0, 0
    with path.open(encoding="utf8") as fh:
        for line in fh:
            row = json.loads(line)
            did = str(row.get("_id"))
            if did in ids:
                dup += 1
                continue
            ids.add(did)
            words += len(merged_text(row).split())
    return ids, dup, words / max(len(ids), 1)


def qrels_stats(path: Path, corpus_ids: set) -> dict:
    per_q, levels, missing, pairs = {}, Counter(), 0, 0
    with path.open(encoding="utf8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header: query-id corpus-id score
        for qid, did, score in reader:
            score = int(score)
            pairs += 1
            levels[score] += 1
            missing += did not in corpus_ids
            per_q.setdefault(qid, []).append(score)
    pos = [sum(s > 0 for s in v) for v in per_q.values()]
    return {"queries": len(per_q), "pairs": pairs, "levels": dict(sorted(levels.items())),
            "queries_without_positive": sum(p == 0 for p in pos),
            "mean_positives_per_query": round(sum(pos) / max(len(pos), 1), 2),
            "qrels_ids_missing_from_corpus": missing}


def compare_with_cf(fever_corpus: Path, cf_corpus: Path, out_dir: Path) -> dict:
    """FEVER vs Climate-FEVER corpus, keyed by _id, first copy of each id (as encode_shards.py)."""
    log("FEVER vs Climate-FEVER: hashing the Climate-FEVER corpus")
    cf = {}
    with cf_corpus.open(encoding="utf8") as fh:
        for line in fh:
            row = json.loads(line)
            did = str(row.get("_id"))
            if did not in cf:
                cf[did] = hashlib.md5(merged_text(row).encode("utf8")).digest()[:8]
    log(f"  {len(cf)} Climate-FEVER ids; scanning FEVER")
    seen, only_fever, mismatch = set(), [], []
    with fever_corpus.open(encoding="utf8") as fh:
        for line in fh:
            row = json.loads(line)
            did = str(row.get("_id"))
            if did in seen:
                continue
            seen.add(did)
            h = cf.get(did)
            if h is None:
                only_fever.append(did)
            elif h != hashlib.md5(merged_text(row).encode("utf8")).digest()[:8]:
                mismatch.append(did)
    only_cf = len(cf) - (len(seen) - len(only_fever))
    to_encode = len(only_fever) + len(mismatch)
    verdict = ("REUSE: FEVER = Climate-FEVER minus some ids; filter the Climate-FEVER shards" if to_encode == 0 else
               "PARTIAL: filter the Climate-FEVER shards and encode the listed ids" if to_encode <= 0.01 * len(seen) else
               "ENCODE: corpora differ too much; encode FEVER from scratch")
    summary = {"fever_ids": len(seen), "cf_ids": len(cf), "common_same_text": len(seen) - to_encode,
               "fever_only": len(only_fever), "common_text_mismatch": len(mismatch), "cf_only": only_cf,
               "verdict": verdict, "examples_fever_only": only_fever[:10], "examples_mismatch": mismatch[:10]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "fever_only_ids.txt").write_text("\n".join(only_fever))
    (out_dir / "text_mismatch_ids.txt").write_text("\n".join(mismatch))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--cf-corpus", help="Climate-FEVER corpus.jsonl (qudar-beir-data) for the FEVER comparison")
    p.add_argument("--no-download", action="store_true", help="only check what is already in <out>/data")
    args = p.parse_args()

    out = Path(args.out)
    data_dir = out / "data"
    report, plan = {}, []
    for name in args.datasets:
        d = data_dir / name if args.no_download else download(name, data_dir)
        corpus_ids, dup, avg_words = corpus_stats(d / "corpus.jsonl")
        with (d / "queries.jsonl").open(encoding="utf8") as fh:
            n_queries = sum(1 for _ in fh)
        splits = {f.stem: qrels_stats(f, corpus_ids) for f in sorted((d / "qrels").glob("*.tsv"))}
        exp_docs, exp_test = EXPECT.get(name, (None, None))
        warn = []
        if exp_docs and len(corpus_ids) != exp_docs:
            warn.append(f"corpus {len(corpus_ids)} != BEIR {exp_docs}")
        if exp_test and splits.get("test", {}).get("queries") != exp_test:
            warn.append(f"test queries {splits.get('test', {}).get('queries')} != BEIR {exp_test}")
        report[name] = {"corpus_docs": len(corpus_ids), "duplicate_ids": dup, "avg_words_per_doc": round(avg_words, 1),
                        "queries_jsonl": n_queries, "qrels": splits, "warnings": warn,
                        "size_gb": round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9, 2)}
        log(f"{name}: {len(corpus_ids)} docs ({dup} duplicate ids), splits {list(splits)}, warnings {warn or 'none'}")
        for split in ("test", "train"):
            if split in splits:
                n = min(splits[split]["queries"], TRAIN_CAP.get(name, 10 ** 9)) if split == "train" else splits[split]["queries"]
                plan.append({"dataset": name, "split": split, "queries": n})
        del corpus_ids

    if "fever" in args.datasets and args.cf_corpus:
        report["fever_vs_climate_fever"] = compare_with_cf(data_dir / "fever" / "corpus.jsonl", Path(args.cf_corpus),
                                                           out / "fever_vs_cf")
        log(f"FEVER vs Climate-FEVER: {report['fever_vs_climate_fever']['verdict']}")
    total = sum(r["queries"] for r in plan)
    report["expansion_plan"] = {"rows": plan, "total_queries": total,
                                "estimated_minutes_T4x2": round(total / EXPANSION_RATE / 60)}
    (out / "step10a_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n===== Step 10a summary (send this block back) =====")
    for name in args.datasets:
        r = report[name]
        q = ", ".join(f"{s}: {v['queries']} q / {v['mean_positives_per_query']} pos/q / levels {v['levels']}"
                      + (f" / {v['queries_without_positive']} q without positive" if v["queries_without_positive"] else "")
                      + (f" / {v['qrels_ids_missing_from_corpus']} missing ids" if v["qrels_ids_missing_from_corpus"] else "")
                      for s, v in r["qrels"].items())
        print(f"{name}: {r['corpus_docs']} docs, {r['avg_words_per_doc']} words/doc, {r['size_gb']} GB | {q}"
              + (f" | WARN {r['warnings']}" if r["warnings"] else ""))
    if "fever_vs_climate_fever" in report:
        s = report["fever_vs_climate_fever"]
        print(f"FEVER vs Climate-FEVER: {s['common_same_text']} same text, {s['fever_only']} only in FEVER, "
              f"{s['common_text_mismatch']} text differs, {s['cf_only']} only in Climate-FEVER -> {s['verdict']}")
    print(f"Expansion plan: {total} queries, about {report['expansion_plan']['estimated_minutes_T4x2']} min on T4 x2: "
          + ", ".join(f"{r['dataset']}/{r['split']} {r['queries']}" for r in plan))


if __name__ == "__main__":
    main()
