"""Step 2 runner: the four QuDAR retrieval lists + all non-LLM results, memory-staged.

Same computations as qudar.pipeline.run_experiment with the LLM part switched off
(qudar's DenseEncoder, DenseIndexer, build_retrieval_result, fusion and evaluation are
imported and called unchanged). Differences are only in orchestration:
  * BM25 (BM25sSparseRetriever, paper setup) is built from a streamed corpus and freed
    before the dense stage, so 5M-document corpora fit in Kaggle RAM;
  * --dense-shards searches encode_shards.py output exactly on the GPU (fp32 inner
    product of the fp16-rounded vectors, as DenseIndexer does) instead of a CPU FAISS
    index; --dense-faiss uses a qudar.build_index index through DenseIndexer.
  * several splits share one BM25 index: --splits test train:10000 (":N" = first N
    queries, the same cut as qudar --max-queries and expand_queries.py --max-queries).

Outputs per split in <out>/<dataset>_<split>/: retrieval_result.json, eval_results.csv,
confidence_weights.json and the three *_hybrid_grid.csv files, as the pipeline writes them.
"""
from __future__ import annotations

import os

# Keep JAX (pre-installed on Kaggle/Colab) off the GPU: it would reserve 75% of GPU memory.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="patched QuDAR clone (for qudar imports)")
    p.add_argument("--data-root", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--expansions", required=True, help="folder with expanded_queries_<ds>_<split>_expansion.json")
    dense = p.add_mutually_exclusive_group(required=True)
    dense.add_argument("--dense-faiss", help="folder with index.faiss + index_meta.faiss")
    dense.add_argument("--dense-shards", help="folder with shard_XXX.npy + shard_XXX.ids.json")
    p.add_argument("--model", default="facebook/contriever")
    p.add_argument("--out", required=True)
    p.add_argument("--top-k", type=int, default=1000)
    p.add_argument("--eval-at", type=int, default=10)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--confidence-topk", type=int, default=10)
    p.add_argument("--confidence-tau", type=float, default=2.0)
    p.add_argument("--query-batch", type=int, default=512)
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_split(data_dir: Path, split: str, max_queries: int | None):
    from beir.datasets.data_loader import GenericDataLoader

    loader = GenericDataLoader(data_folder=str(data_dir))
    loader.qrels_file = str(data_dir / "qrels" / f"{split}.tsv")
    loader._load_queries()
    loader._load_qrels()
    queries = {qid: loader.queries[qid] for qid in loader.qrels}  # exactly GenericDataLoader.load order
    qids = list(queries)[:max_queries]
    return qids, [queries[q] for q in qids], loader.qrels


def stream_corpus(corpus_file: Path):
    doc_ids, texts = [], []
    with corpus_file.open(encoding="utf8") as handle:
        for line in handle:
            row = json.loads(line)
            doc_ids.append(row.get("_id"))
            texts.append((" ".join(part for part in [row.get("title"), row.get("text")] if part)).strip())
    return doc_ids, texts


def search_shards(shard_dir: Path, query_vectors: np.ndarray, top_k: int, batch: int):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.from_numpy(query_vectors.astype(np.float16).astype(np.float32)).to(device)  # DenseIndexer casts queries to fp16
    best_s = torch.full((q.shape[0], top_k), -float("inf"), device=device)
    best_i = torch.full((q.shape[0], top_k), -1, dtype=torch.long, device=device)
    all_ids: list[str] = []
    for emb_file in sorted(shard_dir.glob("shard_*[0-9].npy")):
        ids = json.loads(emb_file.with_name(emb_file.name.replace(".npy", ".ids.json")).read_text())
        emb = torch.from_numpy(np.load(emb_file)).to(device).float()
        offset = len(all_ids)
        all_ids.extend(ids)
        k = min(top_k, emb.shape[0])
        for start in range(0, q.shape[0], batch):
            s, i = (q[start:start + batch] @ emb.T).topk(k, dim=1)
            cat_s = torch.cat([best_s[start:start + batch], s], dim=1)
            cat_i = torch.cat([best_i[start:start + batch], i + offset], dim=1)
            top_s, pos = cat_s.topk(top_k, dim=1)
            best_s[start:start + batch] = top_s
            best_i[start:start + batch] = cat_i.gather(1, pos)
        del emb
        log(f"  dense search: {emb_file.name} done ({len(all_ids)} docs)")
    best_s, best_i = best_s.cpu().numpy(), best_i.cpu().numpy()
    return [([all_ids[j] for j in row_i if j >= 0], row_s[row_i >= 0]) for row_i, row_s in zip(best_i, best_s)]


def check_shards(shard_dir: Path, corpus_n: int, corpus_sample: set) -> None:
    """Stop early if --dense-shards points at another corpus or an incomplete shard set."""
    shard_ids: set[str] = set()
    for ids_file in sorted(shard_dir.glob("shard_*[0-9].ids.json")):
        shard_ids.update(json.loads(ids_file.read_text()))
    missing = len(corpus_sample - shard_ids)
    log(f"Shard check: {len(shard_ids)} shard ids vs {corpus_n} corpus docs, {missing}/{len(corpus_sample)} sampled ids missing")
    if len(shard_ids) != corpus_n or missing:
        raise SystemExit(f"--dense-shards {shard_dir} does not match this corpus "
                         f"({len(shard_ids)} ids vs {corpus_n} docs, {missing} sampled ids missing). Wrong shard folder?")


def evaluate_and_save(out_dir: Path, qids, qrels, sparse, dense, exp_sparse, exp_dense, args) -> None:
    # Post-retrieval part of qudar.pipeline.run_experiment, unchanged apart from LLM being off.
    from qudar.evaluation import evaluate_run, evaluate_static_dynamic_hybrids, summarize_results
    from qudar.fusion import build_confidence_run, build_equal_run, build_rrf_run, build_single_run
    from qudar.io_utils import save_dataframe, save_json
    from qudar.results import build_retrieval_result

    out_dir.mkdir(parents=True, exist_ok=True)
    retrieval_result = build_retrieval_result(qids, sparse, dense, exp_sparse, exp_dense)
    save_json(retrieval_result, out_dir / "retrieval_result.json")
    at_k = args.eval_at
    results = {}
    for source in ("os", "od", "es", "ed"):
        results[source], _ = evaluate_run(qrels, build_single_run(retrieval_result, source), at_k=at_k)
    results["qudar_simple_rrf"], _ = evaluate_run(qrels, build_rrf_run(retrieval_result, k=args.rrf_k), at_k=at_k)
    results["qudar_simple_equal"], _ = evaluate_run(qrels, build_equal_run(retrieval_result), at_k=at_k)
    run_conf, conf_meta = build_confidence_run(retrieval_result, topk=args.confidence_topk, tau=args.confidence_tau)
    results["qudar_confidence"], _ = evaluate_run(qrels, run_conf, at_k=at_k)
    save_json(conf_meta, out_dir / "confidence_weights.json")
    for left, right, name in (("os", "od", "original_retriever_type"), ("os", "es", "sparse_query_format"),
                              ("od", "ed", "dense_query_format")):
        df, sha, shb, dhb = evaluate_static_dynamic_hybrids(qrels, retrieval_result, left_source=left,
                                                            right_source=right, at_k=at_k)
        results[f"sha_{left}_{right}"], results[f"shb_{left}_{right}"] = sha.to_dict(), shb.to_dict()
        results[f"dhb_{left}_{right}"] = dhb
        save_dataframe(df.reset_index().rename(columns={"index": "metric"}), out_dir / f"{name}_hybrid_grid.csv")
    summary = summarize_results(experiment_name=out_dir.name, dataset_name=args.dataset, at_k=at_k, results=results)
    save_dataframe(summary, out_dir / "eval_results.csv")
    print(summary[["method", f"Recall@{at_k}", f"nDCG@{at_k}", "Precision@1"]].to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.repo) / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bm25_fast import BM25sSparseRetriever

    data_dir = Path(args.data_root) / args.dataset
    jobs = []
    for spec in args.splits:
        split, _, n = spec.partition(":")
        qids, qtexts, qrels = load_split(data_dir, split, int(n) if n else None)
        exp_file = Path(args.expansions) / f"expanded_queries_{args.dataset}_{split}_expansion.json"
        expanded = json.loads(exp_file.read_text())
        missing = [q for q in qids if not expanded.get(q)]
        if missing:
            raise SystemExit(f"{exp_file.name}: {len(missing)} queries without expansion, e.g. {missing[:3]}")
        jobs.append({"split": split, "qids": qids, "qtexts": qtexts, "qrels": qrels,
                     "etexts": [expanded[q] for q in qids]})
        log(f"{args.dataset}/{split}: {len(qids)} queries")

    log("BM25: reading corpus")
    doc_ids, texts = stream_corpus(data_dir / "corpus.jsonl")
    log(f"BM25: indexing {len(doc_ids)} documents")
    bm25 = BM25sSparseRetriever.from_texts(doc_ids, texts, top_k=args.top_k)
    corpus_n, corpus_sample = len(doc_ids), set(doc_ids[:: max(1, len(doc_ids) // 2000)])
    del texts, doc_ids
    gc.collect()
    for job in jobs:
        log(f"BM25 search {job['split']}: OS + ES")
        job["sparse"] = bm25.search(job["qtexts"], args.top_k)
        job["exp_sparse"] = bm25.search(job["etexts"], args.top_k)
    del bm25
    gc.collect()

    from qudar.retrievers import build_dense_encoder

    if args.dense_shards:
        check_shards(Path(args.dense_shards), corpus_n, corpus_sample)
    encoder = build_dense_encoder(args.model)
    if args.dense_faiss:
        from qudar.index import DenseIndexer

        indexer = DenseIndexer(768)
        indexer.deserialize_from(args.dense_faiss)
    for job in jobs:
        log(f"Dense {job['split']}: encode queries")
        q_vec = encoder.encode_queries(job["qtexts"])          # as DenseRetriever.encode_queries
        e_vec = encoder.encode_corpus(corpus=job["etexts"])     # as DenseRetriever.encode_passages
        if args.dense_faiss:
            job["dense"] = indexer.search(q_vec, args.top_k)
            job["exp_dense"] = indexer.search(e_vec, args.top_k)
        else:
            both = search_shards(Path(args.dense_shards), np.vstack([q_vec, e_vec]), args.top_k, args.query_batch)
            job["dense"], job["exp_dense"] = both[: len(q_vec)], both[len(q_vec):]
        out_dir = Path(args.out) / f"{args.dataset}_{job['split']}"
        log(f"Evaluate {job['split']} -> {out_dir}")
        evaluate_and_save(out_dir, job["qids"], job["qrels"], job["sparse"], job["dense"],
                          job["exp_sparse"], job["exp_dense"], args)
        for key in ("sparse", "exp_sparse", "dense", "exp_dense"):
            job.pop(key)
        gc.collect()
    log("All done")


if __name__ == "__main__":
    main()
