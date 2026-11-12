"""Step 8: gather everything the laptop demo needs into one folder + zip (run once on Kaggle).

Reads recursively from --input (Kaggle: /kaggle/input), the same datasets as Steps 2-7:
  qudar-beir-data      <dataset>/corpus.jsonl, queries.jsonl, qrels/<split>.tsv
  qudar-expansions     expanded_queries_<dataset>_<split>_expansion.json   (LLaMA 3.1-8B, Step 1)
  qudar-runs           <dataset>_<split>/retrieval_result.json              (Step 2, cached 4 lists)
  qudar-grid           <name>/grid_metrics.npz                              (Step 3; train sets -> Static-train)
  qudar-llm            llm_scores_paper_<dataset>_<split>.csv               (gpt-4o-mini, Step 4)
  qudar-step6          model_{A,B}_both.txt, model_{A,B}_features.json, position_bias_train.json,
                       chosen_weights_<dataset>_<split>.csv
  qudar-index-small    the <dataset> FAISS index (index.faiss + index_meta.faiss) -> embeddings
If no FAISS index of the right size is found, the corpus is re-encoded with Contriever
(needs a GPU: about 5-8 minutes for FiQA on a T4).

Writes <out>/ (about 150 MB) and <out>.zip:
  corpus.jsonl, queries.json, qrels.json, expansions.json, llm_scores.json
  dense_emb.npy (fp16, corpus order), cached_runs.npz (Step 2 lists of the test queries)
  model_A.txt, model_B.txt, model_A_features.json, model_B_features.json, chosen_weights.csv
  manifest.json (static point, position bias, fixed-alpha / fixed-beta2 baselines, file sources)
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np

KEYS = [("sparse_docs", "sparse_scores"), ("dense_docs", "dense_scores"),
        ("pgt_sparse_docs", "pgt_sparse_scores"), ("pgt_dense_docs", "pgt_dense_scores")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_all(root: Path, pattern: str):
    return sorted(root.rglob(pattern))


def find(root: Path, pattern: str, must_contain: str | None = None, avoid: tuple = ()):
    hits = [p for p in find_all(root, pattern)
            if (must_contain is None or must_contain in str(p)) and not any(a in str(p) for a in avoid)]
    if len(hits) > 1:
        log(f"  note: {len(hits)} files match {pattern}; using {hits[0]}")
    return hits[0] if hits else None


def make_grid(step=0.1):
    vals = np.round(np.arange(0, 1 + 1e-9, step), 4)
    grid = np.array([(a, b1, b2) for a in vals for b1 in vals for b2 in vals])
    a, b1, b2 = grid.T
    return grid, np.stack([(1 - a) * (1 - b1), a * (1 - b2), (1 - a) * b1, a * b2], axis=1)


def dense_from_index(root: Path, dataset: str, ids: list[str]):
    """Embeddings in corpus order, taken from the FAISS index built in Step 1."""
    import faiss
    want = len(ids)
    for f in find_all(root, "index.faiss"):
        meta = f.with_name("index_meta.faiss")
        if not meta.exists() or f.stat().st_size > 4 * want * 768 * 1.2:   # skip big (other) corpora unread
            continue
        db_ids = [str(x) for x in pickle.load(meta.open("rb"))]
        if len(db_ids) != want or set(db_ids[:200]) - set(ids):
            continue
        log(f"dense: using {f} ({f.stat().st_size / 1e6:.0f} MB)")
        index = faiss.read_index(str(f))
        emb = index.reconstruct_n(0, index.ntotal).astype(np.float32)
        pos = {d: i for i, d in enumerate(db_ids)}
        return emb[[pos[d] for d in ids]], str(f)
    return None, None


def dense_by_encoding(corpus_rows):
    """Fallback: the same encoder as qudar.retrievers.DenseEncoder (prefix 'passage: ', title + text)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"dense: no matching FAISS index found -> encoding {len(corpus_rows)} passages on {dev}")
    tok = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever").to(dev).eval()
    if dev == "cuda":
        model = model.half()
    texts = ["passage: " + (" ".join(p for p in [r.get("title"), r.get("text")] if p)).strip() for r in corpus_rows]
    order = np.argsort([-len(t) for t in texts])
    out = np.zeros((len(texts), 768), np.float32)
    with torch.no_grad():
        for s in range(0, len(texts), 128):
            idx = order[s:s + 128]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), p=2, dim=1)
            out[idx] = v.cpu().numpy()
            if (s // 128) % 50 == 0:
                log(f"  {s + len(idx)}/{len(texts)}")
    return out, "re-encoded with facebook/contriever"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--out", default="/kaggle/working/demo_data")
    p.add_argument("--dataset", default="fiqa")
    p.add_argument("--split", default="test")
    p.add_argument("--train", nargs="+", default=["fiqa_train", "hotpotqa_train"])
    args = p.parse_args()
    root, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}_{args.split}"
    manifest = {"dataset": args.dataset, "split": args.split, "sources": {}}

    # ---- corpus, queries, qrels ----
    qfile = find(root, "queries.jsonl", must_contain=f"/{args.dataset}/")
    droot = qfile.parent
    rows = [json.loads(line) for line in (droot / "corpus.jsonl").open(encoding="utf8")]
    ids = [str(r["_id"]) for r in rows]
    with (out / "corpus.jsonl").open("w", encoding="utf8") as h:
        for r in rows:
            h.write(json.dumps({"_id": str(r["_id"]), "title": r.get("title") or "", "text": r.get("text") or ""},
                               ensure_ascii=False) + "\n")
    qrels = {}
    with (droot / "qrels" / f"{args.split}.tsv").open(encoding="utf8") as h:
        rd = csv.reader(h, delimiter="\t")
        next(rd)
        for q, d, s in rd:
            qrels.setdefault(q, {})[d] = int(s)
    queries = {}
    for line in qfile.open(encoding="utf8"):
        r = json.loads(line)
        if str(r["_id"]) in qrels:
            queries[str(r["_id"])] = r.get("text", "")
    json.dump(queries, open(out / "queries.json", "w", encoding="utf8"), ensure_ascii=False)
    json.dump(qrels, open(out / "qrels.json", "w"))
    manifest.update(n_docs=len(ids), n_queries=len(queries))
    log(f"corpus {len(ids)} docs, {len(queries)} {args.split} queries")

    # ---- expansions (Step 1) and LLM scores (Step 4) ----
    f = find(root, f"expanded_queries_{args.dataset}_{args.split}_expansion.json")
    exp = json.loads(f.read_text(encoding="utf8"))
    json.dump({q: exp.get(q, "") for q in queries}, open(out / "expansions.json", "w", encoding="utf8"), ensure_ascii=False)
    manifest["sources"]["expansions"] = str(f)
    f = find(root, f"llm_scores_paper_{name}.csv", avoid=("first",))
    llm = {}
    if f:
        with f.open(encoding="utf8") as h:
            for r in csv.DictReader(h):
                llm[r["qid"]] = {"OS": float(r["original_sparse"]), "OD": float(r["original_dense"]),
                                 "ES": float(r["expanded_sparse"]), "ED": float(r["expanded_dense"])}
        manifest["sources"]["llm_scores"] = str(f)
    json.dump(llm, open(out / "llm_scores.json", "w"))
    log(f"expansions {len(exp)}, cached LLM scores {len(llm)}")

    # ---- Step 6 models ----
    for fn, dst in [("model_A_both.txt", "model_A.txt"), ("model_B_both.txt", "model_B.txt"),
                    ("model_A_features.json", "model_A_features.json"), ("model_B_features.json", "model_B_features.json"),
                    ("position_bias_train.json", None), (f"chosen_weights_{name}.csv", "chosen_weights.csv")]:
        f = find(root, fn, avoid=("6b", "step6b"))
        if f is None:
            log(f"  WARNING: {fn} not found")
            continue
        manifest["sources"][fn] = str(f)
        if dst:
            shutil.copy(f, out / dst)
        else:
            manifest["position_bias"] = {int(k): float(v) for k, v in json.loads(f.read_text()).items()}

    # ---- Static-train point (= step6.static_train_point) and the fixed-weight baselines of Table 1 ----
    grid, W = make_grid()
    tr = [np.load(find(root, "grid_metrics.npz", must_contain=f"/{n}/")) for n in args.train]
    st = int(np.argmax(np.concatenate([z["metrics"][:, :, 1].astype(np.float32) for z in tr]).mean(0)))
    manifest["static_train"] = {"grid_index": st, "alpha_beta": grid[st].round(2).tolist(), "weights": W[st].round(4).tolist()}
    log(f"Static-train point (alpha, beta1, beta2) = {grid[st].round(2).tolist()}")
    zt = find(root, "grid_metrics.npz", must_contain=f"/{name}/")
    if zt:
        z = np.load(zt)
        G = z["metrics"][:, :, 1].astype(np.float32).mean(0)
        a, b1, b2 = grid.T
        for key, mask in [("retriever_type_fusion", np.isclose(b1, 0) & np.isclose(b2, 0)),
                          ("query_format_fusion", np.isclose(a, 1) & np.isclose(b1, 0))]:
            i = int(np.flatnonzero(mask)[np.argmax(G[mask])])
            manifest[key] = {"grid_index": i, "alpha_beta": grid[i].round(2).tolist(), "weights": W[i].round(4).tolist()}

    # ---- cached Step 2 lists (exact reproduction mode in the demo) ----
    f = find(root, "retrieval_result.json", must_contain=f"/{name}/")
    rr = json.loads(f.read_text())
    pos = {d: i for i, d in enumerate(ids)}
    qids = [q for q in queries if q in rr]
    K = max(len(rr[q][dk]) for q in qids for dk, _ in KEYS)
    D = np.full((len(qids), 4, K), -1, np.int32)
    S = np.full((len(qids), 4, K), np.nan, np.float32)
    for i, q in enumerate(qids):
        for j, (dk, sk) in enumerate(KEYS):
            n = len(rr[q][dk])
            D[i, j, :n] = [pos[d] for d in rr[q][dk]]
            S[i, j, :n] = rr[q][sk]
    np.savez_compressed(out / "cached_runs.npz", qids=np.array(qids), docs=D, scores=S)
    manifest["sources"]["cached_runs"] = str(f)
    del rr
    log(f"cached lists: {len(qids)} queries x 4 x {K}")

    # ---- dense embeddings ----
    emb, src = dense_from_index(root, args.dataset, ids)
    if emb is None:
        emb, src = dense_by_encoding(rows)
    np.save(out / "dense_emb.npy", emb.astype(np.float16))
    manifest["sources"]["dense"] = src
    manifest["dense_dim"] = int(emb.shape[1])

    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    log("files: " + ", ".join(f"{p.name} {p.stat().st_size / 1e6:.1f} MB" for p in sorted(out.iterdir())))
    z = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    log(f"Saved {z} ({Path(z).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
