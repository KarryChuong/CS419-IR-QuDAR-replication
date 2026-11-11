"""Step 3 + Step 5: dual-perspective weight grid for every query.

Weights over the four lists follow the paper's two axes:
  w_OS = (1-a)(1-b1), w_ES = (1-a) b1, w_OD = a (1-b2), w_ED = a b2
with a = retriever-type weight (sparse 0 -> dense 1), b1 / b2 = query-format weight inside
sparse / dense (original 0 -> expanded 1), each on a 0.0, 0.1, ..., 1.0 grid (1331 points).
Fusion = weighted sum of qudar's min-max normalised scores (as qudar.fusion._fuse_scores);
metrics = Recall@10, nDCG@10 (gain = qrel grade, as trec_eval), P@1.

Per run folder it writes <out>/<name>/:
  grid_metrics.npz   qids, grid (G x 3: a, b1, b2), weights (G x 4: OS, OD, ES, ED),
                     metrics (Q x G x 3, float16): per-query Recall@10, nDCG@10, P@1
  oracle_labels.csv  per query: best grid point (max nDCG@10, then Recall@10, then the
                     point closest to equal weights) -> labels for Step 6
  summary.csv        Static Avg / Static UB / query-wise UB for the 3 single-perspective
                     axes and the dual grid, plus checks against eval_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCES = [("sparse_docs", "sparse_scores_normalized"), ("dense_docs", "dense_scores_normalized"),
           ("pgt_sparse_docs", "pgt_sparse_scores_normalized"), ("pgt_dense_docs", "pgt_dense_scores_normalized")]
METRICS = ["Recall@10", "nDCG@10", "Precision@1"]


def load_qrels(data_root: Path, dataset: str, split: str) -> dict:
    qrels: dict[str, dict[str, int]] = {}
    with (data_root / dataset / "qrels" / f"{split}.tsv").open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)
        for q, d, s in reader:
            qrels.setdefault(q, {})[d] = int(s)
    return qrels


def make_grid(step: float):
    vals = np.round(np.arange(0, 1 + 1e-9, step), 4)
    grid = np.array([(a, b1, b2) for a in vals for b1 in vals for b2 in vals])
    a, b1, b2 = grid.T
    weights = np.stack([(1 - a) * (1 - b1), a * (1 - b2), (1 - a) * b1, a * b2], axis=1)  # OS, OD, ES, ED
    return grid, weights


def query_metrics(entry: dict, rel: dict, weights: np.ndarray, k: int = 10) -> np.ndarray:
    doc_index: dict[str, int] = {}
    rows, cols, vals = [], [], []
    for col, (dk, sk) in enumerate(SOURCES):
        for doc, score in zip(entry[dk], entry[sk]):
            rows.append(doc_index.setdefault(doc, len(doc_index)))
            cols.append(col)
            vals.append(score)
    S = np.zeros((len(doc_index), 4))
    np.add.at(S, (np.array(rows), np.array(cols)), np.array(vals, dtype=float))  # duplicates add up, as in qudar
    fused = S @ weights.T                                                       # U x G
    kk = min(k, fused.shape[0])
    top = np.argpartition(-fused, kk - 1, axis=0)[:kk]
    order = np.argsort(-np.take_along_axis(fused, top, axis=0), axis=0, kind="stable")
    top = np.take_along_axis(top, order, axis=0)                                # kk x G, ranked
    gains = np.zeros(len(doc_index))
    for doc, g in rel.items():
        if g > 0 and doc in doc_index:
            gains[doc_index[doc]] = g
    n_rel = sum(1 for g in rel.values() if g > 0)
    tg = gains[top]
    out = np.zeros((weights.shape[0], 3))
    if n_rel == 0:
        return out
    disc = 1 / np.log2(np.arange(2, kk + 2))[:, None]
    ideal = np.sort([g for g in rel.values() if g > 0])[::-1][:k]
    idcg = float((ideal / np.log2(np.arange(2, len(ideal) + 2))).sum())
    out[:, 0] = (tg > 0).sum(axis=0) / n_rel
    out[:, 1] = (tg * disc).sum(axis=0) / idcg
    out[:, 2] = (tg[0] > 0).astype(float)
    return out


def summarise(metrics: np.ndarray, mask: np.ndarray) -> dict:
    sub = metrics[:, mask, :]                          # Q x g x 3
    per_point = sub.mean(axis=0)                       # g x 3
    best_point = int(np.argmax(per_point[:, 1] * 1e3 + per_point[:, 0]))
    key = sub[:, :, 1] * 1e3 + sub[:, :, 0]
    qbest = np.argmax(key, axis=1)
    dyn = sub[np.arange(sub.shape[0]), qbest].mean(axis=0)
    # Static UB = best weight per metric, as qudar.evaluation does (performance.max over the grid)
    return {"avg": per_point.mean(axis=0), "ub": per_point.max(axis=0), "dyn": dyn,
            "best_index": np.flatnonzero(mask)[best_point]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="folder holding <dataset>_<split>/retrieval_result.json")
    p.add_argument("--data-root", required=True)
    p.add_argument("--names", nargs="+", required=True, help="e.g. fiqa_test hotpotqa_train")
    p.add_argument("--out", required=True)
    p.add_argument("--step", type=float, default=0.1)
    args = p.parse_args()
    grid, weights = make_grid(args.step)
    a, b1, b2 = grid.T
    eq = np.isclose(a, 0.5) & np.isclose(b1, 0.5) & np.isclose(b2, 0.5)
    dist_to_equal = np.abs(weights - 0.25).sum(axis=1)

    for name in args.names:
        dataset, split = name.rsplit("_", 1)
        run_dir = Path(args.runs) / name
        rr = json.loads((run_dir / "retrieval_result.json").read_text())
        qrels = load_qrels(Path(args.data_root), dataset, split)
        qids = list(rr)
        metrics = np.zeros((len(qids), len(grid), 3), dtype=np.float32)
        for i, q in enumerate(qids):
            metrics[i] = query_metrics(rr[q], qrels.get(q, {}), weights)
            if i and i % 2000 == 0:
                print(f"  {name}: {i}/{len(qids)}", flush=True)
        del rr
        out_dir = Path(args.out) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "grid_metrics.npz", qids=np.array(qids), grid=grid, weights=weights,
                            metrics=metrics.astype(np.float16))

        # oracle labels: max nDCG, then Recall, then closest to equal weights
        key = metrics[:, :, 1] * 1e6 + metrics[:, :, 0] * 1e3 - dist_to_equal[None, :] * 1e-3
        best = np.argmax(key, axis=1)
        lab = pd.DataFrame({"qid": qids, "grid_index": best, "alpha": a[best], "beta1": b1[best], "beta2": b2[best],
                            "w_OS": weights[best, 0], "w_OD": weights[best, 1], "w_ES": weights[best, 2],
                            "w_ED": weights[best, 3], "best_nDCG@10": metrics[np.arange(len(qids)), best, 1],
                            "n_ties": (metrics[:, :, 1] == metrics[np.arange(len(qids)), best, 1][:, None]).sum(axis=1)})
        lab.to_csv(out_dir / "oracle_labels.csv", index=False)

        axes = {"retriever_type (OS-OD)": np.isclose(b1, 0) & np.isclose(b2, 0),
                "query_format_sparse (OS-ES)": np.isclose(a, 0) & np.isclose(b2, 0),
                "query_format_dense (OD-ED)": np.isclose(a, 1) & np.isclose(b1, 0),
                "dual (4 lists)": np.ones(len(grid), dtype=bool)}
        rows = []
        for axis, mask in axes.items():
            s = summarise(metrics, mask)
            for kind in ("avg", "ub", "dyn"):
                label = {"avg": "Static Avg", "ub": "Static UB", "dyn": "Query-wise UB"}[kind]
                row = {"axis": axis, "method": label, **dict(zip(METRICS, np.round(s[kind], 4)))}
                if kind == "ub":
                    row["best (a, b1, b2)"] = tuple(np.round(grid[s["best_index"]], 2))
                rows.append(row)
        corners = {"os": (0, 0, 0), "od": (1, 0, 0), "es": (0, 1, 0), "ed": (1, 0, 1)}
        for m, (ca, cb1, cb2) in corners.items():
            idx = np.flatnonzero(np.isclose(a, ca) & np.isclose(b1, cb1) & np.isclose(b2, cb2))[0]
            rows.append({"axis": "check", "method": m, **dict(zip(METRICS, np.round(metrics[:, idx].mean(0), 4)))})
        rows.append({"axis": "check", "method": "qudar_simple_equal",
                     **dict(zip(METRICS, np.round(metrics[:, np.flatnonzero(eq)[0]].mean(0), 4)))})
        summary = pd.DataFrame(rows)
        ev_file = run_dir / "eval_results.csv"
        if ev_file.exists():
            ref = pd.read_csv(ev_file).set_index("method")
            name_map = {"os": "os", "od": "od", "es": "es", "ed": "ed", "qudar_simple_equal": "qudar_simple_equal"}
            chk = summary[summary.axis == "check"]
            for _, r in chk.iterrows():
                if name_map[r.method] not in ref.index:
                    continue
                ref_row = ref.loc[name_map[r.method]]
                diff = max(abs(r[m] - ref_row[m]) for m in METRICS)
                summary.loc[(summary.axis == "check") & (summary.method == r.method), "max |diff| vs eval_results"] = round(diff, 4)
            for axis, pre in [("retriever_type (OS-OD)", "os_od"), ("query_format_sparse (OS-ES)", "os_es"),
                              ("query_format_dense (OD-ED)", "od_ed")]:
                for label, m in [("Static Avg", f"sha_{pre}"), ("Static UB", f"shb_{pre}")]:
                    if m not in ref.index:
                        continue
                    r = summary[(summary.axis == axis) & (summary.method == label)].iloc[0]
                    diff = max(abs(r[x] - ref.loc[m][x]) for x in METRICS)
                    summary.loc[(summary.axis == axis) & (summary.method == label), "max |diff| vs eval_results"] = round(diff, 4)
        summary.to_csv(out_dir / "summary.csv", index=False)
        print(f"\n=== {name} ({len(qids)} queries) ===")
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
