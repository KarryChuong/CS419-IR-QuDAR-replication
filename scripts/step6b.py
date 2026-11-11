"""Step 6.3b: one pre-registered revision of the learned weighting (run once, report as is).

Changes vs step6.py (everything else reused unchanged):
  1. Gated model = Static-train + adaptation: a query leaves the static weights only when the
     surface model predicts a gain above a threshold t. Train queries are split 80 / 20 (fit /
     validation, fixed seed); the surface model is fitted on the 80%, t is chosen on the 20%
     (max mean nDCG@10, ties -> larger t = more conservative). The fallback is exactly the
     Static-train point (best fixed weights on all train queries), so "Gated vs Static-train"
     measures only the query-wise adaptation. The test sets are never used for any choice.
  2. Scale-free feature sets A-sf / B-sf: drop raw-score features whose scale differs between
     corpora (*_raw_top1, *_raw_margin, *_raw_std10); keep ratios, normalised scores, overlaps,
     query features (and LLM scores for B-sf).
  3. More paired tests (Static-train vs QuDAR, Direct vs Static-train, Gated vs Static-train ...)
     with Holm-adjusted p-values per dataset and metric.
  4. Tables rounded to 4 decimals.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step6 import (LABEL, M, TEST, TRAIN, coarse_points, load_run, log, paired_tests,  # noqa: E402
                   position_bias, predict_direct, static_train_point, surface_rows, train_direct, train_surface)

SCALE_DEP = ("_raw_top1", "_raw_margin", "_raw_std10")


def subset(run, mask):
    return {**run, "qids": [q for q, m in zip(run["qids"], mask) if m], "X": run["X"][mask], "G": run["G"][mask]}


def split_runs(runs, frac=0.8, seed=0):
    fit, val = [], []
    for r in runs:
        m = np.random.default_rng(seed).random(len(r["qids"])) < frac
        fit.append(subset(r, m))
        val.append(subset(r, ~m))
    return fit, val


def predict_points(model, run, cols, idx, chunk=400):
    X = run["X"][cols].to_numpy(np.float32)
    out = np.empty((len(X), len(idx)), dtype=np.float32)
    for s in range(0, len(X), chunk):
        rows = surface_rows(X[s:s + chunk], run["grid"], run["weights"], idx)
        out[s:s + chunk] = model.predict(rows, num_iteration=model.best_iteration_).reshape(-1, len(idx))
    return out


def gated_choice(model, run, cols, static, t):
    """Grid index per query: best predicted coarse point if its predicted gain over the static point > t."""
    coarse = coarse_points(run["grid"])
    idx = np.append(coarse, static)
    pred = predict_points(model, run, cols, idx)
    best = pred[:, :-1].argmax(1)
    gain = pred[np.arange(len(best)), best] - pred[:, -1]
    return np.where(gain > t, coarse[best], static), gain


def choose_threshold(model, val_runs, cols, static):
    gains, nd_model, nd_static = [], [], []
    for r in val_runs:
        coarse = coarse_points(r["grid"])
        choice, g = gated_choice(model, r, cols, static, -np.inf)
        gains.append(g)
        nd_model.append(r["G"][np.arange(len(choice)), choice, 1])
        nd_static.append(r["G"][:, static, 1])
    g, a, b = np.concatenate(gains), np.concatenate(nd_model), np.concatenate(nd_static)
    cands = np.unique(np.concatenate([[-np.inf, np.inf], np.quantile(g, np.linspace(0, 0.95, 20))]))
    scores = [np.where(g > t, a, b).mean() for t in cands]
    best = max(range(len(cands)), key=lambda i: (round(scores[i], 6), cands[i]))
    return float(cands[best]), float(scores[best]), float(b.mean()), float(a.mean())


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--input", default="/kaggle/input")
    pa.add_argument("--out", required=True)
    pa.add_argument("--test", nargs="+", default=TEST)
    pa.add_argument("--train", nargs="+", default=TRAIN)
    args = pa.parse_args()
    root, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bias = position_bias(root, args.train)
    train_runs = [load_run(root, n, bias) for n in args.train]
    test_runs = [load_run(root, n, bias) for n in args.test]
    feats = list(train_runs[0]["X"].columns)
    fsets = {"A": [c for c in feats if not c.startswith("llm_")]}
    fsets["A-sf"] = [c for c in fsets["A"] if not c.endswith(SCALE_DEP)]
    fsets["B"] = feats
    fsets["B-sf"] = fsets["A-sf"] + [c for c in feats if c.startswith("llm_")]
    log("feature sets: " + ", ".join(f"{k}={len(v)}" for k, v in fsets.items()))

    configs = {"both": train_runs, **{r["name"].rsplit("_", 1)[0]: [r] for r in train_runs}}
    chosen, info = {}, []
    for cfg, runs in configs.items():
        fit, val = split_runs(runs)
        static_all = static_train_point(runs)
        chosen[("Static-train", cfg)] = {r["name"]: np.full(len(r["qids"]), static_all) for r in test_runs}
        for fs, cols in fsets.items():
            log(f"[{cfg}] surface model {fs}")
            model = train_surface(fit, cols)
            t, val_gated, val_static, val_model = choose_threshold(model, val, cols, static_all)
            for r in test_runs:
                n = r["name"]
                ungated, _ = gated_choice(model, r, cols, static_all, -np.inf)
                gated, _ = gated_choice(model, r, cols, static_all, t)
                chosen.setdefault((f"Model {fs}", cfg), {})[n] = ungated
                chosen.setdefault((f"Gated {fs}", cfg), {})[n] = gated
                info.append({"train on": cfg, "features": fs, "test": LABEL.get(n, n), "threshold": round(t, 5),
                             "val nDCG static": round(val_static, 4), "val nDCG model": round(val_model, 4),
                             "val nDCG gated": round(val_gated, 4),
                             "test queries leaving static %": round(100 * float((gated != static_all).mean()), 1)})
        if cfg == "both":
            for tag in ("A", "B"):
                dm = train_direct(runs, fsets[tag])
                chosen[(f"Direct-{tag}", cfg)] = {r["name"]: predict_direct(dm, r, fsets[tag]) for r in test_runs}
    pd.DataFrame(info).to_csv(out / "table6b_gate_info.csv", index=False)

    perq = {}
    for r in test_runs:
        d = dict(r["rule"])
        for (meth, cfg), sel in chosen.items():
            d[f"{meth} [{cfg}]"] = r["G"][np.arange(len(r["qids"])), sel[r["name"]]]
        d["Static UB (oracle)"] = r["G"][:, int(np.argmax(r["G"][:, :, 1].mean(0)))]
        best = np.argmax(r["G"][:, :, 1] * 1e3 + r["G"][:, :, 0], axis=1)
        d["Query-wise UB (oracle)"] = r["G"][np.arange(len(best)), best]
        perq[r["name"]] = d

    order = ["QuDAR-simple (Equal)", "QuDAR-confidence", "QuDAR-llm", "QuDAR-llm-debiased",
             "Static-train [both]", "Static-train [fiqa]", "Direct-A [both]", "Direct-B [both]"]
    order += [f"{k} {fs} [both]" for fs in fsets for k in ("Model", "Gated")]
    order += ["Static UB (oracle)", "Query-wise UB (oracle)"]
    rows = []
    for n, d in perq.items():
        conf, qub = d["QuDAR-confidence"].mean(0), d["Query-wise UB (oracle)"].mean(0)
        for meth in order:
            if meth in d:
                m = d[meth].mean(0)
                rows.append({"dataset": LABEL.get(n, n), "method": meth, **{k: round(float(v), 4) for k, v in zip(M, m)},
                             "gap closed % (Recall)": round(float(100 * (m[0] - conf[0]) / max(qub[0] - conf[0], 1e-9)), 1)})
    main_t = pd.DataFrame(rows)
    main_t.to_csv(out / "table6b_main.csv", index=False)
    (out / "table6b_main.md").write_text(main_t.to_markdown(index=False))
    print("\n" + main_t.to_string(index=False), flush=True)

    tc = []
    for cfg in configs:
        for meth in ["Static-train"] + [f"{k} {fs}" for fs in fsets for k in ("Model", "Gated")]:
            key = f"{meth} [{cfg}]"
            tc.append({"method": meth, "train on": cfg,
                       **{LABEL.get(n, n): round(float(perq[n][key].mean(0)[0]), 4) for n in perq if key in perq[n]}})
    tc = pd.DataFrame(tc)
    tc.to_csv(out / "table6b_train_config.csv", index=False)
    (out / "table6b_train_config.md").write_text(tc.to_markdown(index=False))
    print("\n" + tc.to_string(index=False), flush=True)

    pairs = [("Static-train [both]", "QuDAR-confidence"), ("Static-train [both]", "QuDAR-llm"),
             ("Direct-A [both]", "Static-train [both]"), ("Direct-B [both]", "Static-train [both]"),
             ("Gated A [both]", "Static-train [both]"), ("Gated A-sf [both]", "Static-train [both]"),
             ("Gated B [both]", "Static-train [both]"), ("Gated B-sf [both]", "Static-train [both]"),
             ("Gated A [both]", "QuDAR-confidence"), ("Gated B [both]", "QuDAR-llm"),
             ("Gated A-sf [fiqa]", "Static-train [fiqa]"), ("QuDAR-llm-debiased", "QuDAR-llm")]
    sig = []
    for n, d in perq.items():
        for mi, metric in [(1, "nDCG@10"), (0, "Recall@10")]:
            block = []
            for a, b in pairs:
                if a in d and b in d:
                    diff, lo, hi, pb, pt = paired_tests(d[a][:, mi], d[b][:, mi])
                    block.append({"dataset": LABEL.get(n, n), "A": a, "B": b, "metric": metric, "mean(A-B)": round(float(diff), 4),
                                  "95% CI": f"[{lo:.4f}, {hi:.4f}]", "p bootstrap": round(pb, 4), "p t-test": round(pt, 4)})
            adj = holm([max(x["p bootstrap"], x["p t-test"]) for x in block])
            for x, h in zip(block, adj):
                x["p Holm"] = round(float(h), 4)
                x["sig (Holm p<0.05)"] = "*" if h < 0.05 else ""
            sig += block
    sig = pd.DataFrame(sig)
    sig.to_csv(out / "table6b_significance.csv", index=False)
    (out / "table6b_significance.md").write_text(sig.to_markdown(index=False))
    print("\n" + sig[sig.metric == "nDCG@10"].to_string(index=False), flush=True)
    print("\n" + pd.DataFrame(info).to_string(index=False), flush=True)
    log(f"Saved Step 6.3b outputs to {out}")


if __name__ == "__main__":
    main()
