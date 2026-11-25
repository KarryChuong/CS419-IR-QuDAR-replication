"""Step 10d-2: static weights tuned on train data vs QuDAR, on 8 test sets and 5 train sets.
CPU only, no API calls. Reuses the Step 6 code (features, grid metrics, LightGBM surface model).

Everything is read recursively from --input (Kaggle: /kaggle/input), as step6.py does:
  <name>/retrieval_result.json, <name>/grid_metrics.npz, llm_scores_paper_<name>.csv (gpt-4o-mini),
  llm_paper_<name>.jsonl, <dataset>/queries.jsonl + qrels, expanded_queries_<dataset>_<split>_expansion.json
Attach only the gpt-4o-mini score datasets: a Qwen score file with the same name would be picked up too.

Methods on every test set
  rule, no LLM : QuDAR-simple (Equal), QuDAR-confidence
  rule, LLM    : QuDAR-llm (gpt-4o-mini, paper prompt, shuffled), QuDAR-llm-debiased (Step 6.4)
  static       : Static-train [in-domain]  best fixed weights (max mean nDCG@10 over the 1331-point grid) on the
                                           train set of the same dataset (5 datasets have one)
                 Static-train [other]      same, on the train sets of the OTHER datasets (no in-domain label)
                 Static-train [all]        same, on all 5 train sets pooled
  learned      : Model A / Model B [all]   Step 6 surface model (A = retrieval + query features, B = A + LLM
                                           scores and positions), trained on all 5 train sets pooled
  oracle       : Static UB (best fixed weights on the test set), Query-wise UB

Families, fixed before looking at the results; Holm inside each family; p = max(bootstrap, t-test),
10,000 bootstrap resamples (as Step 6 / 10-B2):
  S0 confidence_vs_equal  QuDAR-confidence - Equal                                    8 sets x 2 metrics = 16
  S1 static_in_domain     Static-train [in-domain] - Equal / - QuDAR-confidence / - QuDAR-llm   5 x 3 x 2 = 30
  S2 static_transfer      Static-train [other] - QuDAR-confidence                      8 x 2 = 16
  S3 learned_vs_static    Model A [all] - Static-train [all], Model B [all] - Static-train [all]  8 x 2 x 2 = 32

Label budget (descriptive, RQ3 "how many labels"): for each dataset with a train set, n random train queries
(n = 10 ... all, 100 draws each) choose the static point; we report the test nDCG@10 / Recall@10 it reaches
(median, 5th and 95th percentile over draws) and the share of draws that beat QuDAR-confidence on the test set.

Outputs in --out: table_main.csv/.md, sig_tests.csv/.md, static_points.csv/.md, label_budget.csv/.md,
fig_label_budget.png, check.csv, summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_significance import holm, paired_tests  # noqa: E402
from step6 import find, load_run, position_bias, predict_surface, static_train_point, train_surface  # noqa: E402

TEST = ["fiqa_test", "scidocs_test", "climate-fever_test", "hotpotqa_test",
        "scifact_test", "nfcorpus_test", "dbpedia-entity_test", "fever_test"]
TRAIN = ["fiqa_train", "hotpotqa_train", "scifact_train", "nfcorpus_train", "fever_train"]
LABEL = {"fiqa": "FiQA", "scidocs": "SciDocs", "climate-fever": "Climate-FEVER", "hotpotqa": "HotpotQA",
         "scifact": "SciFact", "nfcorpus": "NFCorpus", "dbpedia-entity": "DBPedia-entity", "fever": "FEVER"}
METRICS = [(0, "Recall@10"), (1, "nDCG@10")]
BUDGET = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ds(name):
    return name.rsplit("_", 1)[0]


def to_md(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--out", required=True)
    p.add_argument("--test", nargs="+", default=TEST)
    p.add_argument("--train", nargs="+", default=TRAIN)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--draws", type=int, default=100)
    p.add_argument("--no-models", action="store_true", help="skip the LightGBM models (S3)")
    args = p.parse_args()
    root, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bias = position_bias(root, args.train)
    log(f"position bias from train (score at position - mean): {bias}")
    train_runs = {ds(n): load_run(root, n, bias) for n in args.train}
    test_runs = [load_run(root, n, bias) for n in args.test]
    grid, weights = test_runs[0]["grid"], test_runs[0]["weights"]
    for r in list(train_runs.values()) + test_runs:
        assert np.allclose(r["grid"], grid), f"{r['name']}: different weight grid"

    # ---------- static points ----------
    static = {"all": static_train_point(list(train_runs.values()))}
    for d, r in train_runs.items():
        static[f"in-domain:{d}"] = static_train_point([r])
    for r in test_runs:
        others = [t for d, t in train_runs.items() if d != ds(r["name"])]
        static[f"other:{ds(r['name'])}"] = static_train_point(others)
    sp = pd.DataFrame([{"config": k, "alpha": grid[i][0], "beta1": grid[i][1], "beta2": grid[i][2],
                        **{f"w_{s}": round(float(w), 3) for s, w in zip(["OS", "OD", "ES", "ED"], weights[i])}}
                       for k, i in static.items()])
    sp.to_csv(out / "static_points.csv", index=False)
    (out / "static_points.md").write_text(to_md(sp))
    log("static points:\n" + sp.to_string(index=False))

    # ---------- learned models (Step 6 surface model) on all train sets pooled ----------
    chosen = {}
    if not args.no_models:
        feat_all = list(test_runs[0]["X"].columns)
        cols = {"A": [c for c in feat_all if not c.startswith("llm_")], "B": feat_all}
        has_llm = all(not r["X"]["llm_OS"].isna().all() for r in train_runs.values())
        for tag, c in cols.items():
            if tag == "B" and not has_llm:
                log("WARNING: a train set has no LLM scores -> Model B skipped")
                continue
            log(f"train surface model {tag} on {len(train_runs)} train sets")
            model = train_surface(list(train_runs.values()), c)
            chosen[f"Model {tag} [all]"] = {r["name"]: predict_surface(model, r, c) for r in test_runs}

    # ---------- per-query metrics ----------
    perq = {}
    for r in test_runs:
        n, d = r["name"], ds(r["name"])
        rows = np.arange(len(r["qids"]))
        m = dict(r["rule"])
        if f"in-domain:{d}" in static:
            m["Static-train [in-domain]"] = r["G"][:, static[f"in-domain:{d}"]]
        m["Static-train [other]"] = r["G"][:, static[f"other:{d}"]]
        m["Static-train [all]"] = r["G"][:, static["all"]]
        for meth, sel in chosen.items():
            m[meth] = r["G"][rows, sel[n]]
        m["Static UB (oracle)"] = r["G"][:, int(np.argmax(r["G"][:, :, 1].mean(0)))]
        best = np.argmax(r["G"][:, :, 1] * 1e3 + r["G"][:, :, 0], axis=1)
        m["Query-wise UB (oracle)"] = r["G"][rows, best]
        perq[n] = {k: np.asarray(v, float) for k, v in m.items()}

    order = ["QuDAR-simple (Equal)", "QuDAR-confidence", "QuDAR-llm", "QuDAR-llm-debiased",
             "Static-train [in-domain]", "Static-train [other]", "Static-train [all]",
             "Model A [all]", "Model B [all]", "Static UB (oracle)", "Query-wise UB (oracle)"]
    main_rows = []
    for n, m in perq.items():
        for meth in order:
            if meth in m:
                mu = m[meth].mean(0)
                main_rows.append({"dataset": LABEL[ds(n)], "method": meth, "queries": len(m[meth]),
                                  "Recall@10": round(float(mu[0]), 4), "nDCG@10": round(float(mu[1]), 4),
                                  "P@1": round(float(mu[2]), 4)})
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(out / "table_main.csv", index=False)
    (out / "table_main.md").write_text(to_md(main_df))

    # ---------- check against the numbers of Steps 2 / 4 / 10c ----------
    chk = []
    for n, m in perq.items():
        ev = find(root, "eval_results.csv", n)
        if ev is not None:
            ref = pd.read_csv(ev).set_index("method")
            for meth, key in [("QuDAR-simple (Equal)", "qudar_simple_equal"), ("QuDAR-confidence", "qudar_confidence")]:
                if key in ref.index:
                    mu = m[meth].mean(0)
                    chk.append({"set": n, "method": meth, "ours Recall@10": round(float(mu[0]), 4),
                                "reported": float(ref.loc[key, "Recall@10"]), "ours nDCG@10": round(float(mu[1]), 4),
                                "reported nDCG": float(ref.loc[key, "nDCG@10"]),
                                "max |diff|": round(max(abs(mu[0] - ref.loc[key, "Recall@10"]),
                                                        abs(mu[1] - ref.loc[key, "nDCG@10"])), 4)})
    chk = pd.DataFrame(chk)
    chk.to_csv(out / "check.csv", index=False)

    # ---------- paired tests, Holm per family ----------
    tests = []

    def add(family, n, a, b):
        m = perq[n]
        if a not in m or b not in m:
            return
        for mi, metric in METRICS:
            diff, lo, hi, pb, pt = paired_tests(m[a][:, mi], m[b][:, mi], args.n_boot)
            tests.append({"family": family, "set": LABEL[ds(n)], "A": a, "B": b, "metric": metric,
                          "queries": len(m[a]), "mean A": round(float(m[a][:, mi].mean()), 4),
                          "mean B": round(float(m[b][:, mi].mean()), 4), "mean(A-B)": round(diff, 4),
                          "95% CI": f"[{lo:.4f}, {hi:.4f}]", "p bootstrap": pb, "p t-test": pt, "p": max(pb, pt)})

    for n in perq:
        add("S0 confidence_vs_equal", n, "QuDAR-confidence", "QuDAR-simple (Equal)")
    for n in perq:
        for b in ("QuDAR-simple (Equal)", "QuDAR-confidence", "QuDAR-llm"):
            add("S1 static_in_domain", n, "Static-train [in-domain]", b)
    for n in perq:
        add("S2 static_transfer", n, "Static-train [other]", "QuDAR-confidence")
    for n in perq:
        for a in ("Model A [all]", "Model B [all]"):
            add("S3 learned_vs_static", n, a, "Static-train [all]")
    tests = pd.DataFrame(tests)
    tests["p Holm"] = np.nan
    for _, idx in tests.groupby("family").groups.items():
        tests.loc[idx, "p Holm"] = holm(tests.loc[idx, "p"].values)
    tests["sig (Holm p<0.05)"] = np.where(tests["p Holm"] < 0.05, "*", "")
    for c in ("p bootstrap", "p t-test", "p", "p Holm"):
        tests[c] = tests[c].round(4)
    tests.to_csv(out / "sig_tests.csv", index=False)
    (out / "sig_tests.md").write_text(to_md(tests))

    # ---------- label budget ----------
    rng = np.random.default_rng(0)
    budget = []
    for r in test_runs:
        d = ds(r["name"])
        if d not in train_runs:
            continue
        tr = train_runs[d]["G"][:, :, 1].astype(np.float64)
        te = r["G"].astype(np.float64).mean(0)                   # grid points x 3 metrics, test means
        conf = perq[r["name"]]["QuDAR-confidence"].mean(0)
        equal = perq[r["name"]]["QuDAR-simple (Equal)"].mean(0)
        sizes = [b for b in BUDGET if b < len(tr)] + [len(tr)]
        for nq in sizes:
            k = 1 if nq == len(tr) else args.draws
            picks = [int(np.argmax(tr[rng.choice(len(tr), nq, replace=False)].mean(0))) for _ in range(k)]
            vals = te[picks]
            budget.append({"dataset": LABEL[d], "train queries": nq, "draws": k,
                           "nDCG@10 median": round(float(np.median(vals[:, 1])), 4),
                           "nDCG@10 p05": round(float(np.percentile(vals[:, 1], 5)), 4),
                           "nDCG@10 p95": round(float(np.percentile(vals[:, 1], 95)), 4),
                           "Recall@10 median": round(float(np.median(vals[:, 0])), 4),
                           "share > QuDAR-confidence (nDCG)": round(float((vals[:, 1] > conf[1]).mean()), 3),
                           "share > Equal (nDCG)": round(float((vals[:, 1] > equal[1]).mean()), 3),
                           "QuDAR-confidence nDCG@10": round(float(conf[1]), 4)})
    budget = pd.DataFrame(budget)
    budget.to_csv(out / "label_budget.csv", index=False)
    (out / "label_budget.md").write_text(to_md(budget))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = list(dict.fromkeys(budget["dataset"]))
    fig, axes = plt.subplots(1, len(sets), figsize=(3.4 * len(sets), 3.2), squeeze=False)
    for ax, name in zip(axes[0], sets):
        b = budget[budget["dataset"] == name]
        ax.plot(b["train queries"], b["nDCG@10 median"], marker="o", label="Static-train (median)")
        ax.fill_between(b["train queries"], b["nDCG@10 p05"], b["nDCG@10 p95"], alpha=.25, label="5-95%")
        ax.axhline(b["QuDAR-confidence nDCG@10"].iloc[0], color="gray", ls="--", label="QuDAR-confidence")
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("labelled train queries")
        ax.grid(alpha=.3)
    axes[0][0].set_ylabel("test nDCG@10")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_label_budget.png", dpi=200)
    plt.close(fig)

    summary = {"n_boot": args.n_boot, "draws": args.draws, "position_bias_train": bias,
               "max_check_diff": float(chk["max |diff|"].max()) if len(chk) else None,
               "families": {f: {"tests": int(len(g)), "significant": int((g["sig (Holm p<0.05)"] == "*").sum()),
                                "significant, A > B": int(((g["sig (Holm p<0.05)"] == "*") & (g["mean(A-B)"] > 0)).sum()),
                                "significant, A < B": int(((g["sig (Holm p<0.05)"] == "*") & (g["mean(A-B)"] < 0)).sum())}
                            for f, g in tests.groupby("family")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"saved to {out}")


if __name__ == "__main__":
    main()
