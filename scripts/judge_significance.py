"""Step 10-B2: paired significance tests for the LLM-judge results, with Holm correction.

Compares the two judges of QuDAR-llm: gpt-4o-mini (Steps 4 and 6) and Qwen2.5-7B-Instruct (Step 10-B1).
CPU only, no API calls, no GPU.

Per-query Recall@10 and nDCG@10 are recomputed for every method from <runs>/<name>/retrieval_result.json
with grid_sweep.query_metrics (the fusion of qudar.fusion; checked against llm_eval.csv and
eval_results.csv in check_means.csv). LLM weights = softmax(scores / 3), equal weights when all four
scores are 0 (qudar.fusion.build_llm_weight_lookup); a query without LLM scores gets equal weights, as
llm_score.py does. On a "<name>_first<N>" set the queries are the first N of retrieval_result.json, as
llm_score.py --max-queries takes them (a score file may hold more rows: its cache was copied in).

Families, fixed before looking at the results; Holm is applied inside each family:
  F1 llm_vs_equal  (one family per judge)
       QuDAR-llm, paper prompt + shuffled, full test sets      - Equal
       QuDAR-llm, code prompt + shuffled, ablation sets         - Equal       4 sets x 2 rows x 2 metrics = 16
  F2 judge         QuDAR-llm, paper prompt + shuffled, full sets: judge 2 - judge 1                 4 x 2 = 8
  F3 ablation      (one family per judge) on the ablation sets
       order  | paper prompt : fixed - shuffled        order  | code prompt : fixed - shuffled
       prompt | shuffled     : code - paper            prompt | fixed       : code - paper    4 x 4 x 2 = 32
  F4 position      (one family per judge) "the mean score depends on the position in the prompt",
                   shuffled modes only: paper on the full sets, code_shuffle on the ablation sets.    8

Paired test: 10,000 bootstrap resamples of the per-query differences (percentile 95% CI and two-sided p)
and a paired t-test; p = max(bootstrap, t-test), as in Step 6 / 6b. Position test (randomisation test,
valid because llm_score.py shuffles the four passages with seed = qid): statistic = sum over the four
positions of (mean score at the position - overall mean)^2; null = each query's four scores placed in a
random order; 10,000 draws; p = (1 + #{T* >= T}) / (1 + draws).

Outputs in --out: sig_tests.csv/.md, position_tests.csv/.md, check_means.csv, means.csv, summary.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_sweep import load_qrels, query_metrics  # noqa: E402

SRC = ["OS", "OD", "ES", "ED"]                      # order of the weight vector and of query_metrics
COLS = ["original_sparse", "original_dense", "expanded_sparse", "expanded_dense"]
MODES = ["paper", "paper_fixed", "code_shuffle", "code"]
SHUFFLED = ("paper", "code_shuffle")
METHOD = {"paper": "qudar_llm", "code": "qudar_llm_codeprompt", "paper_fixed": "qudar_llm_paper_fixed",
          "code_shuffle": "qudar_llm_code_shuffle"}
MODE_LABEL = {"paper": "paper prompt + shuffled", "paper_fixed": "paper prompt + fixed",
              "code_shuffle": "code prompt + shuffled", "code": "code prompt + fixed"}
FULL = ["fiqa_test", "scidocs_test", "climate-fever_test", "hotpotqa_test"]
ABLATION = ["fiqa_test", "scidocs_test", "climate-fever_test", "hotpotqa_test_first1500"]
LABEL = {"fiqa_test": "FiQA", "scidocs_test": "SciDocs", "climate-fever_test": "Climate-FEVER",
         "hotpotqa_test": "HotpotQA", "hotpotqa_test_first1500": "HotpotQA (first 1500)"}
METRICS = [(0, "Recall@10"), (1, "nDCG@10")]
EFFECTS = [("paper_fixed", "paper", "order | paper prompt: fixed - shuffled"),
           ("code", "code_shuffle", "order | code prompt: fixed - shuffled"),
           ("code_shuffle", "paper", "prompt | shuffled: code - paper"),
           ("code", "paper_fixed", "prompt | fixed: code - paper")]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def softmax(x, tau):
    z = (np.asarray(x, float) - np.max(x)) / tau
    e = np.exp(z)
    return e / e.sum()


def llm_weights(scores, tau=3.0):
    s = np.clip(np.asarray(scores, float), 0, None)
    return np.full(4, 0.25) if np.all(s == 0) else softmax(s, tau)


def prompt_order(mode: str, qid: str) -> list[str]:
    """Sources in the order the judge saw them (llm_score.build_request)."""
    if mode in SHUFFLED:
        order = ["OS", "OD", "ES", "ED"]
        random.Random(qid).shuffle(order)
        return order
    return ["ES", "OS", "ED", "OD"]


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def paired_tests(a, b, n_boot, seed=0, chunk=1000):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    boots = np.concatenate([d[rng.integers(0, len(d), size=(min(chunk, n_boot - s), len(d)))].mean(1)
                            for s in range(0, n_boot, chunk)])
    p_boot = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    p_t = float(stats.ttest_rel(a, b).pvalue) if np.any(d != 0) else 1.0
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(p_boot), p_t


def position_test(S: np.ndarray, X: np.ndarray, n_perm: int, seed=0, chunk=500):
    """S: Q x 4 scores (any fixed source order); X: Q x 4 the same scores in prompt order."""
    grand = S.mean()
    t_obs = float(((X.mean(0) - grand) ** 2).sum())
    rng = np.random.default_rng(seed)
    hits = 0
    for s in range(0, n_perm, chunk):
        m = min(chunk, n_perm - s)
        perm = np.argsort(rng.random((m, S.shape[0], 4)), axis=2)
        Xp = np.take_along_axis(np.broadcast_to(S, (m,) + S.shape), perm, axis=2)
        hits += int((((Xp.mean(1) - grand) ** 2).sum(1) >= t_obs - 1e-12).sum())
    return X.mean(0), t_obs, (1 + hits) / (1 + n_perm)


def parse_judges(specs):
    judges = {}
    for spec in specs:
        name, _, dirs = spec.partition("=")
        if not dirs:
            raise SystemExit(f"--judge needs NAME=DIR[,DIR...], got {spec!r}")
        judges[name] = [Path(d) for d in dirs.split(",") if d]
        for d in judges[name]:
            if not d.is_dir():
                raise SystemExit(f"judge {name}: folder not found: {d}")
    return judges


def find_one(dirs, filename: str):
    hits = [p for d in dirs for p in sorted(d.rglob(filename))]
    if len(hits) > 1:
        sizes = {p.stat().st_size for p in hits}
        if len(sizes) > 1:
            print(f"  ! {filename}: {len(hits)} different copies, using {hits[0]}", flush=True)
    return hits[0] if hits else None


def load_scores(path: Path) -> dict:
    df = pd.read_csv(path, dtype={"qid": str})
    return {r[0]: np.array(r[1:], float) for r in df[["qid"] + COLS].itertuples(index=False)}


def load_reported(dirs) -> dict:
    rep = {}
    for d in dirs:
        for f in sorted(d.rglob("llm_eval.csv")):
            df = pd.read_csv(f)
            for name, method, rec, ndcg in df[["name", "method", "Recall@10", "nDCG@10"]].itertuples(index=False):
                rep[(name, method)] = (float(rec), float(ndcg))  # the last row wins (re-runs append)
    return rep


def to_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="folder holding <dataset>_test/retrieval_result.json")
    p.add_argument("--data-root", required=True, help="BEIR data: <dataset>/qrels/test.tsv")
    p.add_argument("--judge", action="append", required=True,
                   help="NAME=DIR[,DIR...] holding that judge's llm_scores_<mode>_<set>.csv (and llm_eval.csv); "
                        "give exactly two, the second is compared against the first in F2")
    p.add_argument("--out", required=True)
    p.add_argument("--full", nargs="+", default=FULL)
    p.add_argument("--ablation", nargs="+", default=ABLATION)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=10000)
    args = p.parse_args()
    judges = parse_judges(args.judge)
    if len(judges) != 2:
        raise SystemExit("give exactly two --judge")
    jnames = list(judges)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reported = {j: load_reported(d) for j, d in judges.items()}

    sets = list(dict.fromkeys(args.full + args.ablation))
    by_run = {}
    for s in sets:
        by_run.setdefault(s.split("_first")[0], []).append(s)

    perq, means, checks, pos_rows, missing = {}, [], [], [], []
    for run, run_sets in by_run.items():
        dataset, split = run.rsplit("_", 1)
        log(f"{run}: loading retrieval_result.json")
        rr = json.loads((Path(args.runs) / run / "retrieval_result.json").read_text())
        qrels = load_qrels(Path(args.data_root), dataset, split)
        all_qids = list(rr)
        ev_file = Path(args.runs) / run / "eval_results.csv"
        ref_equal = None
        if ev_file.exists():
            ref = pd.read_csv(ev_file).set_index("method")
            if "qudar_simple_equal" in ref.index:
                ref_equal = (float(ref.loc["qudar_simple_equal", "Recall@10"]), float(ref.loc["qudar_simple_equal", "nDCG@10"]))
        for s in run_sets:
            n = int(s.split("_first")[1]) if "_first" in s else len(all_qids)
            qids = all_qids[:n]
            need = MODES if s in args.ablation else []
            need = list(dict.fromkeys((["paper"] if s in args.full else []) + need))
            methods = {"Equal": None}
            for j, dirs in judges.items():
                for mode in need:
                    f = find_one(dirs, f"llm_scores_{mode}_{s}.csv")
                    if f is None:
                        missing.append(f"{j} {mode} {s}")
                        continue
                    sc = load_scores(f)
                    lacking = sum(q not in sc for q in qids)
                    if lacking:
                        print(f"  ! {j} {mode} {s}: {lacking} queries without scores -> equal weights", flush=True)
                    methods[f"{j}|{mode}"] = sc
                    if mode in SHUFFLED:
                        have = [q for q in qids if q in sc]
                        S = np.stack([sc[q] for q in have])
                        X = np.stack([[sc[q][SRC.index(src)] for src in prompt_order(mode, q)] for q in have])
                        pos_rows.append({"judge": j, "set": s, "mode": mode, "S": S, "X": X})
            names = list(methods)
            vals = np.zeros((len(names), len(qids), 3))
            for i, q in enumerate(qids):
                Wq = np.stack([np.full(4, 0.25) if sc is None or q not in sc else llm_weights(sc[q])
                               for sc in methods.values()])
                vals[:, i] = query_metrics(rr[q], qrels.get(q, {}), Wq)
                if i and i % 2000 == 0:
                    log(f"  {s}: {i}/{len(qids)} queries")
            perq[s] = {m: vals[k] for k, m in enumerate(names)}
            for m in names:
                mean = perq[s][m].mean(0)
                means.append({"set": LABEL.get(s, s), "method": m, "queries": len(qids),
                              "Recall@10": round(float(mean[0]), 4), "nDCG@10": round(float(mean[1]), 4)})
                if m == "Equal":
                    rep = ref_equal if s == run else None
                    src = "eval_results.csv"
                else:
                    j, mode = m.split("|")
                    rep = reported[j].get((s, METHOD[mode]))
                    src = f"llm_eval.csv ({j})"
                if rep is not None:
                    checks.append({"set": s, "method": m, "ours Recall@10": round(float(mean[0]), 4), "reported": rep[0],
                                   "ours nDCG@10": round(float(mean[1]), 4), "reported nDCG": rep[1], "source": src,
                                   "max |diff|": round(max(abs(mean[0] - rep[0]), abs(mean[1] - rep[1])), 4)})
            log(f"{s}: {len(qids)} queries, {len(names) - 1} LLM runs")
        del rr

    # ---------- paired tests ----------
    tests = []

    def add(family, judge, s, comparison, a_key, b_key):
        if s not in perq or a_key not in perq[s] or b_key not in perq[s]:
            missing.append(f"test skipped: {family} {judge} {s} {comparison}")
            return
        A, B = perq[s][a_key], perq[s][b_key]
        for mi, metric in METRICS:
            diff, lo, hi, pb, pt = paired_tests(A[:, mi], B[:, mi], args.n_boot)
            tests.append({"family": family, "judge": judge, "set": LABEL.get(s, s), "comparison": comparison,
                          "metric": metric, "queries": len(A), "mean A": round(float(A[:, mi].mean()), 4),
                          "mean B": round(float(B[:, mi].mean()), 4), "mean(A-B)": round(diff, 4),
                          "95% CI": f"[{lo:.4f}, {hi:.4f}]", "p bootstrap": pb, "p t-test": pt, "p": max(pb, pt)})

    for j in jnames:
        for s in args.full:
            add("F1 llm_vs_equal", j, s, "QuDAR-llm (paper prompt + shuffled) - Equal", f"{j}|paper", "Equal")
        for s in args.ablation:
            add("F1 llm_vs_equal", j, s, "QuDAR-llm (code prompt + shuffled) - Equal", f"{j}|code_shuffle", "Equal")
    for s in args.full:
        add("F2 judge", f"{jnames[1]} - {jnames[0]}", s, f"QuDAR-llm (paper prompt + shuffled): {jnames[1]} - {jnames[0]}",
            f"{jnames[1]}|paper", f"{jnames[0]}|paper")
    for j in jnames:
        for s in args.ablation:
            for a, b, label in EFFECTS:
                add("F3 ablation", j, s, label, f"{j}|{a}", f"{j}|{b}")
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p Holm"] = np.nan
        for _, idx in tests.groupby(["family", "judge"]).groups.items():
            tests.loc[idx, "p Holm"] = holm(tests.loc[idx, "p"].values)
        tests["sig (Holm p<0.05)"] = np.where(tests["p Holm"] < 0.05, "*", "")
        for c in ("p bootstrap", "p t-test", "p", "p Holm"):
            tests[c] = tests[c].round(4)

    # ---------- position randomisation tests ----------
    positions = []
    for r in pos_rows:
        s, mode = r["set"], r["mode"]
        if not ((mode == "paper" and s in args.full) or (mode == "code_shuffle" and s in args.ablation)):
            continue
        pm, t_obs, pval = position_test(r["S"], r["X"], args.n_perm)
        positions.append({"family": "F4 position", "judge": r["judge"], "set": LABEL.get(s, s), "mode": MODE_LABEL[mode],
                          "queries": len(r["S"]), **{f"pos {i + 1}": round(float(v), 3) for i, v in enumerate(pm)},
                          "highest": int(np.argmax(pm)) + 1, "lowest": int(np.argmin(pm)) + 1,
                          "T": round(t_obs, 5), "p": pval})
    positions = pd.DataFrame(positions)
    if len(positions):
        positions["p Holm"] = np.nan
        for _, idx in positions.groupby("judge").groups.items():
            positions.loc[idx, "p Holm"] = holm(positions.loc[idx, "p"].values)
        positions["sig (Holm p<0.05)"] = np.where(positions["p Holm"] < 0.05, "*", "")
        positions["p"] = positions["p"].round(5)
        positions["p Holm"] = positions["p Holm"].round(5)

    # ---------- save ----------
    means, checks = pd.DataFrame(means), pd.DataFrame(checks)
    means.to_csv(out / "means.csv", index=False)
    checks.to_csv(out / "check_means.csv", index=False)
    tests.to_csv(out / "sig_tests.csv", index=False)
    (out / "sig_tests.md").write_text(to_md(tests))
    positions.to_csv(out / "position_tests.csv", index=False)
    (out / "position_tests.md").write_text(to_md(positions))
    summary = {"n_boot": args.n_boot, "n_perm": args.n_perm, "missing": missing,
               "max_check_diff": float(checks["max |diff|"].max()) if len(checks) else None}
    if len(tests):
        summary["families"] = {
            f"{f} | {j}": {"tests": int(len(g)), "significant": int((g["sig (Holm p<0.05)"] == "*").sum()),
                          "significant, A > B": int(((g["sig (Holm p<0.05)"] == "*") & (g["mean(A-B)"] > 0)).sum()),
                          "significant, A < B": int(((g["sig (Holm p<0.05)"] == "*") & (g["mean(A-B)"] < 0)).sum())}
            for (f, j), g in tests.groupby(["family", "judge"])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---------- print ----------
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    if len(checks):
        print("\nCheck: recomputed means vs reported (max |diff| should be <= 0.0001, rounding)")
        print(checks[["set", "method", "ours Recall@10", "reported", "max |diff|"]].to_string(index=False))
    if len(tests):
        print("\nPaired tests (p = max(bootstrap, t-test); Holm inside each family x judge)")
        print(tests[["family", "judge", "set", "comparison", "metric", "mean(A-B)", "95% CI", "p", "p Holm",
                     "sig (Holm p<0.05)"]].to_string(index=False))
    if len(positions):
        print("\nPosition randomisation tests (Holm per judge)")
        print(positions[["judge", "set", "mode", "queries", "pos 1", "pos 2", "pos 3", "pos 4", "T", "p", "p Holm",
                         "sig (Holm p<0.05)"]].to_string(index=False))
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"saved to {out}")


if __name__ == "__main__":
    main()
