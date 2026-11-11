"""Step 6: learned query-wise weighting, position-debiased QuDAR-llm, LLM ablation table,
significance tests (6.2-6.6). CPU only, no API calls.

Everything is read recursively from --input (Kaggle: /kaggle/input):
  <name>/retrieval_result.json   (qudar-runs)        <name>/grid_metrics.npz (qudar-grid)
  llm_scores_<mode>_<name>.csv   (qudar-llm)         llm_<mode>_<name>.jsonl (LLM caches)
  <dataset>/queries.jsonl, <dataset>/qrels/<split>.tsv (qudar-beir-data)
  expanded_queries_<dataset>_<split>_expansion.json   (qudar-expansions)

Methods compared on every test set
  rule, no LLM : QuDAR-simple (Equal), QuDAR-confidence, Static-train (best fixed weights on the train sets)
  rule, LLM    : QuDAR-llm (paper prompt, shuffled), QuDAR-llm-debiased (position bias removed, 6.4)
  learned      : Model A (retrieval + query features), Model B (A + 4 LLM scores + their positions)
                 main model = LightGBM that predicts the (centred) nDCG@10 of each grid point
                 (alpha, beta1, beta2 in steps of 0.2) for a query; the best predicted point is used.
                 Direct-A / Direct-B = baseline that regresses the weights themselves.
  oracle       : Static UB (best fixed weights chosen on the test set), Query-wise UB
Per-query metrics come from grid_sweep.query_metrics (checked equal to qudar.fusion + pytrec_eval).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_sweep import SOURCES, load_qrels, make_grid, query_metrics  # noqa: E402

M = ["Recall@10", "nDCG@10", "Precision@1"]
SRC = ["OS", "OD", "ES", "ED"]
RAW = ["sparse_scores", "dense_scores", "pgt_sparse_scores", "pgt_dense_scores"]
TEST = ["fiqa_test", "scidocs_test", "climate-fever_test", "hotpotqa_test"]
TRAIN = ["fiqa_train", "hotpotqa_train"]
LABEL = {"fiqa_test": "FiQA", "scidocs_test": "SciDocs", "climate-fever_test": "Climate-FEVER", "hotpotqa_test": "HotpotQA"}
QWORDS = ("what", "how", "why", "when", "where", "who", "which", "is", "can", "does", "do", "should")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find(root: Path, pattern: str, parent: str | None = None):
    hits = sorted(p for p in root.rglob(pattern) if parent is None or p.parent.name == parent)
    return hits[0] if hits else None


def softmax(x, tau):
    z = (np.asarray(x, float) - np.max(x)) / tau
    e = np.exp(z)
    return e / e.sum()


# ---------- weights of the rule-based methods (same formulas as qudar.fusion) ----------
def confidence_weights(entry, topk=10, tau=2.0):
    margins = []
    for (_, nk), rk in zip(SOURCES, RAW):
        s = np.asarray(entry.get(nk) or entry.get(rk) or [], float)[:topk]
        margins.append(0.0 if s.size == 0 else float(s[0]) if s.size == 1 else max(float(s[0] - s[1]), 0.0))
    return softmax(margins, tau)


def llm_weights(scores, tau=3.0):
    s = np.clip(np.asarray(scores, float), 0, None)
    return np.ones(4) / 4 if np.all(s == 0) else softmax(s, tau)


def paper_order(qid):
    order = ["OS", "OD", "ES", "ED"]
    random.Random(qid).shuffle(order)
    return order


def mode_order(mode, qid):
    return paper_order(qid) if mode in ("paper", "code_shuffle") else ["ES", "OS", "ED", "OD"]


# ---------- 6.2 features ----------
def query_features(entry, qtext, etext):
    f = {}
    tops = []
    for name, (dk, nk), rk in zip(SRC, SOURCES, RAW):
        raw = np.asarray(entry[rk][:10], float)
        nrm = np.asarray(entry[nk][:10], float)
        tops.append(set(entry[dk][:10]))
        r1 = raw[0] if raw.size else 0.0
        f[f"{name}_raw_top1"] = r1
        f[f"{name}_raw_margin"] = (raw[0] - raw[1]) if raw.size > 1 else 0.0
        f[f"{name}_raw_ratio2"] = (raw[1] / raw[0]) if raw.size > 1 and raw[0] != 0 else 0.0
        f[f"{name}_raw_drop10"] = ((raw[0] - raw[-1]) / abs(raw[0])) if raw.size > 1 and raw[0] != 0 else 0.0
        f[f"{name}_raw_std10"] = float(raw.std()) if raw.size else 0.0
        f[f"{name}_norm_2"] = nrm[1] if nrm.size > 1 else 0.0
        f[f"{name}_norm_10"] = nrm[-1] if nrm.size else 0.0
        f[f"{name}_norm_mean10"] = float(nrm.mean()) if nrm.size else 0.0
    top1 = [entry[dk][0] if entry[dk] else None for dk, _ in SOURCES]
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = tops[i], tops[j]
            f[f"jacc10_{SRC[i]}_{SRC[j]}"] = len(a & b) / max(len(a | b), 1)
            f[f"same_top1_{SRC[i]}_{SRC[j]}"] = float(top1[i] is not None and top1[i] == top1[j])
    f["union10_frac"] = len(set().union(*tops)) / 40.0
    for i, name in enumerate(SRC):
        f[f"{name}_top1_in_others"] = sum(top1[i] in tops[j] for j in range(4) if j != i) / 3.0
    toks = qtext.split()
    f["q_len"] = len(toks)
    f["e_len"] = len(etext.split())
    f["e_over_q"] = f["e_len"] / max(f["q_len"], 1)
    f["q_digit_frac"] = sum(any(c.isdigit() for c in t) for t in toks) / max(len(toks), 1)
    f["q_cap_frac"] = sum(t[:1].isupper() for t in toks[1:]) / max(len(toks) - 1, 1)
    f["q_is_question"] = float(qtext.strip().endswith("?") or (toks[0].lower() in QWORDS if toks else False))
    return f


def llm_features(scores: dict | None, qid: str):
    f = {}
    order = paper_order(qid)
    for name in SRC:
        f[f"llm_{name}"] = float(scores[name]) if scores else np.nan
        f[f"llm_pos_{name}"] = float(order.index(name) + 1)
    return f


# ---------- data loading ----------
def load_queries(root: Path, dataset: str, split: str):
    qfile = find(root, "queries.jsonl", dataset)
    text = {}
    with qfile.open(encoding="utf8") as h:
        for line in h:
            r = json.loads(line)
            text[str(r["_id"])] = r.get("text", "")
    return text


def load_llm_csv(root: Path, mode: str, name: str):
    f = find(root, f"llm_scores_{mode}_{name}.csv")
    if f is None:
        return None
    df = pd.read_csv(f, dtype={"qid": str})
    return {r.qid: {"OS": r.original_sparse, "OD": r.original_dense, "ES": r.expanded_sparse, "ED": r.expanded_dense}
            for r in df.itertuples()}


def load_run(root: Path, name: str, debias: dict | None):
    """One pass over a run: features, grid metrics, per-query metrics of the rule-based methods."""
    dataset, split = name.rsplit("_", 1)
    log(f"{name}: loading")
    rr = json.loads(find(root, "retrieval_result.json", name).read_text())
    z = np.load(find(root, "grid_metrics.npz", name))
    gq = [str(q) for q in z["qids"]]
    gpos = {q: i for i, q in enumerate(gq)}
    gmet = z["metrics"].astype(np.float32)
    qrels = load_qrels(find(root, "queries.jsonl", dataset).parent.parent, dataset, split)
    qtext = load_queries(root, dataset, split)
    exp = json.loads(find(root, f"expanded_queries_{dataset}_{split}_expansion.json").read_text())
    llm = load_llm_csv(root, "paper", name)
    qids = [q for q in rr if q in gpos]
    feats, rule_metrics, rule_weights = [], [], {}
    for q in qids:
        e = rr[q]
        f = query_features(e, qtext.get(q, ""), exp.get(q, ""))
        f.update(llm_features(llm.get(q) if llm else None, q))
        feats.append(f)
        W = {"QuDAR-simple (Equal)": np.full(4, .25), "QuDAR-confidence": confidence_weights(e)}
        if llm and q in llm:
            s = [llm[q][k] for k in SRC]
            W["QuDAR-llm"] = llm_weights(s)
            if debias is not None:
                order = paper_order(q)
                W["QuDAR-llm-debiased"] = llm_weights([llm[q][k] - debias[order.index(k) + 1] for k in SRC])
        rule_weights[q] = W
        rule_metrics.append(query_metrics(e, qrels.get(q, {}), np.stack(list(W.values()))))
    names = list(rule_weights[qids[0]].keys())
    rm = {n: np.stack([m[i] for m in rule_metrics]) for i, n in enumerate(names)}
    del rr
    X = pd.DataFrame(feats, index=qids)
    G = gmet[[gpos[q] for q in qids]]
    log(f"{name}: {len(qids)} queries, {X.shape[1]} features")
    return {"name": name, "qids": qids, "X": X, "G": G, "rule": rm, "grid": z["grid"], "weights": z["weights"]}


def position_bias(root: Path, names):
    """Mean LLM score per prompt position minus the overall mean (paper mode caches)."""
    by_pos = {i: [] for i in range(1, 5)}
    for name in names:
        cache = find(root, f"llm_paper_{name}.jsonl")
        if cache is None:
            continue
        for line in cache.read_text().splitlines():
            r = json.loads(line)
            if r["scores"]:
                for i, s in enumerate(paper_order(r["qid"]), 1):
                    by_pos[i].append(r["scores"][s])
    allv = [v for vs in by_pos.values() for v in vs]
    if not allv:
        return None
    mu = np.mean(allv)
    return {i: float(np.mean(v) - mu) for i, v in by_pos.items()}


# ---------- 6.3 learned weighting ----------
def surface_rows(X: np.ndarray, grid, weights, idx):
    """Repeat query features for each grid point in idx; add the point's (a, b1, b2) and weights."""
    nq, k = X.shape[0], len(idx)
    gp = np.hstack([grid[idx], weights[idx]]).astype(np.float32)
    rows = np.hstack([np.repeat(X, k, axis=0), np.tile(gp, (nq, 1))])
    return rows


def coarse_points(grid):
    """Grid points with every coordinate in {0, .2, .4, .6, .8, 1}: 216 of the 1331 points (training and inference)."""
    return np.flatnonzero(np.all(np.isclose(np.round(grid * 5) / 5, grid), axis=1))


def train_surface(train_runs, cols, seed=0):
    import lightgbm as lgb
    grid, weights = train_runs[0]["grid"], train_runs[0]["weights"]
    sub = coarse_points(grid)
    Xs, ys, groups, offset = [], [], [], 0
    for r in train_runs:
        nd = r["G"][:, :, 1]
        info = nd.max(1) > nd.min(1)          # queries whose score depends on the weights
        X = r["X"][cols].to_numpy(np.float32)[info]
        y = (nd[info] - nd[info].mean(1, keepdims=True))[:, sub]   # centred: which weights are better for THIS query
        Xs.append(surface_rows(X, grid, weights, sub))
        ys.append(y.ravel())
        groups.append(np.repeat(np.arange(info.sum()), len(sub)) + offset)
        offset += int(info.sum())
    X, y, g = np.vstack(Xs), np.concatenate(ys), np.concatenate(groups)
    rng = np.random.default_rng(seed)
    val_q = rng.random(g.max() + 1) < 0.1
    vm = val_q[g]
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.1, num_leaves=31, min_child_samples=200,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=seed, verbose=-1)
    model.fit(X[~vm], y[~vm], eval_set=[(X[vm], y[vm])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    log(f"  surface model: {len(y)} rows, best iteration {model.best_iteration_}")
    return model


def predict_surface(model, run, cols, chunk=400):
    grid, weights = run["grid"], run["weights"]
    idx = coarse_points(grid)
    X = run["X"][cols].to_numpy(np.float32)
    best = np.empty(len(X), dtype=int)
    for s in range(0, len(X), chunk):
        rows = surface_rows(X[s:s + chunk], grid, weights, idx)
        pred = model.predict(rows, num_iteration=model.best_iteration_).reshape(-1, len(idx))
        best[s:s + chunk] = idx[pred.argmax(1)]
    return best


def train_direct(train_runs, cols, T=0.02, seed=0):
    import lightgbm as lgb
    grid = train_runs[0]["grid"]
    Xs, Ys = [], []
    for r in train_runs:
        nd = r["G"][:, :, 1].astype(np.float64)
        info = nd.max(1) > nd.min(1)
        p = np.exp((nd[info] - nd[info].max(1, keepdims=True)) / T)
        p /= p.sum(1, keepdims=True)
        Xs.append(r["X"][cols].to_numpy(np.float32)[info])
        Ys.append(p @ grid)                       # expected (alpha, beta1, beta2) near the optimum
    X, Y = np.vstack(Xs), np.vstack(Ys)
    models = [lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                                verbose=-1).fit(X, Y[:, j]) for j in range(3)]
    return models


def predict_direct(models, run, cols):
    X = run["X"][cols].to_numpy(np.float32)
    abc = np.clip(np.stack([m.predict(X) for m in models], 1), 0, 1)
    abc = np.round(abc * 10) / 10
    grid = run["grid"]
    lut = {tuple(np.round(g, 1)): i for i, g in enumerate(grid)}
    return np.array([lut[tuple(v)] for v in abc])


def static_train_point(train_runs):
    return int(np.argmax(np.concatenate([r["G"][:, :, 1] for r in train_runs]).mean(0)))


# ---------- 6.5 significance ----------
def paired_tests(a, b, n_boot=10000, seed=0):
    from scipy import stats
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    p_boot = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    p_t = float(stats.ttest_rel(a, b).pvalue) if np.any(d != 0) else 1.0
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), p_boot, p_t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--out", required=True)
    p.add_argument("--test", nargs="+", default=TEST)
    p.add_argument("--train", nargs="+", default=TRAIN)
    p.add_argument("--ablation-subset", nargs="*", default=["hotpotqa_test:1500"],
                   help="name:N -> the LLM ablation table uses the first N queries of that run")
    args = p.parse_args()
    root, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 6.4 position bias, estimated on the TRAIN sets only
    bias = position_bias(root, args.train)
    log(f"position bias from train (score at position - mean): {bias}")
    json.dump(bias, open(out / "position_bias_train.json", "w"), indent=1)

    train_runs = [load_run(root, n, bias) for n in args.train]
    test_runs = [load_run(root, n, bias) for n in args.test]
    feat_all = list(train_runs[0]["X"].columns)
    cols_A = [c for c in feat_all if not c.startswith("llm_")]
    cols_B = feat_all
    has_llm_train = all(not r["X"]["llm_OS"].isna().all() for r in train_runs)
    if not has_llm_train:
        log("WARNING: no LLM scores for the train sets -> Model B skipped")

    # ---- train: configs = train sets x feature sets ----
    train_sets = {"both": train_runs, **{r["name"].rsplit("_", 1)[0]: [r] for r in train_runs}}
    chosen = {}       # (method, config) -> {test name: grid index per query}
    importances = {}
    for cfg, runs in train_sets.items():
        for tag, cols in [("A", cols_A), ("B", cols_B)]:
            if tag == "B" and not has_llm_train:
                continue
            log(f"train surface model {tag} on {cfg}")
            model = train_surface(runs, cols)
            if cfg == "both":
                imp = pd.Series(model.booster_.feature_importance("gain"),
                                index=cols + ["alpha", "beta1", "beta2", "w_OS", "w_OD", "w_ES", "w_ED"])
                importances[tag] = imp.sort_values(ascending=False)
                model.booster_.save_model(str(out / f"model_{tag}_{cfg}.txt"))
                json.dump(cols, open(out / f"model_{tag}_features.json", "w"))
            chosen[(f"Model {tag}", cfg)] = {r["name"]: predict_surface(model, r, cols) for r in test_runs}
            if cfg == "both":
                log(f"train direct baseline {tag}")
                dm = train_direct(runs, cols)
                chosen[(f"Direct-{tag}", cfg)] = {r["name"]: predict_direct(dm, r, cols) for r in test_runs}
        st = static_train_point(runs)
        chosen[("Static-train", cfg)] = {r["name"]: np.full(len(r["qids"]), st) for r in test_runs}

    # ---- per-query metric matrix per test set ----
    perq = {}
    for r in test_runs:
        n = r["name"]
        d = dict(r["rule"])
        for (meth, cfg), sel in chosen.items():
            d[f"{meth} [{cfg}]"] = r["G"][np.arange(len(r["qids"])), sel[n]]
        nd = r["G"][:, :, 1]
        d["Static UB (oracle)"] = r["G"][:, int(np.argmax(nd.mean(0)))]
        best = np.argmax(r["G"][:, :, 1] * 1e3 + r["G"][:, :, 0], axis=1)
        d["Query-wise UB (oracle)"] = r["G"][np.arange(len(best)), best]
        perq[n] = d
        # chosen weights of the learned models, for the demo / error analysis
        wdf = pd.DataFrame({"qid": r["qids"]})
        for tag in ("A", "B"):
            if (f"Model {tag}", "both") in chosen:
                g = r["grid"][chosen[(f"Model {tag}", "both")][n]]
                wdf[[f"{tag}_alpha", f"{tag}_beta1", f"{tag}_beta2"]] = g
        wdf.to_csv(out / f"chosen_weights_{n}.csv", index=False)

    # ---- tables ----
    main_rows = []
    order = ["QuDAR-simple (Equal)", "QuDAR-confidence", "Static-train [both]", "QuDAR-llm", "QuDAR-llm-debiased",
             "Direct-A [both]", "Model A [both]", "Direct-B [both]", "Model B [both]",
             "Static UB (oracle)", "Query-wise UB (oracle)"]
    for n, d in perq.items():
        conf, qub = d["QuDAR-confidence"].mean(0), d["Query-wise UB (oracle)"].mean(0)
        for meth in order:
            if meth not in d:
                continue
            m = d[meth].mean(0)
            gap = 100 * (m[0] - conf[0]) / max(qub[0] - conf[0], 1e-9)
            main_rows.append({"dataset": LABEL.get(n, n), "method": meth, **dict(zip(M, np.round(m, 4))),
                              "gap closed % (Recall)": round(gap, 1)})
    main = pd.DataFrame(main_rows)
    main.to_csv(out / "table6_main.csv", index=False)
    (out / "table6_main.md").write_text(main.to_markdown(index=False))
    print("\n" + main.to_string(index=False), flush=True)

    # 2x2: rule / learned x no LLM / LLM (Recall@10)
    two = []
    for n, d in perq.items():
        g = lambda k: round(float(d[k].mean(0)[0]), 4) if k in d else np.nan  # noqa: E731
        two += [{"dataset": LABEL.get(n, n), "": "Rule (paper)", "no LLM": g("QuDAR-confidence"), "LLM": g("QuDAR-llm")},
                {"dataset": LABEL.get(n, n), "": "Rule + debias (ours)", "no LLM": np.nan, "LLM": g("QuDAR-llm-debiased")},
                {"dataset": LABEL.get(n, n), "": "Learned (ours)", "no LLM": g("Model A [both]"), "LLM": g("Model B [both]")}]
    two = pd.DataFrame(two)
    two.to_csv(out / "table6_2x2.csv", index=False)
    (out / "table6_2x2.md").write_text(two.to_markdown(index=False))

    # training-set ablation (Recall@10)
    tr = []
    for (meth, cfg) in chosen:
        if meth.startswith("Model") or meth == "Static-train":
            tr.append({"method": meth, "train on": cfg,
                       **{LABEL.get(n, n): round(float(perq[n][f"{meth} [{cfg}]"].mean(0)[0]), 4) for n in perq}})
    tr = pd.DataFrame(tr)
    tr.to_csv(out / "table6_train_config.csv", index=False)
    (out / "table6_train_config.md").write_text(tr.to_markdown(index=False))
    print("\n" + tr.to_string(index=False), flush=True)

    # 6.1 LLM ablation: 2 prompts x 2 orders (+ debiased) per test set
    subset = {s.split(":")[0]: int(s.split(":")[1]) for s in args.ablation_subset}
    ab, ab_sig = [], []
    for r in test_runs:
        n = r["name"]
        nq = subset.get(n)
        ev_name = f"{n}_first{nq}" if nq else n
        qids = r["qids"][:nq] if nq else r["qids"]
        rr = None
        row = {"dataset": LABEL.get(n, n) + (f" (first {nq})" if nq else "")}
        pq = {}
        for mode, label in [("paper", "paper prompt + shuffled"), ("paper_fixed", "paper prompt + fixed"),
                            ("code_shuffle", "code prompt + shuffled"), ("code", "code prompt + fixed")]:
            sc = load_llm_csv(root, mode, ev_name) or load_llm_csv(root, mode, n)
            if sc is None or not all(q in sc for q in qids):
                row[label] = np.nan
                continue
            if rr is None:
                rr = json.loads(find(root, "retrieval_result.json", n).read_text())
                dataset, split = n.rsplit("_", 1)
                qrels = load_qrels(find(root, "queries.jsonl", dataset).parent.parent, dataset, split)
            vals = np.stack([query_metrics(rr[q], qrels.get(q, {}), llm_weights([sc[q][k] for k in SRC])[None, :])[0]
                             for q in qids])
            row[label] = round(float(vals.mean(0)[0]), 4)
            pq[mode] = vals
            if mode == "paper" and bias is not None:
                vals = np.stack([query_metrics(rr[q], qrels.get(q, {}),
                                               llm_weights([sc[q][k] - bias[paper_order(q).index(k) + 1] for k in SRC])[None, :])[0]
                                 for q in qids])
                row["paper prompt + shuffled + debiased (ours)"] = round(float(vals.mean(0)[0]), 4)
                pq["paper_debiased"] = vals
        for a, b, effect in [("paper_fixed", "paper", "order (paper prompt): fixed - shuffled"),
                             ("code", "code_shuffle", "order (code prompt): fixed - shuffled"),
                             ("code_shuffle", "paper", "prompt (shuffled): code - paper"),
                             ("code", "paper_fixed", "prompt (fixed): code - paper"),
                             ("paper_debiased", "paper", "debias (paper prompt, shuffled)")]:
            if a in pq and b in pq:
                for mi, metric in [(0, "Recall@10"), (1, "nDCG@10")]:
                    diff, lo, hi, pb, pt = paired_tests(pq[a][:, mi], pq[b][:, mi])
                    ab_sig.append({"dataset": row["dataset"], "effect": effect, "metric": metric,
                                   "mean diff": round(diff, 4), "95% CI": f"[{lo:.4f}, {hi:.4f}]",
                                   "p bootstrap": round(pb, 4), "p t-test": round(pt, 4),
                                   "sig (p<0.05)": "*" if max(pb, pt) < 0.05 else ""})
        del rr
        ab.append(row)
    ab = pd.DataFrame(ab)
    ab.to_csv(out / "table6_llm_ablation.csv", index=False)
    (out / "table6_llm_ablation.md").write_text(ab.to_markdown(index=False))
    print("\n" + ab.to_string(index=False), flush=True)
    if ab_sig:
        abs_df = pd.DataFrame(ab_sig)
        abs_df.to_csv(out / "table6_llm_ablation_significance.csv", index=False)
        (out / "table6_llm_ablation_significance.md").write_text(abs_df.to_markdown(index=False))
        print("\n" + abs_df[abs_df.metric == "Recall@10"].to_string(index=False), flush=True)

    # 6.5 significance (nDCG@10 and Recall@10, paired over queries)
    pairs = [("Model A [both]", "QuDAR-confidence"), ("Model A [both]", "QuDAR-llm"), ("Model B [both]", "QuDAR-llm"),
             ("Model B [both]", "Model A [both]"), ("Model A [both]", "Static-train [both]"),
             ("QuDAR-llm-debiased", "QuDAR-llm"), ("QuDAR-llm", "QuDAR-confidence"),
             ("QuDAR-confidence", "QuDAR-simple (Equal)")]
    sig = []
    for n, d in perq.items():
        for a, b in pairs:
            if a not in d or b not in d:
                continue
            for mi, metric in [(1, "nDCG@10"), (0, "Recall@10")]:
                diff, lo, hi, pb, pt = paired_tests(d[a][:, mi], d[b][:, mi])
                sig.append({"dataset": LABEL.get(n, n), "A": a, "B": b, "metric": metric, "mean(A-B)": round(diff, 4),
                            "95% CI": f"[{lo:.4f}, {hi:.4f}]", "p bootstrap": round(pb, 4), "p t-test": round(pt, 4),
                            "sig (p<0.05)": "*" if max(pb, pt) < 0.05 else ""})
    sig = pd.DataFrame(sig)
    sig.to_csv(out / "table6_significance.csv", index=False)
    (out / "table6_significance.md").write_text(sig.to_markdown(index=False))
    print("\n" + sig[sig.metric == "nDCG@10"].to_string(index=False), flush=True)

    # sanity: recomputed rule methods vs Step 2 / Step 4 numbers
    chk = []
    for n, d in perq.items():
        ev = find(root, "eval_results.csv", n)
        if ev is not None:
            ref = pd.read_csv(ev).set_index("method")
            for meth, key in [("QuDAR-simple (Equal)", "qudar_simple_equal"), ("QuDAR-confidence", "qudar_confidence")]:
                chk.append({"dataset": n, "method": meth, "ours": round(float(d[meth].mean(0)[0]), 4),
                            "Step 2 file": float(ref.loc[key, "Recall@10"])})
    if chk:
        print("\nCheck vs eval_results.csv (Recall@10):\n" + pd.DataFrame(chk).to_string(index=False), flush=True)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for tag, imp in importances.items():
        top = imp.head(20)[::-1]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(top.index, top.values / top.values.sum())
        ax.set_xlabel("share of total gain (top 20)")
        ax.set_title(f"Model {tag}: feature importance")
        fig.tight_layout()
        fig.savefig(out / f"fig6_feature_importance_{tag}.png", dpi=200)
        plt.close(fig)
    rows = []
    for name in args.test + args.train:
        cache = find(root, f"llm_paper_{name}.jsonl")
        if cache is None:
            continue
        pos = {i: [] for i in range(1, 5)}
        for line in cache.read_text().splitlines():
            rj = json.loads(line)
            if rj["scores"]:
                for i, s in enumerate(paper_order(rj["qid"]), 1):
                    pos[i].append(rj["scores"][s])
        rows.append((name, [np.mean(pos[i]) for i in range(1, 5)]))
    if rows:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        w = 0.8 / len(rows)
        for j, (name, vals) in enumerate(rows):
            ax.bar(np.arange(1, 5) + (j - len(rows) / 2 + .5) * w, vals, w, label=name)
        ax.set_xticks(range(1, 5), ["position 1", "2", "3", "4"])
        ax.set_ylabel("mean LLM score (0-5)")
        ax.set_title("gpt-4o-mini scores by position in the prompt (passages shuffled)")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(axis="y", alpha=.3)
        fig.tight_layout()
        fig.savefig(out / "fig6_position_bias.png", dpi=200)
        plt.close(fig)
    log(f"Saved Step 6 outputs to {out}")


if __name__ == "__main__":
    main()
