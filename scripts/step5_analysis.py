"""Step 3-5 wrap-up: full result tables vs the paper + Figures 2-5 (Contriever, Recall@10).

Reads (searched recursively under --input): <name>/eval_results.csv (Step 2), <name>/grid_metrics.npz
(grid_sweep.py) and every llm_eval.csv (llm_score.py). Writes to --out:
  table_main.csv / .md     Table 1-2 layout, ours vs paper (Recall@10, nDCG@10, P@1)
  table_upper_bounds.csv   Table 8 layout: static / query-wise upper bounds per axis
  fig2_3_static_sweeps.png Recall@10 vs alpha (OS-OD) and vs beta (OS-ES, OD-ED)
  fig4a_optimal_hist.png   per-query optimal alpha / beta (one-axis sweeps; ties split evenly)
  fig4b_upper_bounds.png   single / one-axis static & dynamic / dual static & dynamic
  fig5_optimal_heatmap.png per-query optimal (alpha, share of expanded) on the dual grid (ties split evenly)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

M = ["Recall@10", "nDCG@10", "Precision@1"]
NAMES = ["fiqa_test", "scidocs_test", "climate-fever_test", "hotpotqa_test"]
LABEL = {"fiqa_test": "FiQA", "scidocs_test": "SciDocs", "climate-fever_test": "Climate-FEVER", "hotpotqa_test": "HotpotQA"}
# Paper Tables 1-2 (Recall@10, nDCG@10, P@1)
PAPER = {
    "fiqa_test": {"OS": (.3135, .2500, .2299), "OD": (.1649, .1299, .1173), "ES": (.2085, .1496, .1127), "ED": (.1835, .1363, .1049),
                  "Static Avg (type)": (.2787, .2232, .2044), "Static UB (type)": (.3361, .2718, .2500), "DAT": (.2842, .2332, .2284),
                  "Static Avg (S)": (.2979, .2303, .1939), "Static UB (S)": (.3404, .2726, .2423), "Static Avg (D)": (.1647, .1262, .1063),
                  "Static UB (D)": (.1810, .1406, .1265), "Static Avg (dual)": (.2779, .2143, .1806), "Static UB (dual)": (.3715, .2915, .2562),
                  "QuDAR-simple (RRF)": (.3219, .2461, .1944), "QuDAR-simple (Equal)": (.3402, .2654, .2284),
                  "QuDAR-confidence": (.3438, .2684, .2330), "QuDAR-llm": (.3639, .2837, .2454), "Query-wise UB (dual)": (.5011, .4385, .4522)},
    "scidocs_test": {"OS": (.1655, .1578, .1810), "OD": (.1112, .1031, .1160), "ES": (.1497, .1370, .1460), "ED": (.1487, .1390, .1570),
                     "Static Avg (type)": (.1597, .1504, .1703), "Static UB (type)": (.1776, .1683, .1970), "DAT": (.1614, .1525, .1720),
                     "Static Avg (S)": (.1682, .1583, .1759), "Static UB (S)": (.1757, .1666, .1880), "Static Avg (D)": (.1362, .1289, .1510),
                     "Static UB (D)": (.1479, .1413, .1750), "Static Avg (dual)": (.1715, .1615, .1820), "Static UB (dual)": (.1917, .1813, .2060),
                     "QuDAR-simple (RRF)": (.1893, .1757, .1950), "QuDAR-simple (Equal)": (.1906, .1792, .2000),
                     "QuDAR-confidence": (.1920, .1799, .1990), "QuDAR-llm": (.1920, .1816, .2050), "Query-wise UB (dual)": (.2745, .2808, .3990)},
    "climate-fever_test": {"OS": (.1750, .1376, .1160), "OD": (.1235, .0909, .0606), "ES": (.2598, .2053, .1720), "ED": (.1960, .1624, .1564),
                           "Static Avg (type)": (.1784, .1420, .1216), "Static UB (type)": (.2040, .1651, .1498), "DAT": (.1699, .1375, .1251),
                           "Static Avg (S)": (.2506, .2020, .1784), "Static UB (S)": (.2886, .2346, .2117), "Static Avg (D)": (.1379, .1080, .0911),
                           "Static UB (D)": (.1527, .1195, .1075), "Static Avg (dual)": (.2169, .1761, .1603), "Static UB (dual)": (.2972, .2408, .2150),
                           "QuDAR-simple (RRF)": (.2411, .1931, .1661), "QuDAR-simple (Equal)": (.2464, .2045, .1967),
                           "QuDAR-confidence": (.2473, .2050, .1967), "QuDAR-llm": (.2583, .2138, .2085), "Query-wise UB (dual)": (.3859, .3544, .4150)},
    "hotpotqa_test": {"OS": (.6039, .5741, .6663), "OD": (.4011, .3531, .3749), "ES": (.4151, .3605, .3625), "ED": (.4838, .4375, .4675),
                      "Static Avg (type)": (.5730, .5359, .6097), "Static UB (type)": (.6419, .6096, .7025), "DAT": (.5781, .5567, .6641),
                      "Static Avg (S)": (.5869, .5397, .5898), "Static UB (S)": (.6746, .6342, .7134), "Static Avg (D)": (.4943, .4439, .4743),
                      "Static UB (D)": (.5570, .5013, .5357), "Static Avg (dual)": (.6024, .5529, .6001), "Static UB (dual)": (.7110, .6694, .7463),
                      "QuDAR-simple (RRF)": (.6741, .6114, .6460), "QuDAR-simple (Equal)": (.6924, .6423, .6948),
                      "QuDAR-confidence": (.6962, .6478, .7047), "QuDAR-llm": (.6883, .6579, .7515), "Query-wise UB (dual)": (.7891, .7784, .8917)},
}
EVAL_MAP = {"OS": "os", "OD": "od", "ES": "es", "ED": "ed", "Static Avg (type)": "sha_os_od", "Static UB (type)": "shb_os_od",
            "Static Avg (S)": "sha_os_es", "Static UB (S)": "shb_os_es", "Static Avg (D)": "sha_od_ed", "Static UB (D)": "shb_od_ed",
            "QuDAR-simple (RRF)": "qudar_simple_rrf", "QuDAR-simple (Equal)": "qudar_simple_equal", "QuDAR-confidence": "qudar_confidence"}
ORDER = ["OS", "OD", "ES", "ED", "Static Avg (type)", "Static UB (type)", "DAT", "Static Avg (S)", "Static UB (S)",
         "Static Avg (D)", "Static UB (D)", "Static Avg (dual)", "Static UB (dual)", "QuDAR-simple (RRF)",
         "QuDAR-simple (Equal)", "QuDAR-confidence", "QuDAR-llm", "QuDAR-llm (code prompt)",
         "QuDAR-llm (paper prompt, fixed order)", "QuDAR-llm (code prompt, shuffled)", "Query-wise UB (dual)"]


def find(root: Path, name: str, filename: str) -> Path | None:
    hits = [p for p in root.rglob(filename) if p.parent.name == name]
    return hits[0] if hits else None


def load_grid(path: Path):
    z = np.load(path)
    return z["grid"], z["metrics"].astype(np.float32)


def axis_masks(grid):
    a, b1, b2 = grid.T
    c = np.isclose
    return {"type": c(b1, 0) & c(b2, 0), "S": c(a, 0) & c(b2, 0), "D": c(a, 1) & c(b1, 0), "dual": np.ones(len(grid), bool)}


def axis_values(grid, axis):
    a, b1, b2 = grid.T
    return {"type": a, "S": b1, "D": b2}[axis]


def optimal_sets(metrics, mask):
    """Per query: boolean mask of ALL grid points that reach the best (nDCG@10, then Recall@10).
    Queries where every point scores the same carry no preference and are dropped."""
    sub = metrics[:, mask, :].astype(np.float64)
    key = np.round(sub[:, :, 1], 6) * 1e3 + np.round(sub[:, :, 0], 6)
    best = key == key.max(axis=1, keepdims=True)
    informative = ~best.all(axis=1)
    return best[informative]


def fractional_hist(values, best_sets, bins):
    """Each query contributes weight 1, spread evenly over all its optimal values (no tie-break bias)."""
    w = best_sets / best_sets.sum(axis=1, keepdims=True)
    return np.histogram(np.repeat(values[None, :], len(w), 0).ravel(), bins=bins, weights=w.ravel())[0]


def dynamic_ub(metrics, mask):
    sub = metrics[:, mask, :]
    best = np.argmax(sub[:, :, 1] * 1e3 + sub[:, :, 0], axis=1)
    return sub[np.arange(sub.shape[0]), best].mean(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--out", required=True)
    p.add_argument("--names", nargs="+", default=NAMES)
    args = p.parse_args()
    root, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    llm = pd.concat([pd.read_csv(f) for f in root.rglob("llm_eval.csv")], ignore_index=True) \
        if list(root.rglob("llm_eval.csv")) else pd.DataFrame(columns=["name", "method", *M])
    llm = llm.drop_duplicates(["name", "method"], keep="last").set_index(["name", "method"])

    rows, ub_rows, grids = [], [], {}
    for name in args.names:
        ev = pd.read_csv(find(root, name, "eval_results.csv")).set_index("method")
        grid, metrics = load_grid(find(root, name, "grid_metrics.npz"))
        grids[name] = (grid, metrics)
        masks = axis_masks(grid)
        ours = {k: tuple(ev.loc[v, M]) for k, v in EVAL_MAP.items()}
        per_point = metrics.mean(0)
        ours["Static Avg (dual)"] = tuple(per_point.mean(0))
        ours["Static UB (dual)"] = tuple(per_point.max(0))
        ours["Query-wise UB (dual)"] = tuple(dynamic_ub(metrics, masks["dual"]))
        for method, label in [("qudar_llm", "QuDAR-llm"), ("qudar_llm_codeprompt", "QuDAR-llm (code prompt)"),
                              ("qudar_llm_paper_fixed", "QuDAR-llm (paper prompt, fixed order)"),
                              ("qudar_llm_code_shuffle", "QuDAR-llm (code prompt, shuffled)"), ("dat", "DAT")]:
            if (name, method) in llm.index:
                ours[label] = tuple(llm.loc[(name, method), M])
        for label in ORDER:
            if label not in ours:
                continue
            row = {"dataset": LABEL.get(name, name), "method": label}
            pap = PAPER.get(name, {}).get(label)
            for i, m in enumerate(M):
                row[m] = round(float(ours[label][i]), 4)
                row[f"{m} paper"] = pap[i] if pap else np.nan
            row["Recall diff %"] = round(100 * (row["Recall@10"] / pap[0] - 1), 1) if pap else np.nan
            rows.append(row)
        for axis in ("type", "S", "D", "dual"):
            st = per_point[masks[axis]].max(0)
            dy = dynamic_ub(metrics, masks[axis])
            ub_rows.append({"dataset": LABEL.get(name, name), "axis": axis,
                            **{f"Static UB {m}": round(float(v), 4) for m, v in zip(M, st)},
                            **{f"Query-wise UB {m}": round(float(v), 4) for m, v in zip(M, dy)},
                            "gain % (Recall)": round(100 * (dy[0] / st[0] - 1), 1)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "table_main.csv", index=False)
    (out / "table_main.md").write_text(table.to_markdown(index=False))
    ubt = pd.DataFrame(ub_rows)
    ubt.to_csv(out / "table_upper_bounds.csv", index=False)
    print(table[["dataset", "method", "Recall@10", "Recall@10 paper", "Recall diff %", "nDCG@10", "Precision@1"]].to_string(index=False))
    print("\n", ubt[["dataset", "axis", "Static UB Recall@10", "Query-wise UB Recall@10", "gain % (Recall)"]].to_string(index=False))

    names = list(grids)
    # Figure 2 / 3: one-axis static sweeps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for ax, axis, title, xl in [(axes[0], "type", "Retriever type: OS (0) to OD (1)", "alpha"),
                                (axes[1], "S", "Query format in BM25: OS (0) to ES (1)", "beta1"),
                                (axes[2], "D", "Query format in Contriever: OD (0) to ED (1)", "beta2")]:
        for name in names:
            grid, metrics = grids[name]
            mask = axis_masks(grid)[axis]
            x = axis_values(grid, axis)[mask]
            y = metrics.mean(0)[mask, 0]
            ax.plot(x, y, marker="o", ms=3, label=LABEL.get(name, name))
            ax.scatter([x[np.argmax(y)]], [y.max()], s=40, zorder=3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xl)
        ax.set_ylabel("Recall@10")
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_3_static_sweeps.png", dpi=200)
    plt.close(fig)

    # Figure 4a: per-query optimal weights (one-axis sweeps)
    fig, axes = plt.subplots(len(names), 2, figsize=(9, 2.3 * len(names)), squeeze=False)
    bins = np.linspace(-0.05, 1.05, 12)
    for r, name in enumerate(names):
        grid, metrics = grids[name]
        masks = axis_masks(grid)
        for c, (axis, xl) in enumerate([("type", "optimal alpha (0 = BM25, 1 = Contriever)"),
                                        ("S", "optimal beta1 (0 = original, 1 = expanded)")]):
            mask = masks[axis]
            sets = optimal_sets(metrics, mask)
            h = fractional_hist(axis_values(grid, axis)[mask], sets, bins)
            axes[r, c].bar((bins[:-1] + bins[1:]) / 2, h, width=0.095, edgecolor="white")
            axes[r, c].set_title(f"{LABEL.get(name, name)}  ({len(sets)} informative queries)", fontsize=9)
            axes[r, c].set_xlabel(xl, fontsize=8)
            axes[r, c].set_ylabel("queries (ties split)", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig4a_optimal_hist.png", dpi=200)
    plt.close(fig)

    # Figure 4b: upper bounds
    labels = ["Best single", "1-axis static UB", "1-axis query-wise UB", "Dual static UB", "Dual query-wise UB"]
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.16
    for i, name in enumerate(names):
        grid, metrics = grids[name]
        masks = axis_masks(grid)
        per_point = metrics.mean(0)
        a, b1, b2 = grid.T
        corners = [np.flatnonzero(np.isclose(a, x) & np.isclose(b1, y) & np.isclose(b2, z))[0]
                   for x, y, z in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 1)]]
        vals = [per_point[corners, 0].max(),
                max(per_point[masks[k], 0].max() for k in ("type", "S", "D")),
                max(dynamic_ub(metrics, masks[k])[0] for k in ("type", "S", "D")),
                per_point[:, 0].max(), dynamic_ub(metrics, masks["dual"])[0]]
        for j, v in enumerate(vals):
            ax.bar(i + (j - 2) * width, v, width, label=labels[j] if i == 0 else None, color=f"C{j}")
    ax.set_xticks(range(len(names)), [LABEL.get(n, n) for n in names])
    ax.set_ylabel("Recall@10")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "fig4b_upper_bounds.png", dpi=200)
    plt.close(fig)

    # Figure 5: per-query optimal (alpha, share of expanded) on the dual grid
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6), squeeze=False)
    for c, name in enumerate(names):
        grid, metrics = grids[name]
        a, b1, b2 = grid.T
        share_expanded = (1 - a) * b1 + a * b2
        sets = optimal_sets(metrics, np.ones(len(grid), bool))
        wts = sets / sets.sum(axis=1, keepdims=True)
        edges = np.linspace(-0.05, 1.05, 12)
        h, _, _ = np.histogram2d(np.tile(a, len(wts)), np.tile(share_expanded, len(wts)), bins=[edges, edges],
                                 weights=wts.ravel())
        im = axes[0, c].imshow(h.T, origin="lower", extent=[-0.05, 1.05, -0.05, 1.05], cmap="viridis", aspect="auto")
        axes[0, c].set_title(LABEL.get(name, name), fontsize=10)
        axes[0, c].set_xlabel("alpha (0 = BM25, 1 = Contriever)", fontsize=8)
        axes[0, c].set_ylabel("share of expanded query", fontsize=8)
        fig.colorbar(im, ax=axes[0, c], fraction=.046)
    fig.tight_layout()
    fig.savefig(out / "fig5_optimal_heatmap.png", dpi=200)
    plt.close(fig)
    # LLM scoring diagnostics: mean score per source and per position shown to the LLM
    import json
    import random
    bias = []
    for cache in sorted(root.rglob("llm_*_*.jsonl")):
        mode = next((m for m in ("paper_fixed", "code_shuffle", "paper", "code", "dat") if cache.name.startswith(f"llm_{m}_")), None)
        if mode is None or mode == "dat":
            continue
        name = cache.stem[len(f"llm_{mode}_"):]
        by_src, by_pos = {s: [] for s in ("OS", "OD", "ES", "ED")}, {i: [] for i in range(1, 5)}
        for line in cache.read_text().splitlines():
            r = json.loads(line)
            if not r["scores"]:
                continue
            order = ["ES", "OS", "ED", "OD"]
            if mode in ("paper", "code_shuffle"):
                order = ["OS", "OD", "ES", "ED"]
                random.Random(r["qid"]).shuffle(order)
            for i, src in enumerate(order, 1):
                by_src[src].append(r["scores"][src])
                by_pos[i].append(r["scores"][src])
        bias.append({"name": name, "mode": mode, **{f"mean {k}": round(float(np.mean(v)), 3) for k, v in by_src.items()},
                     **{f"pos {k}": round(float(np.mean(v)), 3) for k, v in by_pos.items()}})
    if bias:
        bdf = pd.DataFrame(bias)
        bdf.to_csv(out / "llm_score_bias.csv", index=False)
        print("\nLLM mean scores by source and by position:\n", bdf.to_string(index=False))
    print("\nSaved tables and figures to", out)


if __name__ == "__main__":
    main()
