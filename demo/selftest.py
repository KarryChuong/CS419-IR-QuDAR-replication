"""Check that the laptop pipeline reproduces the experiments (Step 2 lists, Step 6 choices, Table numbers).

    python selftest.py            # all FiQA test queries, about 5-10 minutes on a laptop CPU
    python selftest.py --n 50     # quick check

For every test query (cached LLaMA expansion, cached gpt-4o-mini scores) it retrieves the 4 lists
live and compares with the Step 2 lists, then reports mean Recall@10 / nDCG@10 of each method,
live and from the cached lists. The cached column should equal Tables 1-2 / table6_main.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qudar_core import GRID, SRC, Engine, ranking_metrics

HERE = Path(__file__).resolve().parent
SHOW = ["OS", "OD", "ES", "ED", "QuDAR-simple (RRF)", "QuDAR-simple (Equal)", "QuDAR-confidence", "QuDAR-llm",
        "QuDAR-llm-debiased", "Static-train", "Model A", "Model B", "Query-wise UB (oracle)"]


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(HERE / "demo_data"))
    p.add_argument("--model", default="facebook/contriever")
    p.add_argument("--n", type=int, default=0, help="number of test queries (0 = all)")
    args = p.parse_args()
    E = Engine(args.data, model_name=args.model)
    qids = [q for q in E.queries if E.cached and q in E.cached["pos"]]
    if args.n:
        qids = qids[:args.n]
    res = {"live": {}, "cached": {}}
    overlap, model_ok, model_n = {s: [] for s in SRC}, {"A": 0, "B": 0}, {"A": 0, "B": 0}
    t0 = time.time()
    for i, q in enumerate(qids, 1):
        qt, et = E.queries[q], E.expansions.get(q, "")
        for mode in ("live", "cached"):
            entry, _, _ = E.retrieve(qt, et, use_cached_qid=q if mode == "cached" else None)
            if mode == "live":
                for s, v in E.parity(q, entry).items():
                    overlap[s].append(v)
            M, _ = E.methods(q, qt, et, entry, E.llm_cache.get(q), (.5, .5, .5))
            for name, m in M.items():
                ranked = [d for d, _, _ in E.rank(entry, m)]
                res[mode].setdefault(name, []).append(ranking_metrics(ranked, E.qrels[q]) or (0, 0, 0))
            if mode == "cached" and q in E.chosen:
                for tag in ("A", "B"):
                    if f"Model {tag}" in M and tag in E.chosen[q]:
                        model_n[tag] += 1
                        model_ok[tag] += tuple(np.round(GRID[M[f"Model {tag}"]["grid"]], 1)) == tuple(np.round(E.chosen[q][tag], 1))
        if i % 25 == 0 or i == len(qids):
            print(f"  {i}/{len(qids)} ({time.time() - t0:.0f}s)", flush=True)

    print(f"\nTop-10 trùng với danh sách Step 2 (trung bình trên {len(qids)} câu): "
          + ", ".join(f"{s} {np.mean(v):.2f}/10" for s, v in overlap.items()))
    print("Model chọn đúng điểm lưới như Step 6 (dùng danh sách đã lưu): "
          + ", ".join(f"Model {t} {model_ok[t]}/{model_n[t]}" for t in ("A", "B") if model_n[t]))
    rows = []
    for name in SHOW:
        if name in res["cached"]:
            c, live = np.mean(res["cached"][name], 0), np.mean(res["live"].get(name, [[np.nan] * 3]), 0)
            rows.append({"method": name, "Recall@10 (đã lưu)": round(c[0], 4), "Recall@10 (trực tiếp)": round(live[0], 4),
                         "nDCG@10 (đã lưu)": round(c[1], 4), "nDCG@10 (trực tiếp)": round(live[1], 4)})
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    out = Path(args.data) / "selftest.csv"
    df.to_csv(out, index=False)
    print(f"\nĐã lưu {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
