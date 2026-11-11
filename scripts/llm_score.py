"""Step 4: LLM relevance scoring of top-1 passages -> QuDAR-llm and DAT.

Modes
  paper : Prompt 1 of the paper (Appendix D), the 4 top-1 passages (OS, OD, ES, ED) shuffled per
          query (seed = qid) and scores mapped back, as the paper describes.
  code  : qudar/llm.py as released (fixed order ES, OS, ED, OD; its shorter prompt) - ablation.
  dat   : DAT baseline (Hsu & Tzeng 2025): top-1 dense and top-1 BM25 of the original query,
          prompt rebuilt from DAT Appendix A; alpha rules from the DAT paper.
Passages are corpus "text" only, as qudar.pipeline.load_or_create_llm_scores does.

Results are cached line by line in <out>/llm_<mode>_<name>.jsonl (resumable), then
  paper / code -> llm_scores_<mode>_<name>.csv (qudar format) + QuDAR-llm via qudar.fusion
  dat          -> dat_scores_<name>.csv + DAT fusion (alpha * OD + (1 - alpha) * OS)
and the metrics are appended to <out>/llm_eval.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PAPER_PROMPT = """You are an impartial evaluator of retrieval effectiveness. Score each anonymous passage with an integer from 0 to 5 (inclusive) using ONLY the criteria below. Return a JSON array of four integers in order, nothing else.

Scoring Criteria (0-5):
5 = Direct hit (clearly answers the question)
4 = Very close (high likelihood correct answer is nearby
3 = Somewhat close (related, partial answer or near-miss)
2 = Loosely related but likely off in ranking neighborhood
1 = Barely related; unlikely nearby
0 = Unrelated / off-track

Output: A JSON array of four integers (e.g., [3,4,5,1]). No extra text.

Query : {query}

You will see 4 anonymous retrieval results (Top1s), in random order.
1 : {p1}
2 : {p2}
3 : {p3}
4 : {p4}"""

# Rebuilt from the description of DAT's Appendix A (rubric 5 / 3-4 / 1-2 / 0, "dense first, BM25 second").
DAT_PROMPT = """You are an evaluator assessing the retrieval effectiveness of dense retrieval (cosine distance) and BM25 retrieval for finding the correct answer.

Task: Given a question and the top-1 search result from each method, score each method from 0 to 5 based on how likely the correct answer is to appear in or near that result.

Scoring Criteria:
5 = Direct hit: the result directly answers the question.
3-4 = Good wrong result: conceptually close to the answer (related entities, partial answer); the correct answer is likely nearby.
1-2 = Bad wrong result: loosely related but misleading; the correct answer is unlikely nearby.
0 = Completely off-track: unrelated to the question.

Output format: return exactly two integers separated by a space - first the dense retrieval score, then the BM25 score. Output nothing else. Example: 4 2

Question: {query}

Dense retrieval Top1 Result: {dense}

BM25 retrieval Top1 Result: {bm25}"""

SRC = {"OS": "sparse_docs", "OD": "dense_docs", "ES": "pgt_sparse_docs", "ED": "pgt_dense_docs"}


def build_request(mode: str, qid: str, query: str, text: dict):
    """Return (prompt, parser, order); parser maps the reply to a dict of scores per source.
    paper / paper_fixed use Prompt 1, code / code_shuffle use qudar/llm.py's prompt;
    paper / code_shuffle shuffle the passages per query (seed = qid), the others keep ES, OS, ED, OD."""
    if mode in ("paper", "code", "paper_fixed", "code_shuffle"):
        order = ["ES", "OS", "ED", "OD"]
        if mode in ("paper", "code_shuffle"):
            order = ["OS", "OD", "ES", "ED"]
            random.Random(qid).shuffle(order)
        if mode in ("paper", "paper_fixed"):
            prompt = PAPER_PROMPT.format(query=query, **{f"p{i + 1}": text[s] for i, s in enumerate(order)})
        else:
            from qudar.llm import PROMPT_TEMPLATE
            prompt = PROMPT_TEMPLATE.format(query=query, first_passage=text[order[0]], second_passage=text[order[1]],
                                            third_passage=text[order[2]], fourth_passage=text[order[3]])

        def parse(reply: str):
            m = re.search(r"\[[^\]]*\]", reply)
            vals = [int(x) for x in re.findall(r"-?\d+", m.group(0) if m else reply)]
            if len(vals) != 4:
                raise ValueError(reply)
            return {s: max(0, min(5, v)) for s, v in zip(order, vals)}
        return prompt, parse, order
    prompt = DAT_PROMPT.format(query=query, dense=text["OD"], bm25=text["OS"])

    def parse(reply: str):
        vals = [int(x) for x in re.findall(r"-?\d+", reply)]
        if len(vals) != 2:
            raise ValueError(reply)
        return {"OD": max(0, min(5, vals[0])), "OS": max(0, min(5, vals[1]))}
    return prompt, parse, ["OD", "OS"]


def dat_alpha(s_dense: int, s_bm25: int) -> float:
    if s_dense == 0 and s_bm25 == 0:
        return 0.5
    if s_dense == 5 and s_bm25 != 5:
        return 1.0
    if s_bm25 == 5 and s_dense != 5:
        return 0.0
    return round(s_dense / (s_dense + s_bm25), 1)


class Caller:
    def __init__(self, model: str, dry_run: bool) -> None:
        self.model, self.dry_run = model, dry_run
        if not dry_run:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)

    def __call__(self, prompt: str) -> tuple[str, int]:
        if self.dry_run:  # offline test: random but well-formed replies
            n = 2 if "BM25 retrieval Top1" in prompt else 4
            vals = [random.randint(0, 5) for _ in range(n)]
            return (json.dumps(vals) if "JSON array" in prompt else " ".join(map(str, vals))), 0
        r = self.client.chat.completions.create(model=self.model, temperature=0,
                                                messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip(), int(getattr(r.usage, "total_tokens", 0) or 0)


def score_one(caller, mode, qid, query, text, tries=8):
    prompt, parse, _ = build_request(mode, qid, query, text)
    last = ""
    for attempt in range(tries):
        try:
            reply, tokens = caller(prompt)
            last = reply
            return {"qid": qid, "scores": parse(reply), "raw": reply, "tokens": tokens}
        except ValueError:          # malformed reply: ask again
            continue
        except Exception as exc:    # rate limit / network: back off
            if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError", "NotFoundError"):
                raise SystemExit(f"OpenAI {type(exc).__name__}: check OPENAI_API_KEY (must be the secret key 'sk-...') and the model name")
            wait = min(60, 2 ** attempt + random.random())
            print(f"  {qid}: {type(exc).__name__}, retry in {wait:.0f}s", flush=True)
            time.sleep(wait)
    return {"qid": qid, "scores": None, "raw": last, "tokens": 0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--mode", choices=["paper", "code", "paper_fixed", "code_shuffle", "dat"], default="paper")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--llm-tau", type=float, default=3.0)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="no API calls (pipeline test)")
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.repo) / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pandas as pd
    from qudar.evaluation import evaluate_run
    from qudar.fusion import build_llm_run, build_llm_weight_lookup
    from run_retrieval import load_split

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    caller = Caller(args.model, args.dry_run)
    for name in args.names:
        dataset, split = name.rsplit("_", 1)
        # results on a subset get their own name, so they never overwrite full-set rows
        eval_name = f"{name}_first{args.max_queries}" if args.max_queries else name
        rr = json.loads((Path(args.runs) / name / "retrieval_result.json").read_text())
        qids, qtexts, qrels = load_split(Path(args.data_root) / dataset, split, None)
        query_of = dict(zip(qids, qtexts))
        qids = [q for q in rr][: args.max_queries]
        rr = {q: rr[q] for q in qids}
        need = {rr[q][SRC[s]][0] for q in qids for s in SRC}
        corpus_text = {}
        with (Path(args.data_root) / dataset / "corpus.jsonl").open(encoding="utf8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("_id") in need:
                    corpus_text[row["_id"]] = row.get("text") or ""
        cache = out / f"llm_{args.mode}_{name}.jsonl"
        done = {}
        if cache.exists():
            for line in cache.read_text().splitlines():
                row = json.loads(line)
                if row["scores"] is not None:
                    done[row["qid"]] = row
        todo = [q for q in qids if q not in done]
        print(f"{name} [{args.mode}]: {len(qids)} queries, {len(done)} cached, {len(todo)} to score", flush=True)
        lock, t0 = threading.Lock(), time.time()
        with cache.open("a") as fh, ThreadPoolExecutor(args.workers) as pool:
            futs = [pool.submit(score_one, caller, args.mode, q, query_of[q],
                                {s: corpus_text.get(rr[q][SRC[s]][0], "") for s in SRC}) for q in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                with lock:
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    if row["scores"] is not None:
                        done[row["qid"]] = row
                if i % 500 == 0 or i == len(futs):
                    rate = i / max(time.time() - t0, 1e-9) * 60
                    print(f"  {i}/{len(futs)} scored, {rate:.0f}/min, ~{(len(futs) - i) / max(rate, 1e-9):.0f} min left", flush=True)
        failed = [q for q in qids if q not in done]
        tokens = [done[q]["tokens"] for q in done]
        print(f"{name}: {len(failed)} failed (equal weights used), avg tokens/query {sum(tokens) / max(len(tokens), 1):.0f}", flush=True)

        if args.mode != "dat":
            rows = [{"qid": q, "original_sparse": done[q]["scores"]["OS"], "original_dense": done[q]["scores"]["OD"],
                     "expanded_sparse": done[q]["scores"]["ES"], "expanded_dense": done[q]["scores"]["ED"]} for q in done]
            pd.DataFrame(rows).to_csv(out / f"llm_scores_{args.mode}_{eval_name}.csv", index=False)
            lookup = build_llm_weight_lookup(rows, tau=args.llm_tau)
            method = {"paper": "qudar_llm", "code": "qudar_llm_codeprompt", "paper_fixed": "qudar_llm_paper_fixed",
                      "code_shuffle": "qudar_llm_code_shuffle"}[args.mode]
        else:
            rows = [{"qid": q, "dense": done[q]["scores"]["OD"], "bm25": done[q]["scores"]["OS"],
                     "alpha": dat_alpha(done[q]["scores"]["OD"], done[q]["scores"]["OS"])} for q in done]
            pd.DataFrame(rows).to_csv(out / f"dat_scores_{eval_name}.csv", index=False)
            lookup = {r["qid"]: (1 - r["alpha"], r["alpha"], 0.0, 0.0) for r in rows}  # OS, OD, ES, ED
            lookup.update({q: (0.5, 0.5, 0.0, 0.0) for q in failed})
            method = "dat"
        run, meta = build_llm_run(rr, lookup)
        summary, per_query = evaluate_run(qrels, run, at_k=10)
        json.dump(per_query, open(out / f"perquery_{method}_{eval_name}.json", "w"))
        json.dump(meta, open(out / f"weights_{method}_{eval_name}.json", "w"))
        row = {"name": eval_name, "method": method, "queries": len(qids), "failed": len(failed), **summary}
        ev = out / "llm_eval.csv"
        pd.DataFrame([row]).to_csv(ev, mode="a", header=not ev.exists(), index=False)
        print(pd.DataFrame([row])[["name", "method", "Recall@10", "nDCG@10", "Precision@1"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
