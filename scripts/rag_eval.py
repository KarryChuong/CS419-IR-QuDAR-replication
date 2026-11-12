"""Step 7: end-to-end RAG evaluation (paper Table 3), one GPU notebook.

For 500 sampled test queries per dataset (FiQA, HotpotQA; random, seed 0) and every method below,
the top-k fused passages go to Qwen2.5-7B-Instruct, which writes an answer; the same model then
judges each answer against the gold answer with the QA-Eval prompt (Wang et al., 2023a,
"Evaluating Open-QA Evaluation"): Yes / No. "Gen." = share of answers judged correct.
Gold answers: HotpotQA metadata.answer (BEIR queries.jsonl); FiQA has no short answer, so the
relevant passages in qrels are used as golden answers.

Methods (as Table 3, plus ours):
  OS, OD, ES, ED                     single lists
  Retriever-Type Fusion              best fixed alpha on the test set (OS-OD axis)
  Query-Format Fusion                best fixed beta2 on the test set (OD-ED axis; matches the paper's numbers)
  QuDAR-simple (RRF), (Equal), QuDAR-confidence, QuDAR-llm
  Static-train, Model B (ours)       from Step 6
  Query-wise UB (oracle)             per-query best grid point (generation headroom)
Identical generation settings for all methods; identical (question, passages) inputs are generated once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_sweep import load_qrels  # noqa: E402
from step6 import confidence_weights, find, llm_weights, log, paired_tests  # noqa: E402

SRC = ["OS", "OD", "ES", "ED"]
KEYS = [("sparse_docs", "sparse_scores_normalized"), ("dense_docs", "dense_scores_normalized"),
        ("pgt_sparse_docs", "pgt_sparse_scores_normalized"), ("pgt_dense_docs", "pgt_dense_scores_normalized")]
PAPER_GEN = {"fiqa_test": {"OS": .14, "OD": .10, "ES": .12, "ED": .10, "Retriever-Type Fusion": .15, "Query-Format Fusion": .10,
                           "QuDAR-simple (RRF)": .16, "QuDAR-simple (Equal)": .16, "QuDAR-confidence": .17, "QuDAR-llm": .18},
             "hotpotqa_test": {"OS": .36, "OD": .28, "ES": .31, "ED": .33, "Retriever-Type Fusion": .40, "Query-Format Fusion": .39,
                               "QuDAR-simple (RRF)": .42, "QuDAR-simple (Equal)": .43, "QuDAR-confidence": .44, "QuDAR-llm": .44}}

GEN_SYSTEM = "You are a helpful assistant. Answer the question using only the information in the passages. Be concise."
GEN_USER = "Passages:\n{passages}\n\nQuestion: {question}\nAnswer:"
JUDGE = ("Here is a question, a set of golden answers (split with /), an AI-generated answer. "
         "Can you judge whether the AI-generated answer is correct according to the question and golden answers, "
         "simply answer Yes or No.\nQuestion: {question}\nGolden answers: {gold}\nAI answer: {answer}")


def fused_ranking(entry, weights, k):
    scores = {}
    for w, (dk, nk) in zip(weights, KEYS):
        if w == 0:
            continue
        for d, s in zip(entry[dk], entry[nk]):
            scores[d] = scores.get(d, 0.0) + w * float(s)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]


def rrf_ranking(entry, k, c=60):
    scores = {}
    for dk, _ in KEYS:
        for r, d in enumerate(entry[dk], 1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (c + r)
    return [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]


def recall_at(ranking, rel, k=10):
    pos = {d for d, g in rel.items() if g > 0}
    return len(pos & set(ranking[:k])) / len(pos) if pos else 0.0


def words(text, n):
    w = (text or "").split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


class LLM:
    def __init__(self, backend, model):
        self.backend = backend
        if backend == "dry":
            return
        from vllm import LLM as V, SamplingParams
        import torch
        self.SP = SamplingParams
        self.llm = V(model=model, dtype="half", tensor_parallel_size=max(torch.cuda.device_count(), 1),
                     max_model_len=4096, gpu_memory_utilization=0.90, enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()

    def chat(self, messages_list, max_tokens):
        if self.backend == "dry":
            return [random.choice(["Yes", "No"]) if "golden answers" in m[-1]["content"] else "dummy answer"
                    for m in messages_list]
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        outs = self.llm.generate(prompts, self.SP(temperature=0.0, max_tokens=max_tokens))
        return [o.outputs[0].text.strip() for o in outs]


def cached_run(cache_path: Path, items: dict, fn, batch=512):
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            done[r["key"]] = r["out"]
    todo = [k for k in items if k not in done]
    log(f"{cache_path.name}: {len(items)} unique, {len(done)} cached, {len(todo)} to run")
    with cache_path.open("a") as fh:
        for s in range(0, len(todo), batch):
            keys = todo[s:s + batch]
            for k, o in zip(keys, fn([items[k] for k in keys])):
                done[k] = o
                fh.write(json.dumps({"key": k, "out": o}) + "\n")
            fh.flush()
            log(f"  {min(s + batch, len(todo))}/{len(todo)}")
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/kaggle/input")
    p.add_argument("--out", required=True)
    p.add_argument("--names", nargs="+", default=["fiqa_test", "hotpotqa_test"])
    p.add_argument("--train", nargs="+", default=["fiqa_train", "hotpotqa_train"])
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--k", type=int, default=5, help="passages given to the generator")
    p.add_argument("--passage-words", type=int, default=180)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--backend", choices=["vllm", "dry"], default="vllm")
    p.add_argument("--max-answer-tokens", type=int, default=200)
    args = p.parse_args()
    root, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Static-train point: best mean nDCG@10 over the pooled train grids (as step6.static_train_point)
    tr = [np.load(find(root, "grid_metrics.npz", n)) for n in args.train]
    static_idx = int(np.argmax(np.concatenate([z["metrics"][:, :, 1].astype(np.float32) for z in tr]).mean(0)))

    samples, need = {}, {}
    for name in args.names:
        dataset, split = name.rsplit("_", 1)
        rr = json.loads(find(root, "retrieval_result.json", name).read_text())
        data_root = find(root, "queries.jsonl", dataset).parent.parent
        qrels = load_qrels(data_root, dataset, split)
        qmeta = {}
        with (data_root / dataset / "queries.jsonl").open(encoding="utf8") as h:
            for line in h:
                r = json.loads(line)
                qmeta[str(r["_id"])] = (r.get("text", ""), (r.get("metadata") or {}).get("answer"))
        z = np.load(find(root, "grid_metrics.npz", name))
        gq = {str(q): i for i, q in enumerate(z["qids"])}
        grid, W, G = z["grid"], z["weights"], z["metrics"].astype(np.float32)
        a, b1, b2 = grid.T
        rt_mask = np.isclose(b1, 0) & np.isclose(b2, 0)
        qf_mask = np.isclose(a, 1) & np.isclose(b1, 0)
        rt_idx = np.flatnonzero(rt_mask)[np.argmax(G[:, rt_mask, 1].mean(0))]
        qf_idx = np.flatnonzero(qf_mask)[np.argmax(G[:, qf_mask, 1].mean(0))]
        llm = pd.read_csv(find(root, f"llm_scores_paper_{name}.csv"), dtype={"qid": str}).set_index("qid")
        cw = find(root, f"chosen_weights_{name}.csv")
        mb = pd.read_csv(cw, dtype={"qid": str}).set_index("qid") if cw else None

        qids = sorted(random.Random(0).sample([q for q in rr if q in gq], args.n))
        rows = []
        for q in qids:
            e = rr[q]
            best = int(np.argmax(G[gq[q], :, 1] * 1e3 + G[gq[q], :, 0]))
            meth = {"OS": (1, 0, 0, 0), "OD": (0, 1, 0, 0), "ES": (0, 0, 1, 0), "ED": (0, 0, 0, 1),
                    "Retriever-Type Fusion": W[rt_idx], "Query-Format Fusion": W[qf_idx],
                    "QuDAR-simple (Equal)": (.25, .25, .25, .25), "QuDAR-confidence": confidence_weights(e),
                    "QuDAR-llm": llm_weights([llm.loc[q, c] for c in ("original_sparse", "original_dense",
                                                                       "expanded_sparse", "expanded_dense")]),
                    "Static-train (ours)": W[static_idx], "Query-wise UB (oracle)": W[best]}
            if mb is not None and q in mb.index:
                aa, bb1, bb2 = mb.loc[q, ["B_alpha", "B_beta1", "B_beta2"]]
                meth["Model B (ours)"] = ((1 - aa) * (1 - bb1), aa * (1 - bb2), (1 - aa) * bb1, aa * bb2)
            rankings = {m: fused_ranking(e, w, 10) for m, w in meth.items()}
            rankings["QuDAR-simple (RRF)"] = rrf_ranking(e, 10)
            question, answer = qmeta.get(q, ("", None))
            if dataset == "fiqa" or not answer:
                gold_ids = [d for d, g in qrels.get(q, {}).items() if g > 0][:3]
                need.setdefault(dataset, set()).update(gold_ids)
                gold = ("ids", gold_ids)
            else:
                gold = ("text", answer)
            for m, rk in rankings.items():
                need.setdefault(dataset, set()).update(rk[:args.k])
                rows.append({"name": name, "qid": q, "method": m, "top": rk[:args.k],
                             "recall10": recall_at(rk, qrels.get(q, {})), "question": question, "gold": gold})
        samples[name] = rows
        del rr
        log(f"{name}: {len(qids)} queries x {len({r['method'] for r in rows})} methods")

    texts = {}
    for dataset, ids in need.items():
        data_root = find(root, "queries.jsonl", dataset).parent.parent
        with (data_root / dataset / "corpus.jsonl").open(encoding="utf8") as h:
            for line in h:
                r = json.loads(line)
                if r.get("_id") in ids:
                    texts[(dataset, r["_id"])] = (" ".join(x for x in [r.get("title"), r.get("text")] if x)).strip()
        log(f"{dataset}: loaded {sum(1 for k in texts if k[0] == dataset)} passages")

    llm_engine = LLM(args.backend, args.model)
    gen_items, judge_rows = {}, []
    for name, rows in samples.items():
        dataset = name.rsplit("_", 1)[0]
        for r in rows:
            ctx = "\n".join(f"[{i + 1}] {words(texts.get((dataset, d), ''), args.passage_words)}" for i, d in enumerate(r["top"]))
            msg = [{"role": "system", "content": GEN_SYSTEM},
                   {"role": "user", "content": GEN_USER.format(passages=ctx, question=r["question"])}]
            r["gen_key"] = hashlib.sha1(json.dumps(msg).encode()).hexdigest()
            gen_items[r["gen_key"]] = msg
    answers = cached_run(out / "generations.jsonl", gen_items, lambda b: llm_engine.chat(b, args.max_answer_tokens))

    judge_items = {}
    for name, rows in samples.items():
        dataset = name.rsplit("_", 1)[0]
        for r in rows:
            kind, val = r["gold"]
            gold = val if kind == "text" else " / ".join(words(texts.get((dataset, d), ""), 150) for d in val)
            msg = [{"role": "user", "content": JUDGE.format(question=r["question"], gold=gold, answer=answers[r["gen_key"]])}]
            r["judge_key"] = hashlib.sha1(json.dumps(msg).encode()).hexdigest()
            judge_items[r["judge_key"]] = msg
    verdicts = cached_run(out / "judgments.jsonl", judge_items, lambda b: llm_engine.chat(b, 4))

    allrows = []
    for name, rows in samples.items():
        for r in rows:
            v = verdicts[r["judge_key"]].strip().lower()
            allrows.append({"name": name, "qid": r["qid"], "method": r["method"], "recall10": r["recall10"],
                            "correct": float(bool(re.match(r"\W*yes", v))), "answer": answers[r["gen_key"]],
                            "verdict": verdicts[r["judge_key"]]})
    df = pd.DataFrame(allrows)
    df.to_csv(out / "rag_per_query.csv", index=False)

    order = ["OS", "OD", "ES", "ED", "Retriever-Type Fusion", "Query-Format Fusion", "QuDAR-simple (RRF)",
             "QuDAR-simple (Equal)", "QuDAR-confidence", "QuDAR-llm", "Static-train (ours)", "Model B (ours)",
             "Query-wise UB (oracle)"]
    table = []
    for m in order:
        row = {"method": m}
        for name in args.names:
            sub = df[(df.name == name) & (df.method == m)]
            if len(sub):
                row[f"{name} Recall@10"] = round(sub.recall10.mean(), 4)
                row[f"{name} Gen."] = round(sub.correct.mean(), 3)
                row[f"{name} Gen. paper"] = PAPER_GEN.get(name, {}).get(m, np.nan)
        table.append(row)
    table = pd.DataFrame(table)
    table.to_csv(out / "table7_rag.csv", index=False)
    (out / "table7_rag.md").write_text(table.to_markdown(index=False))
    print("\n" + table.to_string(index=False), flush=True)

    pairs = [("QuDAR-confidence", "OS"), ("QuDAR-llm", "OS"), ("QuDAR-confidence", "Retriever-Type Fusion"),
             ("QuDAR-llm", "QuDAR-confidence"), ("Static-train (ours)", "QuDAR-confidence"),
             ("Static-train (ours)", "QuDAR-llm"), ("Model B (ours)", "Static-train (ours)"),
             ("Query-wise UB (oracle)", "QuDAR-confidence")]
    sig = []
    for name in args.names:
        piv = df[df.name == name].pivot(index="qid", columns="method", values="correct")
        for a_, b_ in pairs:
            if a_ in piv and b_ in piv:
                d, lo, hi, pb, pt = paired_tests(piv[a_].values, piv[b_].values)
                sig.append({"dataset": name, "A": a_, "B": b_, "Gen. diff": round(float(d), 3),
                            "95% CI": f"[{lo:.3f}, {hi:.3f}]", "p bootstrap": round(pb, 4), "p t-test": round(pt, 4),
                            "sig (p<0.05)": "*" if max(pb, pt) < 0.05 else ""})
    sig = pd.DataFrame(sig)
    sig.to_csv(out / "table7_significance.csv", index=False)
    (out / "table7_significance.md").write_text(sig.to_markdown(index=False))
    print("\n" + sig.to_string(index=False), flush=True)
    log(f"Saved Step 7 outputs to {out}")


if __name__ == "__main__":
    main()
