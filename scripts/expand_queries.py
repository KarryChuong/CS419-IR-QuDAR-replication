"""Generate expanded queries (paper Appendix G) with LLaMA 3.1-8B-Instruct.

Output: <out-dir>/expanded_queries_<dataset>_<split>_expansion.json = {qid: passage},
the format qudar's config `queries.expanded.path` expects. Query order and the
--max-queries cut follow qudar's pipeline (GenericDataLoader order, first N),
so the file always covers the queries the pipeline will run. Resumable.

  python expand_queries.py --data-root /kaggle/input/qudar-beir-data/data \
      --datasets fiqa scidocs --split test --out-dir /kaggle/working/expansions

Step 10b additions (generation itself is unchanged: same prompt, greedy, 256 tokens):
  * queries are read without loading the corpus (queries.jsonl + qrels, the same order as
    GenericDataLoader.load and run_retrieval.load_split), so FEVER/DBPedia start in seconds;
  * --logprobs K (vLLM only) saves, for every generated token, its id, its log-prob and the
    top-K alternatives, to <out-dir>/logprobs/<dataset>_<split>/part_NNN.npz (one file per
    saved chunk; fields qids, offsets, token_ids, chosen_lp, top_ids, top_lp). Greedy output
    does not change. Token ids refer to the raw generation, before clean().
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def load_queries(data_dir: Path, split: str) -> dict[str, str]:
    """Queries of a split in GenericDataLoader.load order (first appearance in qrels), no corpus."""
    from beir.datasets.data_loader import GenericDataLoader

    loader = GenericDataLoader(data_folder=str(data_dir))
    loader.qrels_file = str(data_dir / "qrels" / f"{split}.tsv")
    loader._load_queries()
    loader._load_qrels()
    return {qid: loader.queries[qid] for qid in loader.qrels}

PROMPT = (
    "You are a neutral fact compiler. Your task is to write a single, dense, factual paragraph "
    "with no formatting. Crucially, do not answer the query. Your goal is to write a descriptive "
    "passage about the topic or entity that the answer refers to. Within this description, you "
    "must naturally embed the specific details mentioned in the query. The final connection must "
    "be inferred by the reader.\n\n"
    "Query : {query}\n"
    "Passage :"
)


def clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(passage\s*:\s*)", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def build_prompts(tokenizer, queries: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(query=q)}], tokenize=False, add_generation_prompt=True
        )
        for q in queries
    ]


def extract_logprobs(completion, k: int):
    """vLLM CompletionOutput -> (token_ids, chosen_lp, top_ids[k], top_lp[k]) as numpy arrays.

    Each position's dict holds the sampled token plus the top-k (k or k+1 entries).
    Missing alternatives (should not happen) are padded with id -1 and log-prob -inf.
    """
    tok = np.asarray(completion.token_ids, dtype=np.int32)
    chosen = np.empty(len(tok), dtype=np.float32)
    top_ids = np.full((len(tok), k), -1, dtype=np.int32)
    top_lp = np.full((len(tok), k), -np.inf, dtype=np.float32)
    for pos, (t, entries) in enumerate(zip(tok, completion.logprobs or [])):
        chosen[pos] = entries[int(t)].logprob if int(t) in entries else np.nan
        ranked = sorted(entries.items(), key=lambda kv: kv[1].logprob, reverse=True)[:k]
        for j, (tid, lp) in enumerate(ranked):
            top_ids[pos, j], top_lp[pos, j] = tid, lp.logprob
    return tok, chosen, top_ids, top_lp


def save_logprobs(path: Path, qids: list[str], rows: list[tuple]) -> None:
    offsets = np.cumsum([0] + [len(r[0]) for r in rows]).astype(np.int64)
    cat = lambda i, shape: np.concatenate([r[i] for r in rows]) if rows else np.zeros(shape)  # noqa: E731
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, qids=np.asarray(qids), offsets=offsets,
                        token_ids=cat(0, (0,)), chosen_lp=cat(1, (0,)), top_ids=cat(2, (0, 0)), top_lp=cat(3, (0, 0)))
    tmp.rename(path)


class VLLMBackend:
    def __init__(self, model: str, max_new_tokens: int, logprobs: int = 0) -> None:
        import torch
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model,
            dtype="half",  # T4 has no bf16
            tensor_parallel_size=max(torch.cuda.device_count(), 1),
            max_model_len=2048,
            gpu_memory_utilization=0.90,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.logprobs = logprobs
        self.params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, logprobs=logprobs or None)
        self.last_logprobs: list[tuple] = []

    def generate(self, queries: list[str]) -> list[str]:
        outputs = self.llm.generate(build_prompts(self.tokenizer, queries), self.params)
        if self.logprobs:
            self.last_logprobs = [extract_logprobs(o.outputs[0], self.logprobs) for o in outputs]
        return [clean(o.outputs[0].text) for o in outputs]


class HFBackend:
    def __init__(self, model: str, max_new_tokens: int, batch_size: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map="auto").eval()
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def generate(self, queries: list[str]) -> list[str]:
        prompts = build_prompts(self.tokenizer, queries)
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        results = [""] * len(prompts)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tokenizer.pad_token_id)
            for row, i in enumerate(idx):
                results[i] = clean(self.tokenizer.decode(out[row, enc["input_ids"].shape[1]:], skip_special_tokens=True))
            print(f"  {min(start + self.batch_size, len(order))}/{len(order)}", flush=True)
        return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="folder that holds fiqa/, scidocs/, ...")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max-queries", type=int, default=None, help="first N queries, as qudar --max-queries")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--chunk", type=int, default=2000, help="save after every chunk of queries")
    p.add_argument("--hf-batch-size", type=int, default=16)
    p.add_argument("--logprobs", type=int, default=0, help="vLLM: save top-K log-probs per generated token (0 = off)")
    args = p.parse_args()
    if args.logprobs and args.backend != "vllm":
        raise SystemExit("--logprobs needs --backend vllm")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = None

    for name in args.datasets:
        queries = load_queries(Path(args.data_root) / name, args.split)
        qids = list(queries.keys())[: args.max_queries]
        lp_dir = out_dir / "logprobs" / f"{name}_{args.split}"
        if args.logprobs:
            lp_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"expanded_queries_{name}_{args.split}_expansion.json"
        done = json.loads(out_path.read_text()) if out_path.exists() else {}
        todo = [q for q in qids if q not in done]
        print(f"{name}/{args.split}: {len(qids)} queries, {len(done)} done, {len(todo)} to generate", flush=True)
        if todo and backend is None:
            backend = VLLMBackend(args.model, args.max_new_tokens, args.logprobs) if args.backend == "vllm" \
                else HFBackend(args.model, args.max_new_tokens, args.hf_batch_size)
        for start in range(0, len(todo), args.chunk):
            chunk = todo[start:start + args.chunk]
            for qid, passage in zip(chunk, backend.generate([queries[q] for q in chunk])):
                done[qid] = passage
            if args.logprobs:  # write log-probs first: a chunk is only "done" once both files exist
                part = len(list(lp_dir.glob("part_*[0-9].npz")))
                save_logprobs(lp_dir / f"part_{part:03d}.npz", chunk, backend.last_logprobs)
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(done, ensure_ascii=False))
            tmp.rename(out_path)
            print(f"  saved {len(done)}/{len(qids)}", flush=True)
        empty = [q for q in qids if not done.get(q)]
        print(f"{name}/{args.split}: finished, {len(empty)} empty outputs", flush=True)


if __name__ == "__main__":
    main()
