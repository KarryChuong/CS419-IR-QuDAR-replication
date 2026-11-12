"""Core of the Step 8 demo: the six IR stages for one query, on the FiQA corpus, on a laptop CPU.

Same formulas as the experiments (copied, not re-invented):
  BM25            scripts/bm25_fast.py BM25sSparseRetriever (bm25s lucene, English stopwords, Snowball stemmer)
  Contriever      qudar.retrievers.DenseEncoder ("query: " / "passage: " prefixes, mean pooling, L2 norm);
                  the expanded query is encoded as a passage, as qudar.pipeline does
  normalisation   qudar.results._normalize (min-max over the top-1000 scores of each list)
  fusion          qudar.fusion: weighted sum of the normalised scores; RRF k = 60
  weights         QuDAR-confidence (tau 2, top-10 margins), QuDAR-llm (tau 3), Step 6 models and features,
                  position-debiased QuDAR-llm (6.4), Static-train, oracle = scripts/grid_sweep.py
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np

SRC = ["OS", "OD", "ES", "ED"]
SRC_LABEL = {"OS": "BM25 · query gốc", "OD": "Contriever · query gốc",
             "ES": "BM25 · query mở rộng", "ED": "Contriever · query mở rộng"}
KEYS = [("sparse_docs", "sparse_scores", "sparse_scores_normalized"),
        ("dense_docs", "dense_scores", "dense_scores_normalized"),
        ("pgt_sparse_docs", "pgt_sparse_scores", "pgt_sparse_scores_normalized"),
        ("pgt_dense_docs", "pgt_dense_scores", "pgt_dense_scores_normalized")]
QWORDS = ("what", "how", "why", "when", "where", "who", "which", "is", "can", "does", "do", "should")
TOP_K = 1000

EXPANSION_PROMPT = (
    "You are a neutral fact compiler. Your task is to write a single, dense, factual paragraph "
    "with no formatting. Crucially, do not answer the query. Your goal is to write a descriptive "
    "passage about the topic or entity that the answer refers to. Within this description, you "
    "must naturally embed the specific details mentioned in the query. The final connection must "
    "be inferred by the reader.\n\n"
    "Query : {query}\n"
    "Passage :"
)
LLM_PROMPT = """You are an impartial evaluator of retrieval effectiveness. Score each anonymous passage with an integer from 0 to 5 (inclusive) using ONLY the criteria below. Return a JSON array of four integers in order, nothing else.

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
GEN_SYSTEM = "You are a helpful assistant. Answer the question using only the information in the passages. Be concise."
GEN_USER = "Passages:\n{passages}\n\nQuestion: {question}\nAnswer:"


# ------------------------------------------------------------------ formulas
def make_grid(step=0.1):
    vals = np.round(np.arange(0, 1 + 1e-9, step), 4)
    grid = np.array([(a, b1, b2) for a in vals for b1 in vals for b2 in vals])
    a, b1, b2 = grid.T
    return grid, np.stack([(1 - a) * (1 - b1), a * (1 - b2), (1 - a) * b1, a * b2], axis=1)


GRID, GRID_W = make_grid()
COARSE = np.flatnonzero(np.all(np.isclose(np.round(GRID * 5) / 5, GRID), axis=1))   # 216 points, as Step 6


def to_weights(a, b1, b2):
    return np.array([(1 - a) * (1 - b1), a * (1 - b2), (1 - a) * b1, a * b2])


def to_axes(w):
    """(alpha, beta1, beta2) of any 4-weight vector: alpha = dense share, beta = expanded share inside each side
    (None where undefined, e.g. beta2 when no weight goes to the dense lists)."""
    w = np.asarray(w, float)
    sp, de = w[0] + w[2], w[1] + w[3]
    tot = sp + de
    return (de / tot if tot else None, w[2] / sp if sp > 1e-12 else None, w[3] / de if de > 1e-12 else None)


def softmax(x, tau):
    z = (np.asarray(x, float) - np.max(x)) / tau
    e = np.exp(z)
    return e / e.sum()


def normalize(scores):
    s = np.asarray(scores, float)
    if s.size == 0:
        return s
    return (s - s.min()) / (s.max() - s.min() + 1e-12)


def confidence_weights(entry, topk=10, tau=2.0):
    margins = []
    for _, rk, nk in KEYS:
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


def fuse(entry, w, k=10):
    """Weighted sum of normalised scores. Returns [(doc, fused score, contribution per list)]."""
    acc = {}
    for j, (dk, _, nk) in enumerate(KEYS):
        if w[j] == 0:
            continue
        for d, s in zip(entry[dk], entry[nk]):
            c = acc.setdefault(d, np.zeros(4))
            c[j] += w[j] * float(s)
    ranked = sorted(acc.items(), key=lambda x: -x[1].sum())[:k]
    return [(d, float(c.sum()), c) for d, c in ranked]


def rrf(entry, k=10, c=60):
    acc = {}
    for j, (dk, _, _) in enumerate(KEYS):
        for r, d in enumerate(entry[dk], 1):
            v = acc.setdefault(d, np.zeros(4))
            v[j] += 1.0 / (c + r)
    ranked = sorted(acc.items(), key=lambda x: -x[1].sum())[:k]
    return [(d, float(v.sum()), v) for d, v in ranked]


def ranking_metrics(ranked_ids, rel, k=10):
    """Recall@10, nDCG@10 (gain = qrel grade), P@1 of one ranking - same definitions as grid_sweep.py."""
    pos = {d: g for d, g in rel.items() if g > 0}
    if not pos:
        return None
    top = ranked_ids[:k]
    dcg = sum(pos.get(d, 0) / np.log2(i + 2) for i, d in enumerate(top))
    ideal = sorted(pos.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return (sum(d in pos for d in top) / len(pos), dcg / idcg, float(bool(top) and top[0] in pos))


def grid_metrics(entry, rel, weights, k=10):
    """scripts/grid_sweep.query_metrics: metrics of many weight vectors at once (G x 3)."""
    doc_index, rows, cols, vals = {}, [], [], []
    for col, (dk, _, nk) in enumerate(KEYS):
        for doc, score in zip(entry[dk], entry[nk]):
            rows.append(doc_index.setdefault(doc, len(doc_index)))
            cols.append(col)
            vals.append(score)
    S = np.zeros((len(doc_index), 4))
    np.add.at(S, (np.array(rows), np.array(cols)), np.array(vals, dtype=float))
    fused = S @ weights.T
    kk = min(k, fused.shape[0])
    top = np.argpartition(-fused, kk - 1, axis=0)[:kk]
    order = np.argsort(-np.take_along_axis(fused, top, axis=0), axis=0, kind="stable")
    top = np.take_along_axis(top, order, axis=0)
    gains = np.zeros(len(doc_index))
    for doc, g in rel.items():
        if g > 0 and doc in doc_index:
            gains[doc_index[doc]] = g
    n_rel = sum(1 for g in rel.values() if g > 0)
    out = np.zeros((weights.shape[0], 3))
    if n_rel == 0:
        return out
    tg = gains[top]
    disc = 1 / np.log2(np.arange(2, kk + 2))[:, None]
    ideal = np.sort([g for g in rel.values() if g > 0])[::-1][:k]
    idcg = float((ideal / np.log2(np.arange(2, len(ideal) + 2))).sum())
    out[:, 0] = (tg > 0).sum(axis=0) / n_rel
    out[:, 1] = (tg * disc).sum(axis=0) / idcg
    out[:, 2] = (tg[0] > 0).astype(float)
    return out


def query_features(entry, qtext, etext):
    """step6.query_features (55 features of Model A)."""
    f, tops = {}, []
    for name, (dk, rk, nk) in zip(SRC, KEYS):
        raw = np.asarray(entry[rk][:10], float)
        nrm = np.asarray(entry[nk][:10], float)
        tops.append(set(entry[dk][:10]))
        f[f"{name}_raw_top1"] = raw[0] if raw.size else 0.0
        f[f"{name}_raw_margin"] = (raw[0] - raw[1]) if raw.size > 1 else 0.0
        f[f"{name}_raw_ratio2"] = (raw[1] / raw[0]) if raw.size > 1 and raw[0] != 0 else 0.0
        f[f"{name}_raw_drop10"] = ((raw[0] - raw[-1]) / abs(raw[0])) if raw.size > 1 and raw[0] != 0 else 0.0
        f[f"{name}_raw_std10"] = float(raw.std()) if raw.size else 0.0
        f[f"{name}_norm_2"] = nrm[1] if nrm.size > 1 else 0.0
        f[f"{name}_norm_10"] = nrm[-1] if nrm.size else 0.0
        f[f"{name}_norm_mean10"] = float(nrm.mean()) if nrm.size else 0.0
    top1 = [entry[dk][0] if entry[dk] else None for dk, _, _ in KEYS]
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


def llm_features(scores, qid):
    order = paper_order(qid)
    f = {}
    for name in SRC:
        f[f"llm_{name}"] = float(scores[name]) if scores else np.nan
        f[f"llm_pos_{name}"] = float(order.index(name) + 1)
    return f


def clean_expansion(text):
    text = re.sub(r"^(passage\s*:\s*)", "", (text or "").strip(), flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def live_qid(text):
    return "live-" + hashlib.sha1(text.strip().encode("utf8")).hexdigest()[:10]


def words(text, n):
    w = (text or "").split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


# ------------------------------------------------------------------ LLM backends (OpenAI or Ollama)
class Chat:
    """OpenAI chat API, or Ollama through its OpenAI-compatible endpoint (http://localhost:11434/v1)."""

    def __init__(self):
        self._clients = {}

    def __call__(self, backend, model, messages, max_tokens):
        import os
        from openai import OpenAI
        if backend not in self._clients:
            if backend == "openai":
                key = os.environ.get("OPENAI_API_KEY", "").strip()
                if not key.startswith("sk-"):
                    raise RuntimeError("Chưa có OPENAI_API_KEY (phải bắt đầu bằng sk-). Đặt trong file demo/.env.")
                self._clients[backend] = OpenAI(api_key=key, max_retries=2, timeout=60)
            else:
                url = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")
                self._clients[backend] = OpenAI(base_url=url, api_key="ollama", max_retries=0, timeout=300)
        r = self._clients[backend].chat.completions.create(model=model, temperature=0, max_tokens=max_tokens,
                                                           messages=messages)
        return (r.choices[0].message.content or "").strip()


# ------------------------------------------------------------------ the engine
class Engine:
    def __init__(self, data_dir, model_name="facebook/contriever", log=print):
        import os
        self.dir = Path(data_dir)
        self.log = log
        self.model_name = model_name
        self.stats = {}
        t = time.time()
        man = json.loads((self.dir / "manifest.json").read_text(encoding="utf8"))
        self.manifest = man
        self.ids, self.titles, self.texts = [], [], []
        with (self.dir / "corpus.jsonl").open(encoding="utf8") as h:
            for line in h:
                r = json.loads(line)
                self.ids.append(r["_id"])
                self.titles.append(r.get("title") or "")
                self.texts.append(r.get("text") or "")
        self.pos = {d: i for i, d in enumerate(self.ids)}
        self.queries = json.loads((self.dir / "queries.json").read_text(encoding="utf8"))
        self.qrels = json.loads((self.dir / "qrels.json").read_text(encoding="utf8"))
        self.expansions = json.loads((self.dir / "expansions.json").read_text(encoding="utf8"))
        self.llm_cache = json.loads((self.dir / "llm_scores.json").read_text(encoding="utf8"))
        self.bias = {int(k): v for k, v in (man.get("position_bias") or {}).items()} or None
        self.static = man.get("static_train")
        self.fixed = {k: man[k] for k in ("retriever_type_fusion", "query_format_fusion") if k in man}
        self.stats["load_corpus_s"] = time.time() - t
        log(f"corpus: {len(self.ids)} tài liệu, {len(self.queries)} câu hỏi test ({time.time() - t:.1f}s)")

        cr = self.dir / "cached_runs.npz"
        self.cached = None
        if cr.exists():
            z = np.load(cr)
            self.cached = {"pos": {str(q): i for i, q in enumerate(z["qids"])}, "docs": z["docs"], "scores": z["scores"]}

        self.models = {}
        try:
            import lightgbm as lgb
            for tag in ("A", "B"):
                mf, ff = self.dir / f"model_{tag}.txt", self.dir / f"model_{tag}_features.json"
                if mf.exists() and ff.exists():
                    self.models[tag] = (lgb.Booster(model_file=str(mf)), json.loads(ff.read_text()))
        except ImportError:
            log("lightgbm chưa cài -> bỏ Model A/B")
        self.chosen = {}
        cw = self.dir / "chosen_weights.csv"
        if cw.exists():
            import csv
            with cw.open(encoding="utf8") as h:
                for r in csv.DictReader(h):
                    self.chosen[r["qid"]] = {t: tuple(float(r[f"{t}_{x}"]) for x in ("alpha", "beta1", "beta2"))
                                             for t in ("A", "B") if f"{t}_alpha" in r}

        # stage "indexing": BM25 inverted index (built once, cached on disk) + dense matrix
        t = time.time()
        import bm25s
        import Stemmer
        self.bm25s = bm25s
        self.stemmer = Stemmer.Stemmer("english")
        bdir = self.dir / "bm25_index"
        if (bdir / "params.index.json").exists():
            self.bm25 = bm25s.BM25.load(str(bdir))
            how = "nạp từ đĩa"
        else:
            texts = [(" ".join(p for p in [ti, tx] if p)).strip() for ti, tx in zip(self.titles, self.texts)]
            tokens = bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False)
            self.bm25 = bm25s.BM25(method="lucene")
            self.bm25.index(tokens, show_progress=False)
            self.bm25.save(str(bdir))
            how = "xây mới"
        self.stats["bm25_index_s"] = time.time() - t
        self.stats["bm25_vocab"] = len(self.bm25.vocab_dict) if hasattr(self.bm25, "vocab_dict") and self.bm25.vocab_dict else None
        log(f"BM25: {how} ({time.time() - t:.1f}s)")
        t = time.time()
        self.emb = np.load(self.dir / "dense_emb.npy").astype(np.float32)
        assert self.emb.shape[0] == len(self.ids), "dense_emb.npy không khớp corpus"
        self.stats["dense_load_s"] = time.time() - t
        self.encoder = None
        self.chat = Chat()
        self.threads = int(os.environ.get("QUDAR_THREADS", "0")) or None

    # ---------------- stage: query analysis + retrieval
    def _encoder(self):
        if self.encoder is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            if self.threads:
                torch.set_num_threads(self.threads)
            t = time.time()
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            self.encoder = AutoModel.from_pretrained(self.model_name).eval()
            self.torch = torch
            self.log(f"Contriever: nạp {self.model_name} ({time.time() - t:.1f}s)")
        return self.encoder

    def encode(self, text, prefix):
        model = self._encoder()
        torch = self.torch
        with torch.no_grad():
            enc = self.tok([prefix + text], padding=True, truncation=True, max_length=512, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            v = torch.nn.functional.normalize(v, p=2, dim=1)
        return v[0].numpy().astype(np.float16).astype(np.float32), int(enc["attention_mask"].sum())   # fp16 as DenseIndexer

    def bm25_tokens(self, text):
        return self.bm25s.tokenize([text], stopwords="en", stemmer=self.stemmer, return_ids=False,
                                   show_progress=False)[0]

    def bm25_search(self, text, k=TOP_K):
        toks = self.bm25s.tokenize([text], stopwords="en", stemmer=self.stemmer, return_ids=False, show_progress=False)
        toks = [[t for t in toks[0] if t in self.bm25.vocab_dict]]     # words absent from the corpus score 0
        if not toks[0]:
            return [], np.zeros(0)
        docs, scores = self.bm25.retrieve(toks, k=min(k, len(self.ids)), show_progress=False, backend_selection="numpy")
        return [self.ids[i] for i in docs[0]], np.asarray(scores[0], float)

    def dense_search(self, vec, k=TOP_K):
        s = self.emb @ vec
        top = np.argsort(-s, kind="stable")[:k]          # exact search, ties -> lower corpus index (a few ms)
        return [self.ids[i] for i in top], s[top].astype(float)

    def retrieve(self, query, expanded, use_cached_qid=None):
        timings, info = {}, {}
        if use_cached_qid is not None and self.cached and use_cached_qid in self.cached["pos"]:
            i = self.cached["pos"][use_cached_qid]
            entry = {}
            for j, (dk, rk, nk) in enumerate(KEYS):
                ok = self.cached["docs"][i, j] >= 0
                entry[dk] = [self.ids[d] for d in self.cached["docs"][i, j][ok]]
                entry[rk] = self.cached["scores"][i, j][ok].astype(float).tolist()
                entry[nk] = normalize(entry[rk]).tolist()
            info["mode"] = "cached"
            return entry, timings, info
        entry = {}
        t = time.time(); d, s = self.bm25_search(query); timings["Truy xuất OS (BM25, query gốc)"] = time.time() - t
        entry["sparse_docs"], entry["sparse_scores"] = d, s
        t = time.time(); d, s = self.bm25_search(expanded) if expanded else ([], np.zeros(0))
        timings["Truy xuất ES (BM25, query mở rộng)"] = time.time() - t
        entry["pgt_sparse_docs"], entry["pgt_sparse_scores"] = d, s
        t = time.time(); vq, nq = self.encode(query, "query: "); timings["Encode query gốc (Contriever)"] = time.time() - t
        t = time.time(); ve, ne = self.encode(expanded, "passage: ") if expanded else (None, 0)
        timings["Encode query mở rộng (Contriever)"] = time.time() - t
        t = time.time(); d, s = self.dense_search(vq); timings["Truy xuất OD (tích vô hướng)"] = time.time() - t
        entry["dense_docs"], entry["dense_scores"] = d, s
        t = time.time(); d, s = self.dense_search(ve) if ve is not None else ([], np.zeros(0))
        timings["Truy xuất ED (tích vô hướng)"] = time.time() - t
        entry["pgt_dense_docs"], entry["pgt_dense_scores"] = d, s
        t = time.time()
        for dk, rk, nk in KEYS:
            entry[nk] = normalize(entry[rk]).tolist()
            entry[rk] = np.asarray(entry[rk], float).tolist()
        timings["Chuẩn hoá min-max (top-1000)"] = time.time() - t
        info.update(mode="live", q_tokens=self.bm25_tokens(query), e_tokens=self.bm25_tokens(expanded) if expanded else [],
                    q_wordpieces=nq, e_wordpieces=ne)
        return entry, timings, info

    def parity(self, qid, entry):
        """Overlap of the live top-10 with the Step 2 lists (same query, same expansion)."""
        if not self.cached or qid not in self.cached["pos"]:
            return None
        ref, _, _ = self.retrieve("", "", use_cached_qid=qid)
        return {s: len(set(entry[dk][:10]) & set(ref[dk][:10])) for s, (dk, _, _) in zip(SRC, KEYS)}

    # ---------------- stage: query expansion and LLM scoring
    def expand(self, query, backend, model):
        out = self.chat(backend, model, [{"role": "user", "content": EXPANSION_PROMPT.format(query=query)}], 256)
        return clean_expansion(out)

    def llm_score(self, qid, query, entry, backend, model):
        order = paper_order(qid)
        top1 = {s: self.texts[self.pos[entry[dk][0]]] if entry[dk] else "" for s, (dk, _, _) in zip(SRC, KEYS)}
        prompt = LLM_PROMPT.format(query=query, **{f"p{i + 1}": top1[s] for i, s in enumerate(order)})
        reply = self.chat(backend, model, [{"role": "user", "content": prompt}], 50)
        m = re.search(r"\[[^\]]*\]", reply)
        vals = [int(x) for x in re.findall(r"-?\d+", m.group(0) if m else reply)]
        if len(vals) != 4:
            raise RuntimeError(f"LLM trả lời không đúng định dạng: {reply!r}")
        return {s: max(0, min(5, v)) for s, v in zip(order, vals)}, order

    # ---------------- stage: ranking (all weighting methods)
    def predict_model(self, tag, feats):
        booster, cols = self.models[tag]
        x = np.array([[feats.get(c, np.nan) for c in cols]], np.float32)
        if np.isnan(x).any():
            return None
        rows = np.hstack([np.repeat(x, len(COARSE), 0), GRID[COARSE], GRID_W[COARSE]]).astype(np.float32)
        pred = booster.predict(rows)
        return int(COARSE[int(np.argmax(pred))])

    def methods(self, qid, qtext, etext, entry, llm_scores, manual):
        """name -> dict(weights, note, group). RRF has weights None (rank-based)."""
        M = {}
        for j, s in enumerate(SRC):
            w = np.zeros(4); w[j] = 1
            M[s] = dict(w=w, group="1 danh sách", note=SRC_LABEL[s])
        for key, label, note in [("retriever_type_fusion", "Retriever-Type Fusion", "α cố định tốt nhất trên test (oracle của paper)"),
                                 ("query_format_fusion", "Query-Format Fusion", "β2 cố định tốt nhất trên test (oracle của paper)")]:
            if key in self.fixed:
                M[label] = dict(w=np.array(self.fixed[key]["weights"]), group="Trộn 1 chiều", note=note)
        M["QuDAR-simple (RRF)"] = dict(w=None, group="QuDAR (paper)", note="1/(60 + hạng), cộng 4 danh sách")
        M["QuDAR-simple (Equal)"] = dict(w=np.full(4, .25), group="QuDAR (paper)", note="mỗi danh sách 0.25")
        cm = []
        for _, rk, nk in KEYS:
            s = np.asarray(entry[nk][:10], float)
            cm.append(0.0 if s.size == 0 else float(s[0]) if s.size == 1 else max(float(s[0] - s[1]), 0.0))
        M["QuDAR-confidence"] = dict(w=confidence_weights(entry), group="QuDAR (paper)",
                                     note="softmax(margin top1−top2 / τ=2): " + ", ".join(f"{s} {m:.3f}" for s, m in zip(SRC, cm)))
        if llm_scores:
            sc = [llm_scores[s] for s in SRC]
            M["QuDAR-llm"] = dict(w=llm_weights(sc), group="QuDAR (paper)",
                                  note="softmax(điểm LLM / τ=3): " + ", ".join(f"{s} {int(v)}" for s, v in zip(SRC, sc)))
            if self.bias:
                order = paper_order(qid)
                adj = [llm_scores[s] - self.bias[order.index(s) + 1] for s in SRC]
                M["QuDAR-llm-debiased"] = dict(w=llm_weights(adj), group="Mở rộng (của mình)",
                                               note="trừ độ lệch theo vị trí ước lượng trên train (6.4)")
        if self.static:
            M["Static-train"] = dict(w=np.array(self.static["weights"]), group="Mở rộng (của mình)",
                                     note="trọng số cố định tốt nhất trên FiQA-train + HotpotQA-train")
        feats = query_features(entry, qtext, etext)
        feats.update(llm_features(llm_scores, qid))
        for tag, need in (("A", "không dùng LLM"), ("B", "thêm 4 điểm LLM + vị trí")):
            if tag in self.models:
                g = self.predict_model(tag, feats)
                if g is not None:
                    M[f"Model {tag}"] = dict(w=GRID_W[g], group="Mở rộng (của mình)",
                                             note=f"LightGBM dự đoán bề mặt nDCG trên 216 điểm lưới ({need})", grid=g)
        M["Thủ công"] = dict(w=to_weights(*manual), group="Thử tay", note="trọng số theo 3 thanh trượt α, β1, β2")
        rel = self.qrels.get(qid)
        if rel:
            G = grid_metrics(entry, rel, GRID_W)
            best = int(np.argmax(G[:, 1] * 1e3 + G[:, 0]))
            M["Query-wise UB (oracle)"] = dict(w=GRID_W[best], group="Oracle (biết nhãn)",
                                               note=f"điểm lưới tốt nhất cho riêng câu này; {int((G[:, 1] >= G[best, 1] - 1e-9).sum())}/1331 điểm hoà")
        return M, feats

    def rank(self, entry, m, k=10):
        return rrf(entry, k) if m["w"] is None else fuse(entry, m["w"], k)

    def passage(self, doc_id):
        i = self.pos[doc_id]
        return self.titles[i], self.texts[i]

    def rag_messages(self, question, ranked, k=5, n_words=180):
        ctx = "\n".join(f"[{i + 1}] {words((' '.join(p for p in self.passage(d) if p)).strip(), n_words)}"
                        for i, (d, _, _) in enumerate(ranked[:k]))
        return [{"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": GEN_USER.format(passages=ctx, question=question)}]
