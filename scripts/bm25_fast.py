"""Drop-in replacement for qudar.retrievers.BM25SparseRetriever.

Reproduces rank_bm25.BM25Okapi scoring exactly (k1=1.5, b=0.75, epsilon=0.25,
same tokenization: lower().split() + English Snowball stemmer), but stores the
corpus as a sparse matrix so it scales to corpora with millions of documents.
"""
from __future__ import annotations

import math
from array import array
from collections import Counter

import numpy as np
import Stemmer
from scipy import sparse
from tqdm import tqdm


class FastBM25SparseRetriever:
    def __init__(self, corpus: dict, top_k: int, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25) -> None:
        self.top_k = top_k
        self.stemmer = Stemmer.Stemmer("english")
        self.doc_ids = list(corpus.keys())
        self.vocab: dict[str, int] = {}

        rows, cols, tfs = array("i"), array("i"), array("f")
        doc_len = np.zeros(len(self.doc_ids), dtype=np.float32)
        for row, doc_id in enumerate(tqdm(self.doc_ids, desc="BM25 indexing")):
            document = corpus[doc_id]
            merged = (" ".join(part for part in [document.get("title", ""), document.get("text", "")] if part)).strip()
            tokens = self._tokenize(merged)
            doc_len[row] = len(tokens)
            for token, count in Counter(tokens).items():
                col = self.vocab.setdefault(token, len(self.vocab))
                rows.append(row)
                cols.append(col)
                tfs.append(count)

        n_docs, n_vocab = len(self.doc_ids), len(self.vocab)
        rows_np = np.frombuffer(rows, dtype=np.int32)
        cols_np = np.frombuffer(cols, dtype=np.int32)
        tf_np = np.frombuffer(tfs, dtype=np.float32)
        del rows, cols, tfs

        # IDF exactly as rank_bm25.BM25Okapi._calc_idf
        df = np.bincount(cols_np, minlength=n_vocab).astype(np.float64)
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        average_idf = idf.sum() / max(n_vocab, 1)
        idf[idf < 0] = epsilon * average_idf

        avgdl = doc_len.sum() / max(n_docs, 1)
        norm = k1 * (1 - b + b * doc_len / avgdl)
        weights = idf[cols_np] * tf_np * (k1 + 1) / (tf_np + norm[rows_np])
        self.matrix = sparse.csc_matrix(
            (weights.astype(np.float32), (rows_np, cols_np)), shape=(n_docs, n_vocab)
        )

    def _tokenize(self, text: str):
        return self.stemmer.stemWords(text.lower().split())

    def get_scores(self, query_text: str) -> np.ndarray:
        # rank_bm25 sums over query tokens including repeats; unknown tokens add 0
        counts = Counter(t for t in self._tokenize(query_text) if t in self.vocab)
        if not counts:
            return np.zeros(len(self.doc_ids), dtype=np.float64)
        cols = [self.vocab[t] for t in counts]
        mult = np.asarray([counts[t] for t in counts], dtype=np.float32)
        return np.asarray(self.matrix[:, cols] @ mult, dtype=np.float64).ravel()

    def search(self, query_texts, top_k: int):
        if isinstance(query_texts, str):
            scores = self.get_scores(query_texts)
            effective_top_k = min(top_k, len(self.doc_ids))
            top_idx = np.argpartition(-scores, effective_top_k - 1)[:effective_top_k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            return [self.doc_ids[index] for index in top_idx], scores[top_idx]
        return [self.search(query_text, top_k) for query_text in tqdm(query_texts, desc="BM25 retrieval")]


class BM25sSparseRetriever:
    """BM25 as the paper's numbers were produced: bm25s defaults (method="lucene",
    k1=1.5, b=0.75), bm25s regex tokenizer, English stopwords, English Snowball stemmer.

    Checked on FiQA / SciDocs test: Recall@10 0.3178 / 0.1653, nDCG@10 0.2514 / 0.1578,
    P@1 0.2299 / 0.183 vs paper Table 1: 0.3135 / 0.1655, 0.2500 / 0.1578, 0.2299 / 0.1810.
    """

    def __init__(self, corpus: dict, top_k: int, method: str = "lucene", stopwords: str | None = "en") -> None:
        doc_ids = list(corpus.keys())
        texts = [
            (" ".join(part for part in [corpus[d].get("title"), corpus[d].get("text")] if part)).strip()
            for d in doc_ids
        ]
        self._build(doc_ids, texts, top_k, method, stopwords)

    @classmethod
    def from_texts(cls, doc_ids: list[str], texts: list[str], top_k: int, method: str = "lucene",
                   stopwords: str | None = "en") -> "BM25sSparseRetriever":
        """Build without a BEIR corpus dict (saves RAM on 5M-document corpora)."""
        obj = cls.__new__(cls)
        obj._build(doc_ids, texts, top_k, method, stopwords)
        return obj

    def _build(self, doc_ids, texts, top_k, method, stopwords) -> None:
        import bm25s

        self.bm25s = bm25s
        self.top_k = top_k
        self.stopwords = stopwords
        self.stemmer = Stemmer.Stemmer("english")
        self.doc_ids = doc_ids
        tokens = bm25s.tokenize(texts, stopwords=stopwords, stemmer=self.stemmer, show_progress=True)
        self.retriever = bm25s.BM25(method=method)
        self.retriever.index(tokens, show_progress=True)
        del tokens

    def _retrieve(self, query_texts: list[str], top_k: int):
        k = min(top_k, len(self.doc_ids))
        tokens = self.bm25s.tokenize(query_texts, stopwords=self.stopwords, stemmer=self.stemmer,
                                     return_ids=False, show_progress=False)
        # backend_selection="numpy": with JAX installed (Kaggle/Colab) bm25s would otherwise start JAX,
        # which grabs 75% of GPU memory and leaves Contriever out of memory.
        docs, scores = self.retriever.retrieve(tokens, k=k, show_progress=len(query_texts) > 1,
                                               backend_selection="numpy")
        return [([self.doc_ids[i] for i in docs[row]], np.asarray(scores[row], dtype=float)) for row in range(len(query_texts))]

    def search(self, query_texts, top_k: int):
        if isinstance(query_texts, str):
            return self._retrieve([query_texts], top_k)[0]
        return self._retrieve(list(query_texts), top_k)
