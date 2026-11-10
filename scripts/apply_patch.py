"""Point qudar's 'bm25' sparse retriever at the bm25s setup that matches the paper.

Run once per fresh clone:  python apply_patch.py /path/to/QuDAR   (needs: pip install bm25s)
"""
import re
import shutil
import sys
from pathlib import Path

repo = Path(sys.argv[1])
pkg = repo / "src" / "qudar"
shutil.copy(Path(__file__).with_name("bm25_fast.py"), pkg / "bm25_fast.py")
retrievers = pkg / "retrievers.py"
src = retrievers.read_text()
new = ("        from .bm25_fast import BM25sSparseRetriever\n"
       "        return BM25sSparseRetriever(corpus=corpus, top_k=top_k)")
pattern = re.compile(
    r"        (?:from \.bm25_fast import \w+\n        return \w+|return BM25SparseRetriever)\(corpus=corpus, top_k=top_k\)"
)
if new in src:
    print("already patched (bm25s)")
elif pattern.search(src):
    retrievers.write_text(pattern.sub(new, src, count=1))
    print("patched", retrievers, "-> BM25sSparseRetriever")
else:
    raise SystemExit("retrievers.py changed upstream: patch target not found")
