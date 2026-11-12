"""Step 8 demo: QuDAR on FiQA, run on a laptop (CPU is enough).

    python app.py                      # http://127.0.0.1:7860
    python app.py --data demo_data --port 7860

Test questions of FiQA work fully offline (expansion from Step 1, LLM scores from Step 4,
answers from Step 7). Your own questions need an LLM for the expansion: OpenAI (gpt-4o-mini)
or Ollama on the laptop (llama3.1:8b, the paper's model family). Keys go in demo/.env.
"""
from __future__ import annotations

import argparse
import html
import os
import random
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from qudar_core import GRID, KEYS, SRC, SRC_LABEL, Engine, live_qid, ranking_metrics, to_axes, words

HERE = Path(__file__).resolve().parent
COLORS = {"OS": "#2563eb", "ES": "#7fb0ff", "OD": "#ea580c", "ED": "#fdb07a"}   # hue = retriever, light = expanded
METHODS = ["OS", "OD", "ES", "ED", "Retriever-Type Fusion", "Query-Format Fusion",
           "QuDAR-simple (RRF)", "QuDAR-simple (Equal)", "QuDAR-confidence", "QuDAR-llm",
           "QuDAR-llm-debiased", "Static-train", "Model A", "Model B", "Thủ công", "Query-wise UB (oracle)"]
STEP7_NAME = {"Static-train": "Static-train (ours)", "Model B": "Model B (ours)"}
EXP_CACHED, EXP_OPENAI, EXP_OLLAMA, EXP_MANUAL = ("Đã lưu (LLaMA 3.1-8B, Step 1)", "OpenAI gpt-4o-mini",
                                                  "Ollama trên laptop", "Tự gõ / sửa tay")
LLM_CACHED, LLM_OPENAI, LLM_OLLAMA, LLM_OFF = ("Đã lưu (gpt-4o-mini, Step 4)", "OpenAI gpt-4o-mini",
                                              "Ollama trên laptop", "Tắt")
RAG_CACHED, RAG_OPENAI, RAG_OLLAMA = ("Đã lưu (Qwen2.5-7B, Step 7)", "OpenAI gpt-4o-mini", "Ollama trên laptop")

CSS = """
.res{border:1px solid var(--border-color-primary);border-radius:10px;padding:10px 12px;margin:0 0 8px 0}
.res .top{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.res .rk{font-weight:700;min-width:1.6em}
.res .ti{font-weight:600}
.res .meta{color:var(--body-text-color-subdued);font-size:.85em}
.res .sn{margin-top:4px;font-size:.92em;line-height:1.45}
.bar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin-top:6px;background:var(--background-fill-secondary)}
.rel{display:inline-block;padding:0 7px;border-radius:9px;font-size:.8em;font-weight:700;background:#16a34a;color:#fff}
.lists{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
.lst{border:1px solid var(--border-color-primary);border-radius:10px;padding:8px 10px}
.lst h4{margin:0 0 6px 0;font-size:.95em}
.lst ol{margin:0;padding-left:1.4em}
.lst li{margin:0 0 5px 0;font-size:.86em;line-height:1.35}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle}
.legend span{margin-right:14px;font-size:.88em}
.note{color:var(--body-text-color-subdued);font-size:.9em}
"""


def load_env():
    f = HERE / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def legend():
    return "<div class='legend'>" + "".join(
        f"<span><i class='sw' style='background:{COLORS[s]}'></i>{s} = {SRC_LABEL[s]}</span>" for s in SRC) + "</div>"


def fmt(x, n=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


class App:
    def __init__(self, data_dir, model_name):
        load_env()
        self.E = Engine(data_dir, model_name=model_name)
        self.choices = [f"{q} — {words(t, 18)}" for q, t in self.E.queries.items()]
        self.step7 = self._load_step7(Path(data_dir))

    def _load_step7(self, data_dir):
        for f in [data_dir / "rag_per_query.csv", HERE.parent / "results" / "step7" / "rag_per_query.csv"]:
            if f.exists():
                df = pd.read_csv(f, dtype={"qid": str})
                df = df[df.name == f"{self.E.manifest['dataset']}_{self.E.manifest['split']}"]
                print(f"Step 7: {df.qid.nunique()} câu hỏi có câu trả lời đã lưu ({f})")
                self.step7_methods = sorted(df.method.unique())
                return {(r.qid, r.method): (r.answer, r.correct) for r in df.itertuples()}
        self.step7_methods = []
        return {}

    # ------------------------------------------------------------ rendering
    def ranks(self, entry):
        return {s: {d: i + 1 for i, d in enumerate(entry[dk])} for s, (dk, _, _) in zip(SRC, KEYS)}

    def render_results(self, st, method):
        E = self.E
        m = st["methods"].get(method)
        if m is None:
            return f"<p class='note'>Phương pháp <b>{html.escape(method)}</b> không dùng được cho câu này " \
                   f"(Model B và QuDAR-llm cần điểm LLM; oracle cần câu hỏi có nhãn).</p>"
        ranked = E.rank(st["entry"], m)
        rel = E.qrels.get(st["qid"], {})
        rk = st["ranks"]
        w = m["w"]
        head = legend()
        if w is not None:
            ax = " · ".join(f"{n} = {v:.2f}" for n, v in zip(("α", "β1", "β2"), to_axes(w)) if v is not None)
            head += (f"<p><b>{html.escape(method)}</b>: " + " · ".join(f"w<sub>{s}</sub> = {v:.3f}" for s, v in zip(SRC, w))
                     + f" &nbsp;→&nbsp; {ax}</p>")
        else:
            head += f"<p><b>{html.escape(method)}</b>: điểm = Σ 1/(60 + hạng) qua 4 danh sách (không dùng điểm số).</p>"
        items = []
        for i, (d, score, contrib) in enumerate(ranked, 1):
            title, text = E.passage(d)
            tot = contrib.sum() or 1
            bar = "".join(f"<span title='{s}: {c:.3f}' style='width:{100 * c / tot:.1f}%;background:{COLORS[s]}'></span>"
                          for s, c in zip(SRC, contrib) if c > 0)
            pos = " · ".join(f"{s} {rk[s].get(d, '—')}" for s in SRC)
            badge = f"<span class='rel'>liên quan (nhãn {rel[d]})</span>" if rel.get(d, 0) > 0 else ""
            items.append(f"<div class='res'><div class='top'><span class='rk'>{i}.</span>"
                         f"<span class='ti'>{html.escape(title or '(không tiêu đề)')}</span>{badge}"
                         f"<span class='meta'>doc {html.escape(d)} · điểm trộn {score:.4f} · hạng trong {pos}</span></div>"
                         f"<div class='sn'>{html.escape(words(text, 70))}</div><div class='bar'>{bar}</div></div>")
        return head + "".join(items)

    def render_lists(self, st):
        E, entry = self.E, st["entry"]
        rel = E.qrels.get(st["qid"], {})
        cols = []
        for s, (dk, rk, nk) in zip(SRC, KEYS):
            lis = []
            for d, raw, nrm in list(zip(entry[dk], entry[rk], entry[nk]))[:10]:
                title, text = E.passage(d)
                mark = " <span class='rel'>✓</span>" if rel.get(d, 0) > 0 else ""
                lis.append(f"<li><b>{html.escape(words(title or text, 9))}</b>{mark}<br>"
                           f"<span class='meta'>doc {html.escape(d)} · thô {raw:.3f} · chuẩn hoá {nrm:.3f}</span></li>")
            n = len(entry[dk])
            cols.append(f"<div class='lst'><h4><i class='sw' style='background:{COLORS[s]}'></i>{s} — {SRC_LABEL[s]}"
                        f" <span class='meta'>({n} kết quả)</span></h4><ol>{''.join(lis)}</ol></div>")
        return "<div class='lists'>" + "".join(cols) + "</div>"

    def table(self, st):
        E = self.E
        rel = E.qrels.get(st["qid"])
        rows = []
        for name in METHODS:
            m = st["methods"].get(name)
            if m is None:
                continue
            ranked = [d for d, _, _ in E.rank(st["entry"], m)]
            met = ranking_metrics(ranked, rel) if rel else None
            w = m["w"]
            a = to_axes(w) if w is not None else (None, None, None)
            rows.append({"Nhóm": m["group"], "Phương pháp": name,
                         **{f"w_{s}": fmt(w[j]) if w is not None else "" for j, s in enumerate(SRC)},
                         "α": fmt(a[0], 2), "β1": fmt(a[1], 2), "β2": fmt(a[2], 2),
                         "Recall@10": fmt(met[0]) if met else "", "nDCG@10": fmt(met[1]) if met else "",
                         "P@1": fmt(met[2], 0) if met else "", "Ghi chú": m["note"]})
        return pd.DataFrame(rows)

    def pipeline_md(self, st):
        E, info, t = self.E, st["info"], st["timings"]
        lines = ["### Quy trình cho câu hỏi này"]
        lines.append(f"**1. Phân tích tài liệu (offline).** {len(E.ids):,} tài liệu FiQA. BM25: tiêu đề + nội dung, "
                     "tách từ bằng regex, bỏ stopword tiếng Anh, stem Snowball. Contriever: `passage: ` + tiêu đề + nội dung, "
                     "cắt 512 wordpiece.")
        vocab = len(E.bm25.vocab_dict)
        lines.append(f"**2. Lập chỉ mục (offline).** Chỉ mục ngược BM25 (bm25s, lucene, k1 = 1.5, b = 0.75): {vocab:,} từ, "
                     f"{E.stats['bm25_index_s']:.1f}s khi mở app. Ma trận dense {E.emb.shape[0]:,} × {E.emb.shape[1]} "
                     f"(fp16, lấy từ chỉ mục FAISS của Step 1).")
        if info.get("mode") == "live":
            lines.append(f"**3. Phân tích câu truy vấn.** Token BM25 của query gốc: `{' '.join(info['q_tokens'])}`; "
                         f"query mở rộng có {len(info['e_tokens'])} token BM25, {info['e_wordpieces']} wordpiece "
                         f"(query gốc {info['q_wordpieces']}). Nguồn query mở rộng: {st['exp_source']}.")
            lines.append("**4. Truy xuất chỉ mục.** 4 danh sách top-1000 (OS, OD, ES, ED).")
        else:
            lines.append(f"**3–4. Phân tích truy vấn và truy xuất:** dùng 4 danh sách top-1000 đã lưu từ Step 2 "
                         f"(tái lập chính xác). Nguồn query mở rộng: {st['exp_source']}.")
        lines.append(f"**5. Xếp hạng.** Chuẩn hoá min-max từng danh sách, gán trọng số theo phương pháp, cộng có trọng số. "
                     f"Điểm LLM: {st['llm_source']}.")
        lines.append("**6. Đánh giá.** " + ("Có nhãn qrels nên bảng so sánh có Recall@10, nDCG@10, P@1 và oracle."
                                         if E.qrels.get(st["qid"]) else "Câu tự nhập không có nhãn nên không có chỉ số."))
        lines = ["\n\n".join(lines)]
        if t:
            lines += ["", "| Công đoạn | Thời gian |", "|---|---:|"] + [f"| {k} | {v * 1000:.0f} ms |" for k, v in t.items()]
            lines.append(f"| **Tổng** | **{sum(t.values()) * 1000:.0f} ms** |")
        if st.get("parity"):
            p = st["parity"]
            lines += ["", "**Kiểm tra khớp Step 2** (số tài liệu top-10 trùng với danh sách đã lưu): "
                      + ", ".join(f"{s} {p[s]}/10" for s in SRC)]
        ch = E.chosen.get(st["qid"])
        if ch:
            chk = []
            for tag in ("A", "B"):
                m = st["methods"].get(f"Model {tag}")
                if m is not None and tag in ch:
                    got = tuple(round(float(x), 1) for x in GRID[m["grid"]])
                    ref = tuple(round(x, 1) for x in ch[tag])
                    chk.append(f"Model {tag} chọn (α, β1, β2) = {got}, Step 6 = {ref} {'✓' if got == ref else '≠'}")
            if chk:
                lines += ["", "**Kiểm tra khớp Step 6:** " + "; ".join(chk)]
        return "\n".join(lines)

    # ------------------------------------------------------------ actions
    def search(self, src, test_choice, free_text, exp_backend, exp_text, llm_backend, use_cached, method, a, b1, b2,
               progress=gr.Progress()):
        E = self.E
        timings = {}
        if src == "Câu hỏi test FiQA (có nhãn)":
            if not test_choice:
                raise gr.Error("Chọn một câu hỏi test.")
            qid = test_choice.split(" — ")[0]
            qtext = E.queries[qid]
        else:
            qtext = (free_text or "").strip()
            if not qtext:
                raise gr.Error("Nhập câu hỏi.")
            qid = next((q for q, t in E.queries.items() if t.strip().lower() == qtext.lower()), None) or live_qid(qtext)
        known = qid in E.queries

        progress(0.1, desc="Mở rộng truy vấn")
        if exp_backend == EXP_CACHED:
            if not known:
                raise gr.Error("Câu tự nhập không có query mở rộng đã lưu. Chọn OpenAI, Ollama hoặc tự gõ.")
            etext, exp_source = E.expansions.get(qid, ""), "LLaMA 3.1-8B (đã lưu, Step 1)"
        elif exp_backend == EXP_MANUAL:
            etext, exp_source = (exp_text or "").strip(), "gõ tay"
        else:
            backend = "openai" if exp_backend == EXP_OPENAI else "ollama"
            model = "gpt-4o-mini" if backend == "openai" else os.environ.get("OLLAMA_EXPAND_MODEL", "llama3.1:8b")
            t = time.time()
            try:
                etext = E.expand(qtext, backend, model)
            except Exception as e:  # noqa: BLE001
                raise gr.Error(f"Không gọi được {exp_backend}: {e}")
            timings[f"Mở rộng truy vấn ({model})"] = time.time() - t
            exp_source = f"{model} (sinh trực tiếp; khác cấu hình của Step 1)"
        if not etext:
            gr.Warning("Query mở rộng rỗng: ES và ED sẽ không có kết quả.")

        progress(0.35, desc="Truy xuất 4 danh sách")
        cached_qid = qid if (use_cached and known and exp_backend == EXP_CACHED) else None
        if use_cached and cached_qid is None:
            gr.Warning("Danh sách đã lưu chỉ dùng được với câu test + query mở rộng đã lưu; đang truy xuất trực tiếp.")
        entry, tr, info = E.retrieve(qtext, etext, use_cached_qid=cached_qid)
        timings.update(tr)
        parity = E.parity(qid, entry) if (info["mode"] == "live" and known and exp_backend == EXP_CACHED) else None

        progress(0.7, desc="Chấm điểm LLM")
        llm_scores, llm_source = None, "tắt"
        if llm_backend == LLM_CACHED:
            if known and qid in E.llm_cache:
                llm_scores, llm_source = E.llm_cache[qid], "gpt-4o-mini (đã lưu, Step 4; chấm trên top-1 của danh sách Step 2)"
            else:
                gr.Warning("Câu này không có điểm LLM đã lưu: QuDAR-llm và Model B bị bỏ.")
        elif llm_backend in (LLM_OPENAI, LLM_OLLAMA):
            backend = "openai" if llm_backend == LLM_OPENAI else "ollama"
            model = "gpt-4o-mini" if backend == "openai" else os.environ.get("OLLAMA_SCORE_MODEL", "llama3.1:8b")
            t = time.time()
            try:
                llm_scores, order = E.llm_score(qid, qtext, entry, backend, model)
                llm_source = f"{model} (gọi trực tiếp, prompt 1 của paper, thứ tự xáo {'/'.join(order)})"
            except Exception as e:  # noqa: BLE001
                gr.Warning(f"Không chấm được bằng LLM: {e}")
            timings[f"Chấm điểm top-1 ({model})"] = time.time() - t

        progress(0.85, desc="Xếp hạng")
        t = time.time()
        methods, feats = E.methods(qid, qtext, etext, entry, llm_scores, (a, b1, b2))
        timings["Tính trọng số mọi phương pháp + trộn"] = time.time() - t
        st = dict(qid=qid, qtext=qtext, etext=etext, entry=entry, methods=methods, timings=timings, info=info,
                  parity=parity, exp_source=exp_source, llm_source=llm_source, ranks=self.ranks(entry))
        head = f"**Câu hỏi** ({'qid ' + qid if known else 'tự nhập'}): {qtext}"
        return (st, head, etext, self.table(st), self.render_results(st, method), self.render_lists(st),
                self.pipeline_md(st))

    def rerank(self, st, method, a, b1, b2):
        if not st:
            return gr.update(), gr.update()
        st["methods"]["Thủ công"]["w"] = np.array([(1 - a) * (1 - b1), a * (1 - b2), (1 - a) * b1, a * b2])
        return self.render_results(st, method), self.table(st)

    def answer(self, st, method, backend_label):
        if not st:
            raise gr.Error("Bấm Tìm kiếm trước.")
        E = self.E
        out = []
        for name in dict.fromkeys(["OS", method]):
            m = st["methods"].get(name)
            if m is None:
                out.append(f"#### {name}\nKhông dùng được cho câu này.")
                continue
            ranked = E.rank(st["entry"], m)
            top5 = ", ".join(d for d, _, _ in ranked[:5])
            n_rel = sum(E.qrels.get(st["qid"], {}).get(d, 0) > 0 for d, _, _ in ranked[:5])
            ctx = f"top-5: {top5}" + (f" ({n_rel} tài liệu liên quan)" if E.qrels.get(st["qid"]) else "")
            if backend_label == RAG_CACHED:
                key = (st["qid"], STEP7_NAME.get(name, name))
                if key not in self.step7:
                    if not any(k[0] == st["qid"] for k in self.step7):
                        why = "câu này không nằm trong 500 câu ngẫu nhiên của Step 7"
                    else:
                        why = "Step 7 chỉ chạy các phương pháp: " + ", ".join(self.step7_methods)
                    out.append(f"#### {name}\nKhông có câu trả lời đã lưu ({why}). Chọn OpenAI hoặc Ollama để sinh trực tiếp.")
                    continue
                ans, ok = self.step7[key]
                verdict = "✅ Qwen chấm **đúng**" if ok == 1 else "❌ Qwen chấm **sai**"
                out.append(f"#### {name}\n{ans}\n\n<span class='note'>{verdict} · {ctx} · Step 7 dùng danh sách đã lưu của Step 2</span>")
            else:
                backend = "openai" if backend_label == RAG_OPENAI else "ollama"
                model = "gpt-4o-mini" if backend == "openai" else os.environ.get("OLLAMA_GEN_MODEL", "qwen2.5:7b")
                t = time.time()
                try:
                    ans = E.chat(backend, model, E.rag_messages(st["qtext"], ranked), 200)
                except Exception as e:  # noqa: BLE001
                    raise gr.Error(f"Không gọi được {backend_label}: {e}")
                out.append(f"#### {name}\n{ans}\n\n<span class='note'>{model} · {time.time() - t:.1f}s · {ctx}</span>")
        return "\n\n".join(out)

    # ------------------------------------------------------------ layout
    def build(self):
        E = self.E
        with gr.Blocks(title="QuDAR demo") as demo:
            st = gr.State(None)
            gr.Markdown(f"# QuDAR — trộn 4 danh sách truy xuất theo từng câu hỏi\n"
                        f"FiQA ({len(E.ids):,} tài liệu, {len(E.queries)} câu hỏi test có nhãn). "
                        "Đồ án CS419 — tái hiện QuDAR (ACL 2026) và mở rộng.")
            with gr.Row():
                with gr.Column(scale=3):
                    src = gr.Radio(["Câu hỏi test FiQA (có nhãn)", "Tự nhập"], value="Câu hỏi test FiQA (có nhãn)",
                                   label="Nguồn câu hỏi")
                    with gr.Row():
                        test_q = gr.Dropdown(self.choices, value=self.choices[0], label="Câu hỏi test", scale=5,
                                             filterable=True)
                        rnd = gr.Button("🎲 Ngẫu nhiên", scale=1, min_width=90)
                    free = gr.Textbox(label="Câu hỏi tự nhập", placeholder="vd. Is it better to pay off my mortgage early?",
                                      visible=False)
                    exp_text = gr.Textbox(label="Query mở rộng (được điền sau khi tìm; sửa được khi chọn 'Tự gõ / sửa tay')",
                                          lines=3)
                with gr.Column(scale=2):
                    exp_b = gr.Dropdown([EXP_CACHED, EXP_OPENAI, EXP_OLLAMA, EXP_MANUAL], value=EXP_CACHED,
                                        label="Mở rộng truy vấn")
                    llm_b = gr.Dropdown([LLM_CACHED, LLM_OPENAI, LLM_OLLAMA, LLM_OFF], value=LLM_CACHED,
                                        label="Chấm điểm LLM (cho QuDAR-llm, Model B)")
                    cached = gr.Checkbox(False, label="Dùng 4 danh sách đã lưu từ Step 2 (tái lập chính xác bảng kết quả)")
                    method = gr.Dropdown(METHODS, value="Model A", label="Phương pháp cho kết quả cuối")
                    with gr.Accordion("Trọng số thủ công (phương pháp 'Thủ công')", open=False):
                        a = gr.Slider(0, 1, 0.5, step=0.1, label="α: sparse (0) ↔ dense (1)")
                        b1 = gr.Slider(0, 1, 0.5, step=0.1, label="β1: trong BM25, gốc (0) ↔ mở rộng (1)")
                        b2 = gr.Slider(0, 1, 0.5, step=0.1, label="β2: trong Contriever, gốc (0) ↔ mở rộng (1)")
                    go = gr.Button("🔎 Tìm kiếm", variant="primary")
            head = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("Kết quả cuối"):
                    res = gr.HTML()
                with gr.Tab("So sánh phương pháp"):
                    gr.Markdown("<span class='note'>Trọng số mỗi phương pháp gán cho câu này, và chỉ số của top-10 "
                                "(chỉ có với câu test). α = phần dense, β1/β2 = phần query mở rộng trong BM25/Contriever.</span>")
                    tbl = gr.Dataframe(wrap=True, max_height=2000,
                                       column_widths=["9%", "13%"] + ["5%"] * 7 + ["7%", "7%", "4%", "23%"])
                with gr.Tab("4 danh sách"):
                    lists = gr.HTML()
                with gr.Tab("Quy trình & thời gian"):
                    pipe = gr.Markdown()
                with gr.Tab("Sinh câu trả lời (RAG)"):
                    gr.Markdown("<span class='note'>So sánh câu trả lời khi đưa top-5 của OS và của phương pháp đang chọn "
                                "cho LLM (cùng prompt với Step 7).</span>")
                    with gr.Row():
                        rag_b = gr.Dropdown([RAG_CACHED, RAG_OPENAI, RAG_OLLAMA], value=RAG_CACHED, label="Mô hình", scale=3)
                        rag_go = gr.Button("Sinh câu trả lời", scale=1)
                    rag = gr.Markdown()

            src.change(lambda s: (gr.update(visible=s != "Tự nhập"), gr.update(visible=s != "Tự nhập"),
                                  gr.update(visible=s == "Tự nhập"),
                                  gr.update(value=EXP_CACHED if s != "Tự nhập" else EXP_OPENAI),
                                  gr.update(value=LLM_CACHED if s != "Tự nhập" else LLM_OFF),
                                  gr.update(value=RAG_CACHED if s != "Tự nhập" else RAG_OPENAI)),
                       src, [test_q, rnd, free, exp_b, llm_b, rag_b])
            rnd.click(lambda: random.choice(self.choices), None, test_q)
            ins = [src, test_q, free, exp_b, exp_text, llm_b, cached, method, a, b1, b2]
            outs = [st, head, exp_text, tbl, res, lists, pipe]
            go.click(self.search, ins, outs)
            free.submit(self.search, ins, outs)
            for c in (method,):
                c.change(self.rerank, [st, method, a, b1, b2], [res, tbl])
            for s in (a, b1, b2):
                s.release(self.rerank, [st, method, a, b1, b2], [res, tbl])
            rag_go.click(self.answer, [st, method, rag_b], rag)
        return demo


def main():
    try:
        import sys
        sys.stdout.reconfigure(errors="replace")    # PowerShell consoles that are not UTF-8
    except Exception:  # noqa: BLE001
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(HERE / "demo_data"))
    p.add_argument("--model", default=os.environ.get("CONTRIEVER", "facebook/contriever"))
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="tạo link công khai tạm thời (72 giờ) để người khác mở")
    args = p.parse_args()
    app = App(args.data, args.model)
    print("Nạp Contriever (lần đầu tải khoảng 440 MB từ Hugging Face)...")
    app.E._encoder()
    app.build().launch(server_port=args.port, share=args.share, theme=gr.themes.Soft(), css=CSS, inbrowser=True)


if __name__ == "__main__":
    main()
