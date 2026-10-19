"""Interface de démonstration Gradio."""
from __future__ import annotations

from .pipeline import RAGPipeline


def build_demo(pipeline: RAGPipeline):
    import gradio as gr

    index = pipeline.load_index()
    doc_types = ["(tous)"] + sorted({c["doc_type"] for c in index.chunks if c.get("doc_type")})
    modes = ["hybrid", "bm25", "dense"] if index.embeddings is not None else ["bm25"]

    def answer_fn(question, doc_type, only_in_force, mode, use_llm):
        if not question.strip():
            return "Pose une question.", ""
        filters = {}
        if doc_type != "(tous)":
            filters["doc_type"] = doc_type
        if only_in_force:
            filters["statut"] = "en_vigueur"
        ans = pipeline.ask(question, use_llm=use_llm, mode=mode, filters=filters or None)
        passages = "\n\n".join(f"**[{h.rank}]** {h.chunk['text']}" for h in ans.hits)
        body = ans.answer or "_Mode sans LLM : passages retrouvés uniquement._"
        return body, f"### Sources\n{ans.sources_markdown()}\n\n### Passages\n{passages}"

    with gr.Blocks(title="RAG-BTP") as demo:
        gr.Markdown("# 🏗️ RAG-BTP — recherche documentaire\nRéponses citées, issues uniquement du corpus indexé.")
        q = gr.Textbox(label="Question", placeholder="Ex. Sous quel délai le DOE doit-il être remis ?")
        with gr.Row():
            dt = gr.Dropdown(doc_types, value="(tous)", label="Type de document")
            mode = gr.Dropdown(modes, value=modes[0], label="Retrieval")
            force = gr.Checkbox(label="Uniquement les versions en vigueur")
            llm = gr.Checkbox(value=True, label="Générer une réponse (LLM)")
        btn = gr.Button("Rechercher", variant="primary")
        out = gr.Markdown()
        src = gr.Markdown()
        btn.click(answer_fn, [q, dt, force, mode, llm], [out, src])
        q.submit(answer_fn, [q, dt, force, mode, llm], [out, src])
    return demo
