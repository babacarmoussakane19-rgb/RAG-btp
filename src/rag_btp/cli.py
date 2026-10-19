"""Interface en ligne de commande : build | ask | eval | app."""
from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import RAGPipeline


def _filters(args) -> dict | None:
    f = {}
    if getattr(args, "doc_type", None):
        f["doc_type"] = args.doc_type
    if getattr(args, "statut", None):
        f["statut"] = args.statut
    return f or None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rag-btp", description="RAG de recherche documentaire BTP")
    p.add_argument("--config", default=None, help="chemin du fichier YAML (défaut : configs/config.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="parse, découpe, vectorise et indexe data/raw")
    b.add_argument("--no-embed", action="store_true", help="BM25 seul (pas de modèle d'embeddings)")

    a = sub.add_parser("ask", help="poser une question")
    a.add_argument("question")
    a.add_argument("--mode", choices=["bm25", "dense", "hybrid"])
    a.add_argument("-k", type=int, dest="top_k")
    a.add_argument("--doc-type")
    a.add_argument("--statut", help="ex. en_vigueur")
    a.add_argument("--rerank", action="store_true", default=None)
    a.add_argument("--no-llm", action="store_true", help="afficher seulement les passages retrouvés")

    e = sub.add_parser("eval", help="évaluer retrieval (et génération avec --with-llm)")
    e.add_argument("--modes", default="bm25,dense,hybrid")
    e.add_argument("--questions", default="eval/questions.jsonl")
    e.add_argument("--rerank", action="store_true", default=None)
    e.add_argument("--with-llm", action="store_true")

    g = sub.add_parser("app", help="lancer l'interface Gradio")
    g.add_argument("--share", action="store_true", help="lien public temporaire (Colab)")

    args = p.parse_args(argv)
    pipe = RAGPipeline(load_config(args.config))

    if args.cmd == "build":
        pipe.build_index(embed=not args.no_embed)

    elif args.cmd == "ask":
        ans = pipe.ask(
            args.question,
            use_llm=not args.no_llm,
            mode=args.mode,
            top_k=args.top_k,
            filters=_filters(args),
            rerank=args.rerank,
        )
        if ans.answer:
            print(ans.answer, "\n")
        print("Sources :")
        print(ans.sources_markdown())
        if args.no_llm:
            for h in ans.hits:
                print(f"\n--- [{h.rank}] score={h.score:.3f}\n{h.chunk['text']}")

    elif args.cmd == "eval":
        from . import evaluation as ev

        questions = ev.load_questions(args.questions)
        modes = [m for m in args.modes.split(",") if m]
        if pipe.load_index().embeddings is None:
            modes = [m for m in modes if m == "bm25"]
            print("Index sans embeddings : seul le mode bm25 est évalué.")
        retrieval = [ev.evaluate_retrieval(pipe, questions, m, args.rerank) for m in modes]
        print(ev.retrieval_report_markdown(retrieval))
        out = {"retrieval": retrieval}
        if args.with_llm:
            out["generation"] = [ev.evaluate_generation(pipe, questions, m, args.rerank) for m in modes]
            for g_ in out["generation"]:
                print(
                    f"[{g_['mode']}] réponses contenant la valeur attendue : {g_['answer_keyword_rate']:.2f} | "
                    f"abstention correcte : {g_['abstention_rate']:.2f}"
                )
        print("Résultats détaillés :", ev.save_results(out))

    elif args.cmd == "app":
        from .app import build_demo

        build_demo(pipe).launch(share=args.share)


if __name__ == "__main__":
    main()
