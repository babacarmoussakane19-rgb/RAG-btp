"""Étape 7 : évaluation du retrieval (hit@k, MRR) et contrôle de bout en bout (proxy).

Format de ``eval/questions.jsonl`` (une question par ligne) :
  {"id": "q01", "question": "...", "answerable": true,
   "gold_doc": "process_interne/procedure_reception_chantier",   # doc_id attendu
   "gold_snippet": "10 jours ouvrés",                            # texte qui doit figurer dans le chunk
   "expected_keyword": "10 jours",                               # attendu dans la réponse générée
   "filters": {"statut": "en_vigueur"}}                          # optionnel
Les questions ``answerable: false`` servent à tester l'abstention (« je ne sais pas »).
"""
from __future__ import annotations

import json
from pathlib import Path

from .generation import REFUSAL
from .textutils import normalize

KS = (1, 3, 5)


def load_questions(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def is_hit(chunk: dict, q: dict) -> bool:
    if q.get("gold_doc") and chunk["doc_id"] != q["gold_doc"]:
        return False
    return normalize(q["gold_snippet"]) in normalize(chunk["text"])


def evaluate_retrieval(pipeline, questions: list[dict], mode: str, rerank: bool | None = None) -> dict:
    answerable = [q for q in questions if q.get("answerable", True)]
    ranks: list[int | None] = []
    failures: list[dict] = []
    for q in answerable:
        hits = pipeline.retrieve(q["question"], top_k=max(KS), mode=mode, filters=q.get("filters"), rerank=rerank)
        rank = next((h.rank for h in hits if is_hit(h.chunk, q)), None)
        ranks.append(rank)
        if rank is None:
            failures.append({"id": q["id"], "question": q["question"]})
    n = len(answerable)
    metrics = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    metrics["mrr"] = sum(1 / r for r in ranks if r) / n
    return {"mode": mode, "n_questions": n, "metrics": metrics, "failures": failures}


def evaluate_generation(pipeline, questions: list[dict], mode: str, rerank: bool | None = None) -> dict:
    """Proxy automatique : mot-clé attendu présent (répondables) / refus explicite (non répondables).

    À compléter par une relecture manuelle ou un juge LLM (voir docs/evaluation.md).
    """
    ok_answerable = n_answerable = ok_abstain = n_unanswerable = 0
    details = []
    for q in questions:
        ans = pipeline.ask(q["question"], mode=mode, filters=q.get("filters"), rerank=rerank)
        text = ans.answer or ""
        if q.get("answerable", True):
            n_answerable += 1
            good = normalize(q["expected_keyword"]) in normalize(text)
            ok_answerable += good
        else:
            n_unanswerable += 1
            good = normalize(REFUSAL)[:30] in normalize(text)
            ok_abstain += good
        details.append({"id": q["id"], "ok": bool(good), "answer": text})
    return {
        "mode": mode,
        "answer_keyword_rate": ok_answerable / max(n_answerable, 1),
        "abstention_rate": ok_abstain / max(n_unanswerable, 1),
        "details": details,
    }


def retrieval_report_markdown(results: list[dict]) -> str:
    lines = [
        "| Mode | " + " | ".join(f"hit@{k}" for k in KS) + " | MRR |",
        "|---|" + "---|" * (len(KS) + 1),
    ]
    for r in results:
        m = r["metrics"]
        lines.append(f"| {r['mode']} | " + " | ".join(f"{m[f'hit@{k}']:.2f}" for k in KS) + f" | {m['mrr']:.2f} |")
    n = results[0]["n_questions"] if results else 0
    return f"Questions répondables évaluées : {n}\n\n" + "\n".join(lines)


def save_results(results: dict, out_dir: str | Path = "eval/results") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "latest.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
