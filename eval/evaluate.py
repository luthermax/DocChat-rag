"""Mesure la qualité de la recherche sur tes questions (eval/questions.json).

Pour chaque question, on regarde si les mots-clés de la réponse ("contient") se trouvent dans les passages
retrouvés. On teste deux situations : tous les documents actifs, ou seulement le document concerné.
À lancer depuis la racine du projet :  python eval/evaluate.py
"""
import json
import re
import statistics
import sys

sys.path.insert(0, "src")
import config
from embedder import get_embedder
from store import Store

KS = [1, 3, 5, 10]
WEIGHTS = [0, 0.5, 1.0]          # poids de BM25 (0 = recherche par le sens seulement)


def norm(t):
    return re.sub(r"\s+", " ", t.lower().replace("’", "'")).strip()


def found(needles, texts):
    blob = norm(" ".join(texts))
    return all(n in blob for n in needles)


def main():
    with open("eval/questions.json", encoding="utf-8") as f:
        all_q = json.load(f)
    questions = [q for q in all_q if q["contient"]]
    off_topic = [q["question"] for q in all_q if not q["contient"]]

    store = Store(get_embedder())
    print("Indexation de data/raw (longue la première fois, instantanée ensuite)...")
    docs = store.add_folder(config.RAW_DIR)
    if not docs:
        sys.exit(f"Aucun PDF dans {config.RAW_DIR}")
    for d in docs:
        print(f"  {d['name']} : {d['pages']} pages, {d['n_chunks']} passages")

    all_ids = [d["id"] for d in docs]
    s_all = store.searcher(all_ids)

    # document qui contient la réponse (un passage qui a tous les mots-clés)
    truth = {}
    for q in questions:
        needles = [norm(n) for n in q["contient"]]
        truth[q["question"]] = next(
            (d["id"] for d in docs
             if any(all(n in norm(c["text"]) for n in needles) for c in store.docs[d["id"]]["chunks"])), None)
    missing = [q["question"] for q in questions if truth[q["question"]] is None]
    if missing:
        print("\nAttention, mots-clés introuvables dans les PDF (à corriger dans questions.json) :")
        for m in missing:
            print("  -", m)

    modes = [("TOUS LES DOCUMENTS ACTIFS", None), ("LE DOCUMENT CONCERNÉ SEUL", "one")]
    failures = []
    for title, mode in modes:
        print(f"\n=== {title} ===")
        for w in WEIGHTS:
            hits = {k: 0 for k in KS}
            final_ok, ctx_len = 0, []
            for q in questions:
                needles = [norm(n) for n in q["contient"]]
                sr = s_all if mode is None or truth[q["question"]] is None else store.searcher([truth[q["question"]]])
                order, _, _ = sr.rank(q["question"], w_kw=w, pool=max(KS))
                for k in KS:
                    if found(needles, [sr.chunks[i]["text"] for i in order[:k]]):
                        hits[k] += 1
                passages = sr.search(q["question"], w_kw=w)["passages"]
                ok = found(needles, [p["text"] for p in passages])
                final_ok += ok
                ctx_len.append(sum(len(p["text"]) for p in passages))
                if mode is None and w == config.BM25_WEIGHT and not ok:
                    failures.append(q["question"])
            n = len(questions)
            rec = " | ".join(f"@{k}: {hits[k] / n:.0%}" for k in KS)
            print(f"BM25 = {w:<4} | {rec} | contexte envoyé au modèle: {final_ok / n:.0%} "
                  f"(~{int(statistics.mean(ctx_len))} car.)")

    if failures:
        print(f"\nQuestions dont le contexte final ne contient pas la réponse (BM25 = {config.BM25_WEIGHT}) :")
        for q in failures:
            print("  ✗", q)

    # --- aide au réglage de MIN_SIM : similarité du meilleur passage ---
    ins = [s_all.rank(q["question"], w_kw=0, pool=1)[2] for q in questions]
    outs = [(q, s_all.rank(q, w_kw=0, pool=1)[2]) for q in off_topic]
    if ins and outs:
        print("\n=== Réglage de MIN_SIM (similarité du meilleur passage) ===")
        print(f"questions du cours : min {min(ins):.2f} | médiane {statistics.median(ins):.2f}")
        for q, s in outs:
            print(f"hors sujet : {s:.2f} | {q}")
        hi_out = max(s for _, s in outs)
        if hi_out < min(ins):
            print(f"-> séparation nette : MIN_SIM = {(hi_out + min(ins)) / 2:.2f} dans config.py convient.")
        else:
            print("-> les scores se chevauchent : un seuil refuserait aussi de bonnes questions. "
                  "Choisis une valeur prudente (juste sous le minimum des questions du cours) ou laisse 0.")


if __name__ == "__main__":
    main()
