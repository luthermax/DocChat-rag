"""De la question à la réponse : recherche -> contexte -> LLM -> contrôle de la réponse."""
import difflib
import re
import config
import llm
from retrieve import STOP

REFUS = "Je ne trouve pas ça dans les documents."

SYSTEM = """Tu es un assistant qui répond uniquement à partir d'extraits de documents.
Règles :
1. Utilise seulement les extraits fournis. N'utilise jamais tes connaissances personnelles.
2. Reprends les termes exacts du document. N'invente aucune catégorie ni aucun nom.
3. Si les extraits contiennent plusieurs listes ou classifications différentes, présente-les séparément sans les mélanger.
4. Si la réponse n'est pas dans les extraits, mets AUCUNE comme citation et réponds : Je ne trouve pas ça dans les documents.

Format de sortie, exactement deux parties :
CITATION: une phrase copiée mot pour mot depuis les extraits (ou AUCUNE)
REPONSE: ta réponse en français, claire et courte (une liste courte est permise)"""


# --- Construction du contexte ---------------------------------------------------

def _merge(a, b):
    """Colle deux passages consécutifs en retirant le chevauchement."""
    for n in range(min(len(a), len(b), 250), 24, -1):
        if a.endswith(b[:n]):
            return a + b[n:]
    return a + " " + b


def _pages_label(pages):
    pages = sorted(pages)
    return f"p.{pages[0]}" if len(pages) == 1 else f"p.{pages[0]}-{pages[-1]}"


def build_context(passages):
    """Regroupe les passages consécutifs d'un même document en blocs lisibles.
    Retourne (texte envoyé au modèle, blocs pour l'affichage)."""
    blocks = []
    for p in sorted(passages, key=lambda p: (p["doc_id"], p["idx"])):
        last = blocks[-1] if blocks else None
        if last and last["doc_id"] == p["doc_id"] and p["idx"] == last["last_idx"] + 1:
            last["text"] = _merge(last["text"], p["text"])
            last["last_idx"] = p["idx"]
            last["pages"].add(p["page"])
            last["score"] = max(last["score"], p["score"])
        else:
            blocks.append({"doc_id": p["doc_id"], "source": p["source"], "text": p["text"],
                           "last_idx": p["idx"], "pages": {p["page"]}, "score": p["score"]})
    blocks.sort(key=lambda b: b["score"], reverse=True)          # le plus pertinent en premier
    out = [{"source": b["source"], "pages": _pages_label(b["pages"]), "text": b["text"]} for b in blocks]
    context = "\n\n".join(f"[{b['source']}, {b['pages']}]\n{b['text']}" for b in out)
    return context, out


# --- Lecture et contrôle de la sortie du modèle -----------------------------------

def parse_output(raw):
    """Sépare CITATION et REPONSE. Si le format n'est pas respecté, tout est pris comme réponse."""
    m = re.search(r"(?is)citation\s*:\s*(?P<cit>.*?)\s*r[ée]ponse\s*:\s*(?P<rep>.*)$", raw)
    if m:
        return m.group("cit").strip().strip('«»"“” '), m.group("rep").strip()
    m = re.search(r"(?is)r[ée]ponse\s*:\s*(.*)$", raw)
    if m:
        return None, m.group(1).strip()
    return None, raw.strip()


def _norm(t):
    return re.sub(r"\s+", " ", t.lower().replace("’", "'")).strip()


def citation_found(citation, context):
    """La citation existe-t-elle (presque) mot pour mot dans les extraits ?"""
    c, ctx = _norm(citation), _norm(context)
    if len(c) < 15:
        return False
    sm = difflib.SequenceMatcher(None, ctx, c, autojunk=False)
    return sm.find_longest_match(0, len(ctx), 0, len(c)).size / len(c) >= 0.75


def overlap(answer, context):
    """Part des mots importants de la réponse qui figurent dans les extraits."""
    words = [w for w in re.findall(r"\w{4,}", _norm(answer)) if w not in STOP]
    if not words:
        return 1.0
    ctx_words = set(re.findall(r"\w+", _norm(context)))
    return sum(w in ctx_words for w in words) / len(words)


# --- Point d'entrée ---------------------------------------------------------------

def _result(answer, status, **kw):
    base = {"answer": answer, "status": status, "citation": None, "citation_found": None,
            "overlap": None, "blocks": [], "passages": [], "best_sim": 0.0, "raw": "", "error": None}
    base.update(kw)
    return base


def answer(question, searcher, backend=None, model=None, api_key=None, k=None, strict=None):
    """Répond à une question à partir des documents du `searcher`.

    status : "ok" (appuyée par les extraits), "warn" (à vérifier), "none" (rien trouvé), "error".
    """
    strict = config.STRICT_GROUNDING if strict is None else strict
    res = searcher.search(question, k=k)
    best_sim, passages = res["best_sim"], res["passages"]
    context, blocks = build_context(passages)

    if config.MIN_SIM and best_sim < config.MIN_SIM:             # rien de proche : inutile d'appeler le LLM
        return _result(REFUS, "none", blocks=blocks, passages=passages, best_sim=best_sim)

    user = (f"Extraits :\n{context}\n\nQuestion : {question}\n\n"
            "Réponds au format demandé (CITATION puis REPONSE), uniquement à partir des extraits.")
    try:
        raw = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                       backend=backend, model=model, api_key=api_key)
    except llm.LLMError as e:
        return _result("", "error", error=str(e), blocks=blocks, passages=passages, best_sim=best_sim)

    citation, text = parse_output(raw)
    common = dict(blocks=blocks, passages=passages, best_sim=best_sim, raw=raw, citation=citation)
    plain_context = "\n".join(b["text"] for b in blocks)

    if (citation and citation.strip().upper().startswith("AUCUNE")) or "ne trouve pas ça" in text.lower():
        return _result(REFUS, "none", **common)

    found = citation_found(citation, plain_context) if citation else None
    ov = overlap(text, plain_context)
    ok = bool(found) and ov >= 0.5
    if not ok and strict:
        return _result("Je ne peux pas confirmer cette réponse à partir des documents.", "none",
                       citation_found=found, overlap=ov, **common)
    return _result(text, "ok" if ok else "warn", citation_found=found, overlap=ov, **common)
