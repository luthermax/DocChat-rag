"""Recherche hybride : le sens (FAISS) + les mots exacts (BM25), fusionnés par RRF."""
import re
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import config

STOP = {"le", "la", "les", "un", "une", "des", "de", "du", "d", "l", "et", "ou",
        "est", "c", "qu", "que", "qui", "quoi", "quand", "pourquoi", "comment",
        "à", "a", "au", "aux", "en", "dans", "sur", "pour", "par", "ce", "ces",
        "cette", "il", "elle", "on", "se", "sa", "son", "ses", "sert", "faire",
        "quel", "quelle", "quelles", "quels", "entre", "y", "s", "n", "ne", "pas"}


def tokenize(text):
    # \w garde les underscores : DUP_VAL_ON_INDEX reste un seul mot
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOP]


class Searcher:
    """Moteur de recherche sur un ensemble de passages (les documents actifs)."""

    def __init__(self, embedder, chunks, embeddings):
        assert len(chunks) == len(embeddings) and len(chunks) > 0
        self.embedder = embedder
        self.chunks = chunks
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def rank(self, question, w_kw=None, pool=20, rrf_k=60):
        """Classement des passages. Retourne (ordre, scores fusionnés, meilleure similarité)."""
        w_kw = config.BM25_WEIGHT if w_kw is None else w_kw
        n = len(self.chunks)
        q = np.asarray(self.embedder.encode_query(question), dtype="float32")[None, :]
        sims, ids = self.index.search(q, min(pool, n))
        sem = [int(i) for i in ids[0] if i >= 0]
        best_sim = float(sims[0][0]) if sem else 0.0

        kw = []
        if w_kw > 0:
            scores = self.bm25.get_scores(tokenize(question))
            kw = [i for i in sorted(range(n), key=lambda i: scores[i], reverse=True)[:pool]
                  if scores[i] > 0]                      # on ignore les passages sans aucun mot en commun

        fused = {}
        for ranking, weight in ((sem, 1.0), (kw, w_kw)):
            for r, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + weight / (rrf_k + r + 1)
        order = sorted(fused, key=fused.get, reverse=True)
        return order, fused, best_sim

    def search(self, question, k=None, w_kw=None, neighbors=None):
        """Les k meilleurs passages + les voisins (avant/après) des `neighbors` meilleurs."""
        k = k or config.TOP_K
        nb = config.NEIGHBOR_TOP if neighbors is None else neighbors
        order, fused, best_sim = self.rank(question, w_kw=w_kw, pool=max(20, k))
        top = order[:k]
        picked = list(top)
        for i in top[:nb]:
            for j in (i - 1, i + 1):
                if (0 <= j < len(self.chunks) and j not in picked
                        and self.chunks[j]["doc_id"] == self.chunks[i]["doc_id"]):
                    picked.append(j)
        passages = [dict(self.chunks[i], score=fused.get(i, 0.0), hit=(i in top)) for i in picked]
        return {"passages": passages, "best_sim": best_sim}
