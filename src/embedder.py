"""Transforme du texte en vecteurs. Un vecteur = le sens d'un passage."""
import os
import re
import zlib
import numpy as np
import config


class SentenceEmbedder:
    """Vrai modèle d'embeddings (sentence-transformers), chargé au premier usage."""

    def __init__(self):
        self.name = config.EMBED_MODEL
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(config.EMBED_MODEL)
        return self._model

    @property
    def dim(self):
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts, progress=None, batch_size=32):
        """Retourne un tableau (n, dim) de vecteurs normalisés (produit scalaire = cosinus)."""
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        parts = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            parts.append(self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False))
            if progress:
                progress(min(i + batch_size, len(texts)) / len(texts))
        return np.vstack(parts).astype("float32")

    def encode_query(self, text):
        return self.encode([text])[0]


class HashEmbedder:
    """Faux embedder (comptage de mots hachés) : sert uniquement aux tests, sans télécharger de modèle."""
    name = "hash-384"
    dim = 384

    def encode(self, texts, progress=None, batch_size=32):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for r, t in enumerate(texts):
            for tok in re.findall(r"\w+", t.lower()):
                out[r, zlib.crc32(tok.encode()) % self.dim] += 1.0
            n = np.linalg.norm(out[r])
            if n > 0:
                out[r] /= n
            if progress and (r + 1) % batch_size == 0:
                progress((r + 1) / len(texts))
        if progress:
            progress(1.0)
        return out

    def encode_query(self, text):
        return self.encode([text])[0]


def get_embedder():
    if os.getenv("RAG_EMBEDDER") == "hash":      # mode test uniquement
        return HashEmbedder()
    return SentenceEmbedder()
