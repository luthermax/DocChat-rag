"""Les documents chargés : lecture, découpage, vecteurs, et sauvegarde sur disque (cache)."""
import glob
import hashlib
import io
import json
import os
import threading
import numpy as np
import config
from ingest import extract_pages, build_chunks
from retrieve import Searcher

VERSION = 2   # à incrémenter quand on change la façon de découper : les caches se refont tout seuls


def _signature(embedder):
    keys = (VERSION, config.CHUNK_SIZE, config.CHUNK_OVERLAP, config.MIN_CHUNK,
            config.X_TOLERANCE, embedder.name)
    return hashlib.sha1(repr(keys).encode()).hexdigest()[:10]


class Store:
    def __init__(self, embedder, cache_dir=None):
        self.embedder = embedder
        self.cache_dir = cache_dir or config.CACHE_DIR
        self.docs = {}              # doc_id -> {"name", "pages", "chunks", "embs"}
        self._searchers = {}
        self._lock = threading.RLock()

    # -- documents -------------------------------------------------------------
    def info(self, doc_id):
        d = self.docs[doc_id]
        return {"id": doc_id, "name": d["name"], "pages": d["pages"], "n_chunks": len(d["chunks"])}

    def add_pdf(self, name, data, progress=None):
        """Ajoute un PDF (octets). Si le même fichier a déjà été indexé, on réutilise le cache.
        progress(fraction, texte) permet d'afficher une barre de progression."""
        progress = progress or (lambda f, t: None)
        doc_id = hashlib.sha1(data).hexdigest()[:12]
        with self._lock:
            if doc_id in self.docs:
                return self.info(doc_id)
        sig = _signature(self.embedder)
        folder = os.path.join(self.cache_dir, doc_id)
        meta_path = os.path.join(folder, "meta.json")

        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("signature") == sig:
                    with open(os.path.join(folder, "chunks.json"), encoding="utf-8") as f:
                        chunks = json.load(f)
                    embs = np.load(os.path.join(folder, "embeddings.npy"))
                    if len(chunks) == len(embs):
                        progress(1.0, "Document déjà indexé")
                        return self._register(doc_id, meta["name"], meta["pages"], chunks, embs)
            except Exception:
                pass                                   # cache abîmé : on refait

        progress(0.0, "Lecture du PDF...")
        pages = extract_pages(io.BytesIO(data), progress=lambda f: progress(0.4 * f, "Lecture du PDF..."))
        chunks = build_chunks(pages, name, doc_id)
        progress(0.4, "Calcul des vecteurs...")
        embs = self.embedder.encode([c["text"] for c in chunks],
                                    progress=lambda f: progress(0.4 + 0.6 * f, "Calcul des vecteurs..."))

        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        np.save(os.path.join(folder, "embeddings.npy"), embs)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"name": name, "pages": len(pages), "signature": sig}, f)
        progress(1.0, "Terminé")
        return self._register(doc_id, name, len(pages), chunks, embs)

    def _register(self, doc_id, name, n_pages, chunks, embs):
        with self._lock:
            self.docs[doc_id] = {"name": name, "pages": n_pages, "chunks": chunks, "embs": embs}
            self._searchers.clear()
            return self.info(doc_id)

    def add_folder(self, folder, progress=None):
        """Ajoute tous les PDF d'un dossier (démonstration, évaluation)."""
        records = []
        for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
            with open(path, "rb") as f:
                records.append(self.add_pdf(os.path.basename(path), f.read(), progress))
        return records

    # -- recherche -------------------------------------------------------------
    def searcher(self, doc_ids):
        """Moteur de recherche limité aux documents demandés."""
        key = tuple(sorted(doc_ids))
        if not key:
            raise ValueError("Aucun document actif.")
        with self._lock:
            if key not in self._searchers:
                if len(self._searchers) > 20:
                    self._searchers.clear()
                chunks, embs = [], []
                for d in key:
                    chunks += self.docs[d]["chunks"]
                    embs.append(self.docs[d]["embs"])
                self._searchers[key] = Searcher(self.embedder, chunks, np.vstack(embs))
            return self._searchers[key]
