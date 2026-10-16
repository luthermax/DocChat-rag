"""Lecture d'un PDF et découpage en passages (chunks)."""
import re
from collections import Counter
import pdfplumber
import config


class DocumentError(Exception):
    """Document inutilisable (illisible, scanné, trop long...). Le message est affiché à l'utilisateur."""


# --- Extraction ---------------------------------------------------------------

def extract_pages(source, progress=None):
    """source : chemin ou fichier ouvert. Retourne [(numéro de page, texte), ...]."""
    try:
        with pdfplumber.open(source) as pdf:
            n = len(pdf.pages)
            if n > config.MAX_PAGES:
                raise DocumentError(f"Document trop long : {n} pages (maximum {config.MAX_PAGES}).")
            pages = []
            for i, page in enumerate(pdf.pages):
                pages.append((i + 1, page.extract_text(x_tolerance=config.X_TOLERANCE) or ""))
                if progress:
                    progress((i + 1) / n)
            return pages
    except DocumentError:
        raise
    except Exception as e:
        raise DocumentError(f"Fichier PDF illisible ({type(e).__name__}).") from e


# --- Nettoyage ----------------------------------------------------------------

def _norm_line(line):
    # "7/59" et "12/59" deviennent identiques une fois les chiffres neutralisés
    return re.sub(r"\d+", "0", line.strip())


def repeated_lines(pages, ratio=0.4, min_pages=8):
    """Lignes présentes sur au moins 40 % des pages : en-têtes, pieds de page, numéros de page."""
    if len(pages) < min_pages:
        return set()
    counts = Counter()
    for _, text in pages:
        counts.update({_norm_line(l) for l in text.split("\n") if l.strip()})
    return {l for l, c in counts.items() if c >= ratio * len(pages)}


def strip_repeated(text, repeated):
    return "\n".join(l for l in text.split("\n") if _norm_line(l) not in repeated)


def is_toc(text):
    """Table des matières : beaucoup de séquences '. . . . .'"""
    return len(re.findall(r"(?:\. ){5,}", text)) >= 3


# --- Découpage ----------------------------------------------------------------

# Un titre : "Chapitre 3 ...", "2.1.3 Les types d'exceptions", "1 Les curseurs"
HEADING = re.compile(
    r"^(?:(?i:chapitre|partie|section)\s+\w+.*"
    r"|\d+(?:\.\d+){0,3}\.?\s+[A-ZÀ-ÖØ-Þ].{2,80})$"
)


def is_heading(line):
    s = line.strip()
    return bool(s) and len(s) <= 90 and not s.endswith((".", ";", ",")) and bool(HEADING.match(s))


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


def split_long(text, size, overlap):
    """Coupe un texte trop long entre les phrases, avec un chevauchement."""
    sentences = re.split(r"(?<=[.!?:;])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= size:
            current = f"{current} {s}".strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:]
            tail = tail[tail.find(" ") + 1:]          # ne pas commencer au milieu d'un mot
            current = f"{tail} {s}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_page(text, size=None, overlap=None):
    """Découpe une page en repartant à chaque titre : une section courte reste d'un seul tenant.

    Les sections sont regroupées tant que le passage ne dépasse pas `size`.
    Une section plus longue que `size` est coupée entre les phrases.
    """
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    sections, current = [], []
    for line in text.split("\n"):
        if is_heading(line) and current:
            sections.append(current)
            current = []
        current.append(line)
    if current:
        sections.append(current)
    units = [u for u in (_clean(" ".join(s)) for s in sections) if u]

    chunks, buf = [], ""
    for u in units:
        if len(u) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(split_long(u, size, overlap))
        elif not buf:
            buf = u
        elif len(buf) + 1 + len(u) <= size:
            buf += " " + u
        else:
            chunks.append(buf)
            buf = u
    if buf:
        chunks.append(buf)
    return chunks


def build_chunks(pages, source, doc_id):
    """pages -> liste de passages {doc_id, source, page, idx, text}. idx = position dans le document."""
    total = sum(len(t.strip()) for _, t in pages)
    if total < 200:
        raise DocumentError("Aucun texte trouvé : ce PDF est probablement un scan (image). "
                            "Utilise un PDF dont le texte est sélectionnable.")
    repeated = repeated_lines(pages)
    chunks = []
    for page_num, text in pages:
        text = strip_repeated(text, repeated)
        if is_toc(text):
            continue
        for c in chunk_page(text):
            if len(c) >= config.MIN_CHUNK:
                chunks.append({"doc_id": doc_id, "source": source, "page": page_num,
                               "idx": len(chunks), "text": c})
    if not chunks:
        raise DocumentError("Aucun passage exploitable dans ce PDF.")
    return chunks
