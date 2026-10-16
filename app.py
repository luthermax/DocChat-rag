"""Interface Streamlit : charger des PDF, choisir ceux à interroger, discuter.
Lancer depuis la racine du projet :  streamlit run app.py
"""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st

import config
import llm
from embedder import get_embedder
from generate import answer
from ingest import DocumentError
from store import Store

st.set_page_config(page_title="DocChat", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def get_store():
    return Store(get_embedder())


@st.cache_data(ttl=600, show_spinner=False)
def groq_models(api_key):
    return llm.list_groq_models(api_key)


store = get_store()
ss = st.session_state
ss.setdefault("doc_ids", [])        # documents chargés dans CETTE session (personne d'autre ne les voit)
ss.setdefault("messages", [])
ss.setdefault("uploader_key", 0)
ss.doc_ids = [d for d in ss.doc_ids if d in store.docs]

BADGES = {
    "ok": "✅ Réponse appuyée par le document",
    "warn": "⚠️ À vérifier : la réponse n'est pas clairement appuyée par les extraits",
    "none": "ℹ️ Rien de pertinent trouvé dans les documents",
}


# --- Chargement d'un document ---------------------------------------------------------

def load_pdf(name, data):
    """Indexe un PDF et l'ajoute à la session. Retourne un message d'erreur, ou None."""
    if len(ss.doc_ids) >= config.MAX_DOCS:
        return f"Maximum {config.MAX_DOCS} documents en même temps : retire-en un d'abord."
    bar = st.progress(0.0, text=name)
    try:
        rec = store.add_pdf(name, data, progress=lambda f, t: bar.progress(min(f, 1.0), text=f"{name} : {t}"))
    except DocumentError as e:
        return f"{name} : {e}"
    finally:
        bar.empty()
    if rec["id"] not in ss.doc_ids:
        ss.doc_ids.append(rec["id"])
    ss[f"act_{rec['id']}"] = True
    return None


# --- Affichage des messages -------------------------------------------------------------

def render_body(m):
    if m.get("error"):
        st.error(m["error"])
        return
    st.markdown(m["content"])
    meta = m.get("meta")
    if meta:
        st.caption(BADGES[meta["status"]])
        srcs = []
        for b in meta["blocks"]:
            label = f"{b['source']} {b['pages']}"
            if label not in srcs:
                srcs.append(label)
        if srcs:
            st.caption("📄 " + " · ".join(srcs))
        with st.expander("Voir les extraits utilisés"):
            for b in meta["blocks"]:
                st.markdown(f"**{b['source']} — {b['pages']}**")
                st.write(b["text"])
    st.caption(m["time"] + (f" · {m['meta']['seconds']:.0f} s" if m.get("meta") else ""))


def render_message(m):
    with st.chat_message(m["role"]):
        render_body(m)


# --- Barre latérale : modèle de langage -------------------------------------------------

with st.sidebar:
    st.title("📚 DocChat")
    st.caption("Pose des questions à tes documents. Les réponses viennent uniquement de leur contenu.")
    st.subheader("Modèle de langage")
    choice = st.radio("Modèle", ["Ollama (sur ce PC)", "Groq (API en ligne)"],
                      index=0 if config.LLM_BACKEND == "ollama" else 1, label_visibility="collapsed")
    backend = "ollama" if choice.startswith("Ollama") else "groq"
    model, api_key = None, None
    if backend == "ollama":
        model = st.text_input("Modèle Ollama", value=config.OLLAMA_MODEL)
        if llm.ollama_ready():
            st.success("Ollama est prêt")
        else:
            st.error("Ollama ne répond pas : lance l'application Ollama.")
    else:
        api_key = st.text_input("Clé API Groq", type="password", value=os.getenv("GROQ_API_KEY", ""))
        if api_key:
            try:
                models = groq_models(api_key)
            except llm.LLMError as e:
                st.error(str(e))
                models = []
            if models:
                idx = models.index(config.GROQ_MODEL) if config.GROQ_MODEL in models else 0
                model = st.selectbox("Modèle Groq", models, index=idx)
            else:
                model = st.text_input("Modèle Groq", value=config.GROQ_MODEL)
        else:
            st.info("Colle ta clé API Groq (créée sur console.groq.com). Tes documents sont envoyés à Groq.")
    with st.expander("Réglages avancés"):
        k = st.slider("Passages retrouvés par question", 2, 8, config.TOP_K)
        strict = st.checkbox("Mode strict : refuser les réponses non vérifiées", value=config.STRICT_GROUNDING)
    st.divider()
    if st.button("Nouvelle conversation"):
        ss.messages = []
        st.rerun()

# --- Page principale ----------------------------------------------------------------------

st.header("Bonjour 👋")
st.caption("Charge un ou plusieurs PDF, choisis ceux à interroger, puis pose ta question.")
left, right = st.columns([1, 1.7], gap="large")

with left:
    with st.container(border=True):
        st.subheader("1. Ajouter des documents")
        for msg in ss.pop("upload_errors", []):
            st.error(msg)
        uploads = st.file_uploader("PDF avec du texte sélectionnable", type="pdf", accept_multiple_files=True,
                                   key=f"up_{ss.uploader_key}")
        if uploads:
            errors = [e for e in (load_pdf(f.name, f.getvalue()) for f in uploads) if e]
            ss.upload_errors = errors
            ss.uploader_key += 1          # vide le champ de chargement
            st.rerun()
        demo = sorted(glob.glob(os.path.join(config.RAW_DIR, "*.pdf")))
        if demo:
            with st.expander("Documents de démonstration"):
                for path in demo:
                    name = os.path.basename(path)
                    if st.button(f"Ajouter {name}", key=f"demo_{name}"):
                        with open(path, "rb") as fh:
                            err = load_pdf(name, fh.read())
                        if err:
                            ss.upload_errors = [err]
                        st.rerun()

    with st.container(border=True):
        st.subheader("2. Documents à interroger")
        if not ss.doc_ids:
            st.info("Aucun document pour l'instant : charge un PDF ci-dessus.")
        for d in list(ss.doc_ids):
            info = store.info(d)
            c1, c2 = st.columns([6, 1])
            c1.checkbox(f"📄 {info['name']}", key=f"act_{d}")
            c1.caption(f"{info['pages']} pages · {info['n_chunks']} passages")
            if c2.button("✖", key=f"del_{d}", help="Retirer ce document"):
                ss.doc_ids.remove(d)
                ss.pop(f"act_{d}", None)
                st.rerun()
        if ss.doc_ids:
            st.caption("Coche un seul document pour des réponses ciblées, plusieurs pour croiser les sources.")

active = [d for d in ss.doc_ids if ss.get(f"act_{d}")]

with right:
    st.subheader("Discussion")
    chat_box = st.container()
    with chat_box:
        for m in ss.messages:
            render_message(m)

question = st.chat_input("Pose ta question ici...")
if question:
    now = time.strftime("%H:%M")
    ss.messages.append({"role": "user", "content": question, "time": now})
    with chat_box:
        render_message(ss.messages[-1])
        if not active:
            reply = {"role": "assistant", "time": now, "content": "Coche au moins un document avant de poser ta question."}
            render_message(reply)
        else:
            with st.chat_message("assistant"):
                with st.spinner("Je cherche dans les documents..."):
                    t0 = time.time()
                    res = answer(question, store.searcher(active), backend=backend, model=model,
                                 api_key=api_key, k=k, strict=strict)
                    seconds = time.time() - t0
                reply = {"role": "assistant", "time": time.strftime("%H:%M"), "content": res["answer"],
                         "error": res["error"], "meta": {"status": res["status"], "blocks": res["blocks"], "seconds": seconds}}
                render_body(reply)
    ss.messages.append(reply)
