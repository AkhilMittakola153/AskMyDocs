import re

import streamlit as st
from docx import Document
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from llm import MODEL, LLMError, chat

CHUNK_WORDS, OVERLAP, TOP_K = 250, 50, 5


def read_file(f):
    """Return a list of (filename, page_or_None, text)."""
    ext = f.name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        reader = PdfReader(f)
        return [(f.name, f"p.{i+1}", p.extract_text() or "") for i, p in enumerate(reader.pages)]
    if ext == "docx":
        doc = Document(f)
        lines = [p.text for p in doc.paragraphs]
        for table in doc.tables:                      # text inside tables was being skipped
            for row in table.rows:
                lines.append(" | ".join(c.text for c in row.cells))
        return [(f.name, None, "\n".join(lines))]
    return [(f.name, None, f.read().decode("utf-8", errors="ignore"))]


def chunk(name, loc, text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP
    out = []
    for k, i in enumerate(range(0, max(len(words), 1), step)):
        piece = " ".join(words[i:i + CHUNK_WORDS])
        if piece.strip():
            out.append({"file": name, "loc": loc or f"part {k+1}", "text": piece})
    return out


def tokenize(s):
    return re.findall(r"\w+", s.lower())


def build_index(files):
    chunks, empty = [], []
    for f in files:
        got = 0
        for name, loc, text in read_file(f):
            c = chunk(name, loc, text)
            got += len(c)
            chunks += c
        if got == 0:
            empty.append(f.name)
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None
    return chunks, bm25, empty


def retrieve(q, chunks, bm25):
    q_tokens = set(tokenize(q))
    scores = bm25.get_scores(list(q_tokens))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    # Keep chunks sharing at least one word with the question. (Filtering on score > 0 fails for
    # small files: with only 1-2 chunks BM25's IDF is zero or negative, so every score is <= 0.)
    hits = [chunks[i] for i in ranked if q_tokens & set(tokenize(chunks[i]["text"]))]
    return hits[:TOP_K]


def answer(q, hits):
    context = "\n\n".join(f"[{h['file']}, {h['loc']}]\n{h['text']}" for h in hits)
    system = ("Answer the question using ONLY the sources below. Cite each fact as "
              "[filename, location]. If the sources do not contain the answer, "
              "say 'Not found in the uploaded files.' Do not guess.\n\n" + context)
    return chat(system, q)


st.title("Ask My Docs")
st.caption(f"Model: `{MODEL}` (falls back to `openrouter/free` if it fails)")

files = st.file_uploader("Upload PDF, DOCX or TXT files", type=["pdf", "docx", "txt"],
                         accept_multiple_files=True)

if st.button("Index files") and files:
    chunks, bm25, empty = build_index(files)
    st.session_state.update(chunks=chunks, bm25=bm25, indexed=sorted(f.name for f in files), answers={})
    st.success(f"Indexed {len(files)} files into {len(chunks)} chunks.")
    if empty:
        st.warning(f"No text found in: {', '.join(empty)} (scanned PDF?)")

if files and st.session_state.get("indexed") and sorted(f.name for f in files) != st.session_state.indexed:
    st.info("Files changed. Click **Index files** again.")

q = st.text_input("Ask a question")
if q and st.session_state.get("bm25"):
    hits = retrieve(q, st.session_state.chunks, st.session_state.bm25)
    if not hits:
        st.info("No matching passages. Try different keywords.")
    else:
        cache = st.session_state.setdefault("answers", {})
        if q not in cache:                            # don't re-call the API on every rerun
            try:
                with st.spinner("Thinking..."):
                    cache[q] = answer(q, hits)
            except LLMError as e:
                st.error(str(e))
        if q in cache:
            text, used = cache[q]
            st.write(text)
            st.caption(f"Answered by: {used}")
        with st.expander("Sources used"):
            for h in hits:
                st.markdown(f"**{h['file']}, {h['loc']}**")
                st.caption(h["text"][:400] + "...")
elif q:
    st.info("Upload files and click **Index files** first.")