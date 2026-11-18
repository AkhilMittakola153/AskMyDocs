"""
Ask My Docs — chat with your PDF, Word and text files.
Run:  streamlit run rag.py
"""

import hashlib
from datetime import datetime

import streamlit as st

import docs_engine as eng
from llm import LLMError, stream_chat

st.set_page_config(page_title="Ask My Docs", page_icon="📚", layout="centered")

st.markdown("""
<style>
#MainMenu, footer, [data-testid="stDecoration"] {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 820px;}
.hero {text-align: center; padding: 2.5rem 1rem 1rem;}
.hero h1 {font-size: 2.3rem; margin-bottom: .2rem;}
.hero p {opacity: .75; font-size: 1.05rem;}
.steps {display: flex; gap: .8rem; justify-content: center; flex-wrap: wrap; margin-top: 1.4rem;}
.step {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: .8rem 1rem; width: 210px; text-align: left;}
.step b {display: block; margin-bottom: .2rem;}
.step span {opacity: .7; font-size: .9rem;}
.src {border-left: 3px solid #e8590c; padding: .3rem .8rem; margin: .5rem 0; font-size: .9rem;}
.src-title {font-weight: 600; margin-bottom: .2rem;}
.doc-row {font-size: .88rem; padding: .15rem 0;}
</style>
""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("index_sig", None)


@st.cache_data(show_spinner=False, max_entries=50)
def parse(name, data):
    """Cached by file content, so re-uploading the same file is instant."""
    return eng.read_file(name, data)


# ---------- Sidebar: documents and settings ----------
with st.sidebar:
    st.header("📚 Your documents")
    files = st.file_uploader("Upload PDF, DOCX, TXT or MD", type=["pdf", "docx", "txt", "md"],
                             accept_multiple_files=True, label_visibility="collapsed")

    files = files or []
    too_big = [f.name for f in files if f.size > eng.MAX_FILE_MB * 1024 * 1024]
    files = [f for f in files if f.name not in too_big]
    if too_big:
        st.warning(f"Skipped (over {eng.MAX_FILE_MB} MB): {', '.join(too_big)}")

    # index automatically whenever the set of files changes
    sig = tuple(sorted((f.name, hashlib.md5(f.getvalue()).hexdigest()) for f in files))
    if sig != ss.index_sig:
        if files:
            with st.spinner("Reading documents..."):
                parsed, failed = {}, []
                for f in files:
                    try:
                        parsed[f.name] = parse(f.name, f.getvalue())
                    except Exception:
                        failed.append(f.name)
                chunks, bm25, stats = eng.build_index(parsed)
            ss.update(chunks=chunks, bm25=bm25, stats=stats, failed=failed)
        else:
            ss.update(chunks=[], bm25=None, stats={}, failed=[])
        ss.index_sig = sig

    stats = ss.get("stats", {})
    if stats:
        for name, s in stats.items():
            if s["chunks"] == 0:
                st.markdown(f"<div class='doc-row'>⚠️ <b>{name}</b> — no text found (scanned PDF?)</div>",
                            unsafe_allow_html=True)
            else:
                size = f"{s['pages']} pages · " if s["pages"] else ""
                st.markdown(f"<div class='doc-row'>✅ <b>{name}</b><br><span style='opacity:.65'>"
                            f"{size}{s['words']:,} words</span></div>", unsafe_allow_html=True)
    for name in ss.get("failed", []):
        st.error(f"Couldn't read {name}")

    st.divider()
    with st.expander("⚙️ Settings"):
        style = st.radio("Answer style", ["Concise", "Detailed"], horizontal=True)
        top_k = st.slider("Passages used per answer", 3, 10, 5,
                          help="More passages = more context, but slower and uses more tokens.")

    col1, col2 = st.columns(2)
    if col1.button("🗑️ Clear chat", disabled=not ss.messages):
        ss.messages = []
        st.rerun()
    if ss.messages:
        md = "\n\n".join(f"**{'You' if m['role'] == 'user' else 'Ask My Docs'}:** {m['content']}"
                         for m in ss.messages)
        col2.download_button("⬇️ Export", md, file_name=f"askmydocs-{datetime.now():%Y%m%d-%H%M}.md")

ready = bool(ss.get("chunks"))


# ---------- Helpers ----------
def show_sources(sources, question):
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"<div class='src'><div class='src-title'>📄 {s['file']} · {s['loc']}</div></div>",
                        unsafe_allow_html=True)
            st.markdown(eng.highlight(s["text"], question))


def answer(question):
    prev_q = next((m["content"] for m in reversed(ss.messages) if m["role"] == "user"), "")
    history = list(ss.messages)
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📚"):
        hits = eng.retrieve(question, ss.chunks, ss.bm25, k=top_k, previous_question=prev_q)
        if not hits:
            msg = "I couldn't find anything related to that in your documents. Try different keywords."
            st.markdown(msg)
            ss.messages.append({"role": "assistant", "content": msg, "sources": []})
            return
        info = {}
        try:
            text = st.write_stream(stream_chat(eng.build_prompt(question, hits, history, style), info))
        except LLMError as e:
            st.error(str(e))
            ss.messages.pop()  # forget the failed question so it can simply be asked again
            return
        model = info.get("model", "")
        if model:
            st.caption(f"Answered by {model}")
        show_sources(hits, question)
        ss.messages.append({"role": "assistant", "content": text, "sources": hits,
                            "model": model, "question": question})


# ---------- Main area ----------
prompt = st.chat_input("Ask about your documents..." if ready else "Upload a document first",
                       disabled=not ready)  # always pinned to the bottom of the page
prompt = prompt or ss.pop("pending", None)

if not ss.messages and not prompt:
    st.markdown("""
    <div class="hero">
      <h1>📚 Ask My Docs</h1>
      <p>Chat with your PDFs, Word and text files. Every answer cites the page it came from.</p>
      <div class="steps">
        <div class="step"><b>1 · Upload</b><span>Add files in the sidebar. They're read automatically.</span></div>
        <div class="step"><b>2 · Ask</b><span>Type a question or tap a suggestion below.</span></div>
        <div class="step"><b>3 · Verify</b><span>Open “Sources” to see the exact passages used.</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    if ready:
        st.write("")
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        for i, sugg in enumerate(eng.suggested_questions(ss.stats)):
            if cols[i % 2].button(sugg, key=f"sugg{i}"):
                ss.pending = sugg.split(" ", 1)[1]  # drop the emoji
                st.rerun()
    else:
        st.info("⬅️ Upload a document in the sidebar to get started.", icon="📄")

for m in ss.messages:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "📚"):
        st.markdown(m["content"])
        if m.get("model"):
            st.caption(f"Answered by {m['model']}")
        if m["role"] == "assistant":
            show_sources(m.get("sources"), m.get("question", ""))

if prompt and ready:
    answer(prompt)
