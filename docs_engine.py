"""
docs_engine.py — reading, chunking, searching and prompt-building for Ask My Docs.
No Streamlit code here, so it can be tested on its own.
"""

import io
import re

from rank_bm25 import BM25Okapi

CHUNK_WORDS, OVERLAP = 250, 50
MAX_FILE_MB = 20

STOPWORDS = set("""
a an the and or but if then so of to in on at by for with from as is are was were be been being am
do does did doing have has had having i me my we our you your he she it its they them their this that
these those what which who whom whose when where why how there here can could should would will shall
may might must not no nor only own same than too very just about above after again against all any
both each few more most other some such into through during before below up down out off over under
further once please tell me give show explain describe list document documents file files pdf
""".split())

SUMMARY_RE = re.compile(
    r"\b(summar\w*|overview|key points|main points|highlights|gist|tl;?dr|"
    r"what (is|are) (this|these|the) (document|file|pdf|paper)s?( about)?)\b", re.I)


# ---------- reading ----------
def read_file(name: str, data: bytes):
    """Return a list of (page_label_or_None, text) for one uploaded file."""
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [(f"p.{i + 1}", page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if ext == "docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        lines = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                lines.append(" | ".join(c.text for c in row.cells))
        return [(None, "\n".join(lines))]
    return [(None, data.decode("utf-8", errors="ignore"))]  # txt / md


# ---------- chunking and index ----------
def chunk(name, loc, text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP
    out = []
    for k, i in enumerate(range(0, max(len(words), 1), step)):
        piece = " ".join(words[i:i + CHUNK_WORDS])
        if piece.strip():
            out.append({"file": name, "loc": loc or f"part {k + 1}", "text": piece})
    return out


def tokenize(s):
    return re.findall(r"\w+", s.lower())


def keywords(s):
    return [t for t in tokenize(s) if t not in STOPWORDS and len(t) > 1]


def build_index(parsed):
    """parsed: {filename: [(loc, text), ...]} -> (chunks, bm25, per-file stats)."""
    chunks, stats = [], {}
    for name, pages in parsed.items():
        file_chunks = []
        for loc, text in pages:
            file_chunks += chunk(name, loc, text)
        chunks += file_chunks
        stats[name] = {
            "pages": sum(1 for loc, _ in pages if loc) or None,
            "words": sum(len(t.split()) for _, t in pages),
            "chunks": len(file_chunks),
        }
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None
    return chunks, bm25, stats


# ---------- retrieval ----------
def is_summary_request(q):
    return bool(SUMMARY_RE.search(q))


def spread_chunks(chunks, k):
    """Evenly spaced chunks covering every file, for summaries and overviews."""
    if len(chunks) <= k:
        return list(chunks)
    by_file = {}
    for c in chunks:
        by_file.setdefault(c["file"], []).append(c)
    picked = []
    for file_chunks in by_file.values():
        n = max(1, round(k * len(file_chunks) / len(chunks)))   # share of k, at least 1 per file
        step = len(file_chunks) / n
        picked += [file_chunks[int(i * step)] for i in range(min(n, len(file_chunks)))]
    return picked


def retrieve(q, chunks, bm25, k=5, previous_question=""):
    """Top-k chunks for the question. Keywords from the previous question are added at half weight,
    so follow-ups like 'and what about its cost?' still find the right passages."""
    if not chunks:
        return []
    if is_summary_request(q):
        return spread_chunks(chunks, k + 1)
    q_terms = keywords(q)
    terms = q_terms + [t for t in keywords(previous_question) if t not in q_terms]
    if not terms:
        terms = tokenize(q)
    scores = bm25.get_scores(q_terms or terms)
    if previous_question:
        prev_scores = bm25.get_scores([t for t in terms if t not in q_terms] or q_terms)
        scores = [s + 0.5 * p for s, p in zip(scores, prev_scores)]
    term_set = set(terms)
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    # keep chunks sharing a real keyword (BM25 scores can be <= 0 for tiny files)
    hits = [chunks[i] for i in ranked if term_set & set(tokenize(chunks[i]["text"]))]
    return hits[:k]


# ---------- prompt ----------
STYLES = {
    "Concise": "Answer in a few sentences or a short bullet list.",
    "Detailed": "Give a thorough, well-structured answer with headings or bullet points where helpful.",
}


def build_prompt(q, hits, history, style="Concise"):
    sources = "\n\n".join(f"[{h['file']}, {h['loc']}]\n{h['text']}" for h in hits)
    convo = ""
    recent = history[-4:]  # last 2 exchanges, for follow-up questions
    if recent:
        convo = "Conversation so far (for context only):\n" + "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:600]}" for m in recent) + "\n\n"
    return (
        "You are Ask My Docs, an assistant that answers questions about the user's uploaded documents.\n"
        "Rules:\n"
        "- Use ONLY the sources below. Do not use outside knowledge or guess.\n"
        "- After each fact, cite its source like [filename, p.3].\n"
        "- If the sources don't contain the answer, say: \"I couldn't find that in your documents.\" "
        "Then mention what related information the sources do contain, if any.\n"
        f"- {STYLES.get(style, STYLES['Concise'])}\n"
        "- Use Markdown formatting.\n\n"
        f"{convo}Sources:\n{sources}\n\n---\nQuestion: {q}"
    )


def highlight(text, q, limit=350):
    """Short snippet with the question's keywords in bold."""
    snippet = text[:limit].replace("*", "").replace("_", " ") + ("…" if len(text) > limit else "")
    terms = sorted(set(keywords(q)), key=len, reverse=True)
    if terms:
        snippet = re.sub(r"\b(" + "|".join(map(re.escape, terms)) + r")\b", r"**\1**", snippet, flags=re.I)
    return snippet


def suggested_questions(stats):
    multi = len(stats) > 1
    return [
        "📝 Summarize " + ("these documents" if multi else "this document"),
        "🔑 What are the key points?",
        "📅 What dates, deadlines or phases are mentioned?",
        "❓ What questions can this document answer?" if not multi else "⚖️ How do these documents differ?",
    ]
