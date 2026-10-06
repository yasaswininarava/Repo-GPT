"""Streamlit interface for Chat with GitHub.

Run this file with Streamlit. The page collects a repository URL and a
question; the work happens in the chat_github package.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from chat_github.config import (
    MAX_HISTORY_MESSAGES,
    MAX_QUESTION_CHARS,
    SKIP_REASON_LABELS,
    github_token,
)
from chat_github.embeddings import SentenceTransformerEmbedder
from chat_github.errors import AnswerError, AppError
from chat_github.index_store import load_index
from chat_github.pipeline import ask_question, load_repository

st.set_page_config(
    page_title="Chat with GitHub",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,560;9..144,640&family=Source+Sans+3:wght@400;600&display=swap');

html, body, [class*="css"] {
  font-family: "Source Sans 3", "Segoe UI", sans-serif;
}
.stApp {
  background:
    radial-gradient(900px 420px at -10% -20%, rgba(31, 111, 91, 0.16), transparent 60%),
    radial-gradient(700px 360px at 110% -10%, rgba(184, 122, 62, 0.14), transparent 55%),
    #f7f4ef;
}
header[data-testid="stHeader"] {
  background: transparent;
}
div[data-testid="stToolbar"] {
  display: none;
}
footer {
  display: none;
}
.block-container {
  max-width: 1180px;
  padding-top: 1.6rem;
  padding-bottom: 7rem;
}
.masthead {
  display: flex;
  gap: 1rem;
  align-items: flex-start;
  margin-bottom: 0.85rem;
}
.mark {
  flex: 0 0 auto;
  width: 3.1rem;
  height: 3.1rem;
  border-radius: 0.9rem;
  background: #1f6f5b;
  color: #f7f4ef;
  display: grid;
  place-items: center;
  font-family: Fraunces, Georgia, serif;
  font-size: 1.35rem;
  box-shadow: 0 10px 24px rgba(31, 111, 91, 0.22);
}
.eyebrow {
  margin: 0;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-size: 0.72rem;
  font-weight: 600;
  color: #1f6f5b;
}
.masthead h1 {
  margin: 0.1rem 0 0.25rem;
  font-family: Fraunces, Georgia, serif;
  font-size: 2.7rem;
  line-height: 1.05;
  font-weight: 560;
  color: #1f1a17;
}
.lede {
  margin: 0;
  max-width: 42rem;
  color: #5c564f;
  font-size: 1.05rem;
}
.notice {
  margin: 0 0 1.4rem;
  padding: 0.7rem 0.9rem;
  border-radius: 0.8rem;
  background: rgba(255, 253, 251, 0.8);
  border: 1px solid #e4ddd2;
  color: #4e4944;
  font-size: 0.92rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255, 253, 251, 0.88);
  border: 1px solid #e4ddd2 !important;
  border-radius: 1rem !important;
  box-shadow: 0 12px 30px rgba(60, 42, 24, 0.05);
}
p.panel-label {
  margin: 0 0 0.35rem;
  font-family: Fraunces, Georgia, serif !important;
  font-size: 1.55rem !important;
  font-weight: 560 !important;
  line-height: 1.2 !important;
  color: #1f1a17;
}
@media (max-width: 760px) {
  div[data-testid="stHorizontalBlock"] {
    flex-direction: column;
  }
}
.hint {
  color: #6b625b;
  font-size: 0.92rem;
  margin-top: 0.35rem;
}
.empty-card {
  padding: 0.25rem 0.15rem 0.4rem;
}
.empty-card ol {
  margin: 0.4rem 0 0;
  padding-left: 1.15rem;
  color: #3f3a35;
}
.empty-card li {
  margin: 0.35rem 0;
}
div[data-testid="stMetric"] {
  background: #fffdfb;
  border: 1px solid #e4ddd2;
  border-radius: 0.8rem;
  padding: 0.55rem 0.75rem;
}
div[data-testid="stChatMessage"] {
  background: rgba(255, 253, 251, 0.75);
  border: 1px solid #eadfd2;
  border-radius: 0.95rem;
  padding: 0.35rem 0.2rem;
}
button[data-testid="stBaseButton-primary"] {
  background: #1f6f5b;
  border: 1px solid #1f6f5b;
  border-radius: 0.7rem;
  white-space: nowrap;
}
button[data-testid="stBaseButton-secondary"] {
  border-radius: 0.7rem;
  border-color: #d9d0c4;
  background: #fffdfb;
  color: #1f1a17;
  white-space: nowrap;
}
div[data-testid="stTextInput"] input {
  border-radius: 0.7rem;
  border-color: #d9d0c4;
  background: #fffdfb;
}
</style>
"""


def main() -> None:
    st.markdown(_STYLES, unsafe_allow_html=True)
    st.markdown(
        """
<div class="masthead">
  <div class="mark">Rg</div>
  <div>
    <p class="eyebrow">Public repositories</p>
    <h1>Chat with GitHub</h1>
    <p class="lede">Search the code on this computer. Gemini writes the answer from the excerpts that search finds.</p>
  </div>
</div>
<div class="notice">Selected excerpts are sent to Google for the answer. Embeddings stay on this computer. Do not load a repository that contains secrets.</div>
        """,
        unsafe_allow_html=True,
    )

    _ensure_state()
    library, conversation = st.columns([0.9, 1.25], gap="large")

    with library:
        with st.container(border=True):
            st.markdown('<p class="panel-label">Repository</p>', unsafe_allow_html=True)
            repository_url = st.text_input(
                "GitHub repository URL",
                placeholder="https://github.com/owner/repo",
            )
            st.markdown(
                '<p class="hint">Public repositories load without a token. A token in .env raises the GitHub rate limit. Only the default branch is indexed.</p>',
                unsafe_allow_html=True,
            )
            load_column, refresh_column = st.columns(2)
            load_clicked = load_column.button(
                "Load repository", type="primary", use_container_width=True
            )
            refresh_clicked = refresh_column.button(
                "Refresh repository", use_container_width=True
            )

        if load_clicked or refresh_clicked:
            _load_from_form(repository_url, refresh=refresh_clicked)
        _show_report(st.session_state.report)

    index = _open_index(st.session_state.cache_path)

    with conversation:
        with st.container(border=True):
            st.markdown('<p class="panel-label">Conversation</p>', unsafe_allow_html=True)
            if st.session_state.conversation_note:
                st.info(st.session_state.conversation_note)
            if index is None and not st.session_state.messages:
                st.markdown(
                    """
<div class="empty-card">
  <p class="hint">Load a repository, then ask about a file, function, or design choice.</p>
  <ol>
    <li>Paste a public GitHub URL.</li>
    <li>Click Load repository and wait for the index.</li>
    <li>Ask a question in the box at the bottom of the page.</li>
  </ol>
</div>
                    """,
                    unsafe_allow_html=True,
                )
            elif index is None:
                st.caption("Load a repository before asking a question.")
            for message in st.session_state.messages:
                _show_message(message)

    question = st.chat_input("Ask about the loaded repository")
    if question:
        _handle_question(question, index)


@st.cache_resource(show_spinner="Loading the local embedding model on CPU...")
def get_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


def _ensure_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("cache_path", None)
    st.session_state.setdefault("report", None)
    st.session_state.setdefault("conversation_note", None)


def _load_from_form(repository_url: str, *, refresh: bool) -> None:
    if not repository_url.strip():
        st.error("Enter a public GitHub repository URL, such as https://github.com/owner/repo.")
        return

    previous = st.session_state.report or {}
    with st.status("Working on the repository...", expanded=True) as status:
        def on_status(message: str) -> None:
            status.write(message)

        try:
            loaded = load_repository(
                repository_url,
                refresh=refresh,
                on_status=on_status,
                embedder_factory=get_embedder,
                token=github_token(),
                previous_commit=previous.get("commit"),
            )
        except AppError as exc:
            status.update(label="Could not load the repository", state="error")
            st.error(str(exc))
            return
        status.update(label="Repository ready", state="complete")

    new_identity = loaded.identity
    old_identity = previous.get("identity")
    if old_identity and old_identity != new_identity:
        st.session_state.messages = []
        st.session_state.conversation_note = (
            "The conversation was cleared because the repository or commit changed."
        )
    elif not refresh:
        st.session_state.conversation_note = None

    st.session_state.cache_path = str(loaded.cache_path)
    st.session_state.report = {
        "identity": new_identity,
        "owner": loaded.owner,
        "repo": loaded.repo,
        "commit": loaded.commit,
        "branch": loaded.default_branch,
        "files": loaded.files_indexed,
        "chunks": len(loaded.chunks),
        "skipped": loaded.skipped_summary,
        "incomplete": loaded.incomplete,
        "reasons": loaded.incomplete_reasons,
        "notices": loaded.notices,
        "from_cache": loaded.loaded_from_cache,
    }


def _open_index(cache_path: str | None):
    if not cache_path:
        return None
    try:
        return load_index(Path(cache_path))
    except AppError as exc:
        st.error(str(exc))
        return None


def _show_report(report: dict | None) -> None:
    if not report:
        return
    location = f"{report['owner']}/{report['repo']}"
    commit = report["commit"][:7]
    origin = "Saved index" if report["from_cache"] else "New index"
    files_col, chunks_col, commit_col = st.columns(3)
    files_col.metric("Files", report["files"])
    chunks_col.metric("Chunks", report["chunks"])
    commit_col.metric("Commit", commit)
    st.caption(f"{origin} · {location} · branch {report['branch']}")
    for notice in report.get("notices") or []:
        st.warning(notice)
    skipped = _format_skipped(report.get("skipped") or {})
    if skipped:
        st.warning("Skipped files: " + skipped)
    if report.get("incomplete"):
        for reason in report.get("reasons") or []:
            st.warning(reason)


def _handle_question(question: str, index) -> None:
    cleaned = question.strip()
    if index is None:
        st.error("Load a repository before asking a question.")
        return
    if not cleaned:
        st.error("Enter a question about the repository.")
        return
    if len(cleaned) > MAX_QUESTION_CHARS:
        st.error(f"Please keep the question under {MAX_QUESTION_CHARS} characters.")
        return

    history = list(st.session_state.messages)
    with st.spinner("Searching the local index, then asking Gemini once..."):
        try:
            result = ask_question(index, cleaned, history, get_embedder())
        except AnswerError as exc:
            _append_exchange(cleaned, str(exc), error=True, sources=exc.sources)
        except AppError as exc:
            _append_exchange(cleaned, str(exc), error=True, sources=[])
        else:
            _append_exchange(
                cleaned,
                result["answer"],
                error=False,
                sources=result["sources"],
                matched_ids=result["matched_ids"],
                unmatched_ids=result["unmatched_ids"],
                dropped_count=result["dropped_count"],
                low_confidence=result["low_confidence"],
                model=result["model"],
            )
    st.rerun()


def _append_exchange(question: str, answer: str, **extra) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer, **extra})
    st.session_state.messages = st.session_state.messages[-MAX_HISTORY_MESSAGES:]
    st.session_state.conversation_note = None


def _show_message(message: dict) -> None:
    avatar = "✎" if message["role"] == "user" else "◇"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] != "assistant":
            return
        if message.get("low_confidence"):
            st.caption("The closest local matches were weak. Read the answer with that in mind.")
        if message.get("dropped_count"):
            st.caption(
                f"{message['dropped_count']} extra matches were left out of the Gemini prompt "
                "so the request stays within the context budget."
            )
        if message.get("model"):
            st.caption(f"Answered by {message['model']}.")
        _show_sources(message)


def _show_sources(message: dict) -> None:
    sources = message.get("sources") or []
    if not sources:
        return
    st.markdown("**Retrieved sources**")
    st.caption(
        "Found by local search. GitHub links use the indexed commit and line range. "
        "They are not written by the answering model."
    )
    for source in sources:
        label = (
            f"{source['source_id']} · {source['path']} · "
            f"lines {source['start_line']}-{source['end_line']}"
        )
        with st.expander(label):
            st.markdown(f"[Open this range on GitHub]({source['url']})")
            st.caption(f"Similarity score {source['score']}")
            st.code(source["text"], language=_language(source["path"]))

    matched = message.get("matched_ids") or []
    unmatched = message.get("unmatched_ids") or []
    if not matched and not unmatched:
        return
    st.markdown("**Citations in the answer**")
    st.caption(
        "A citation is matched when the answer names a retrieved id such as [S1]. "
        "A match means the id was retrieved. It does not prove the sentence is correct."
    )
    if matched:
        st.write("Matched retrieved excerpts: " + ", ".join(matched))
    if unmatched:
        names = ", ".join(f"[{item}]" for item in unmatched)
        st.warning(
            f"The answer mentioned {names}, which was not in the retrieved excerpts. "
            "Treat those references as unverified."
        )


def _format_skipped(summary: dict[str, int]) -> str:
    parts = []
    for reason, count in sorted(summary.items()):
        if count <= 0:
            continue
        label = SKIP_REASON_LABELS.get(reason, reason.replace("_", " "))
        parts.append(f"{count} {label}")
    return ", ".join(parts)


def _language(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".md": "markdown",
        ".json": "json",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".html": "html",
        ".css": "css",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".sh": "bash",
    }.get(suffix, "text")


main()
