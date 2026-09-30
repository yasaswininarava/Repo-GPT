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

st.set_page_config(page_title="Chat with GitHub", page_icon="💬")


def main() -> None:
    st.title("Chat with GitHub")
    st.caption(
        "Ask questions about a public repository. Search runs on your computer. Answers come from Gemini."
    )
    st.info(
        "Selected repository excerpts are sent to Google to generate an answer. "
        "Embeddings stay on this computer. Do not load a repository that contains secrets."
    )

    _ensure_state()
    repository_url = st.text_input(
        "GitHub repository URL",
        placeholder="https://github.com/owner/repo",
    )
    st.caption(
        "Public repositories can load without a GitHub token. "
        "A token in .env raises the REST rate limit from about 60 to about 5,000 requests an hour. "
        "This app indexes the default branch only."
    )

    load_column, refresh_column = st.columns(2)
    load_clicked = load_column.button("Load repository", type="primary", use_container_width=True)
    refresh_clicked = refresh_column.button("Refresh repository", use_container_width=True)
    if load_clicked or refresh_clicked:
        _load_from_form(repository_url, refresh=refresh_clicked)

    _show_report(st.session_state.report)
    index = _open_index(st.session_state.cache_path)
    if st.session_state.conversation_note:
        st.info(st.session_state.conversation_note)

    st.subheader("Conversation")
    if index is None:
        st.caption("Load a repository before asking a question.")
    for message in st.session_state.messages:
        _show_message(message)

    question = st.chat_input("Ask a question about the loaded repository")
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
    origin = "Saved index reused." if report["from_cache"] else "New index created."
    st.success(
        f"{origin} Indexed {report['files']} files and {report['chunks']} chunks "
        f"from {location} on branch {report['branch']} at commit {commit}."
    )
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
    with st.chat_message(message["role"]):
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
