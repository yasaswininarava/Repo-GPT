"""Connect GitHub loading, local indexing, and one Gemini answer call."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chat_github.answering import (
    SYSTEM_INSTRUCTION,
    analyze_citations,
    build_prompt,
    generate_text,
    make_client,
    retrieval_query,
)
from chat_github.config import (
    CACHE_ROOT,
    EMBEDDING_MAX_INPUT_TOKENS,
    EMBEDDING_MODEL_NAME,
    gemini_api_key,
    gemini_model,
)
from chat_github.errors import AnswerError, AppError, EmptyIndexError
from chat_github.github_loader import (
    download_files,
    fetch_repo_metadata,
    fetch_tree,
    parse_github_url,
    select_paths,
)
from chat_github.index_store import (
    RepositoryIndex,
    SearchHit,
    build_chunks,
    build_index,
    cache_directory,
    cache_is_ready,
    load_index,
    save_index,
    search,
)


def load_repository(
    url: str,
    *,
    refresh: bool,
    on_status,
    embedder_factory,
    token: str | None,
    previous_commit: str | None = None,
    cache_root: Path = CACHE_ROOT,
) -> RepositoryIndex:
    """Resolve the default-branch commit, then reuse or build a local index."""
    on_status("Checking the repository URL...")
    parsed = parse_github_url(url)
    on_status(f"Resolving the default branch for {parsed.owner}/{parsed.repo}...")
    metadata = fetch_repo_metadata(parsed.owner, parsed.repo, token)
    on_status(
        f"Default branch is {metadata.default_branch}. "
        f"The commit to index is {metadata.commit[:7]}."
    )

    notices: list[str] = []
    if parsed.url_branch and parsed.url_branch != metadata.default_branch:
        notices.append(
            f"The URL mentions branch '{parsed.url_branch}', but this app indexes "
            f"the default branch '{metadata.default_branch}'."
        )

    directory = cache_directory(
        metadata.owner,
        metadata.repo,
        metadata.commit,
        cache_root=cache_root,
    )
    if cache_is_ready(directory):
        if refresh and previous_commit == metadata.commit:
            on_status("No changes on the default branch. Loading the saved index.")
        elif refresh and previous_commit and previous_commit != metadata.commit:
            on_status("A saved index for the new commit was found. Loading it.")
        else:
            on_status("Loading the saved index for this commit.")
        index = load_index(directory)
        index.notices = _merge_notices(index.notices, notices)
        index.cache_path = directory
        return index

    if refresh and previous_commit and previous_commit != metadata.commit:
        on_status("The default branch has a new commit. Building a new index.")
    elif refresh:
        on_status("No saved index for this commit. Building one now.")

    on_status("Fetching the file list...")
    entries, truncated = fetch_tree(metadata.owner, metadata.repo, metadata.commit, token)
    selection = select_paths(entries)
    if truncated:
        selection.incomplete = True
        if "GitHub did not return the full file list, so this index is incomplete." not in selection.incomplete_reasons:
            selection.incomplete_reasons.append(
                "GitHub did not return the full file list, so this index is incomplete."
            )
    if not selection.kept:
        raise EmptyIndexError(
            "No useful source or documentation files were left after filtering. "
            "Binary files, dependency folders, and secret files are skipped."
        )

    on_status(f"Downloading {len(selection.kept)} files. Repository code is not executed.")
    files, download_skips = download_files(
        metadata.owner,
        metadata.repo,
        metadata.commit,
        selection.kept,
        token=token,
        on_status=on_status,
    )
    skipped = _merge_counts(selection.skipped, download_skips)
    if not files:
        raise EmptyIndexError(
            "Files were listed, but none of them could be read as text. "
            "The index is empty."
        )

    on_status("Loading the local embedding model on CPU...")
    embedder = embedder_factory()
    _check_embedder(embedder)

    on_status(f"Splitting {len(files)} files into chunks...")
    chunks = build_chunks(
        files,
        owner=metadata.owner,
        repo=metadata.repo,
        commit=metadata.commit,
        count_tokens=embedder.count_tokens,
        max_tokens=embedder.max_input_tokens,
    )
    if not chunks:
        raise EmptyIndexError("The files did not produce any text chunks to search.")

    on_status(f"Embedding {len(chunks)} chunks locally. This does not call Gemini.")
    vectors = _embed_in_batches(chunks, embedder, on_status)
    on_status("Saving the local search index...")
    index = build_index(
        owner=metadata.owner,
        repo=metadata.repo,
        commit=metadata.commit,
        default_branch=metadata.default_branch,
        chunks=chunks,
        vectors=vectors,
        embedding_model=embedder.name,
        max_input_tokens=embedder.max_input_tokens,
        files_indexed=len(files),
        skipped_summary=skipped,
        incomplete=selection.incomplete,
        incomplete_reasons=selection.incomplete_reasons,
        notices=notices,
    )
    save_index(index, directory)
    index.cache_path = directory
    on_status("Finished indexing.")
    return index


def ask_question(
    index: RepositoryIndex,
    question: str,
    history: list[dict],
    embedder,
) -> dict:
    """Embed the question locally, search FAISS, then ask Gemini once."""
    if embedder.name != index.embedding_model or embedder.max_input_tokens != index.max_input_tokens:
        raise AppError(
            "The local embedding model does not match the saved index. "
            "Click Refresh repository before asking another question."
        )
    if not index.chunks:
        raise EmptyIndexError("The index is empty, so there is nothing to search.")

    query = retrieval_query(question, history)
    query_vector = embedder.embed_query(query)
    hits = search(index, query_vector)
    bundle = build_prompt(question, hits, history)
    if not bundle.used_hits:
        return _no_context_result()

    sources = [_source_payload(hit) for hit in bundle.used_hits]
    api_key = gemini_api_key()
    if not api_key:
        raise AnswerError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env, add your key there, "
            "and restart Streamlit. Do not paste the key into chat. "
            "Retrieved excerpts are shown below, and nothing was sent to Gemini.",
            sources,
        )

    try:
        client = make_client(api_key)
    except ImportError as exc:
        raise AnswerError(
            "The google-genai package is not installed. "
            "Run pip install -r requirements.txt inside the virtual environment.",
            sources,
        ) from exc
    try:
        answer = generate_text(
            client,
            model=gemini_model(),
            system_instruction=SYSTEM_INSTRUCTION,
            prompt=bundle.prompt,
            secrets_to_redact=(api_key,),
        )
    except AppError as exc:
        raise AnswerError(str(exc), sources) from exc

    citations = analyze_citations(answer, bundle.used_hits)
    return {
        "answer": answer,
        "sources": sources,
        "matched_ids": [hit.source_id for hit in citations.matched],
        "unmatched_ids": citations.unmatched_ids,
        "dropped_count": len(bundle.dropped_hits),
        "low_confidence": bundle.low_confidence,
        "model": gemini_model(),
    }


def _no_context_result() -> dict:
    return {
        "answer": (
            "The available context is insufficient. Local search did not return any "
            "excerpts for that question, so Gemini was not called."
        ),
        "sources": [],
        "matched_ids": [],
        "unmatched_ids": [],
        "dropped_count": 0,
        "low_confidence": True,
        "model": None,
    }


def _embed_in_batches(chunks, embedder, on_status) -> np.ndarray:
    batch_size = 32
    matrices = []
    texts = [chunk.text for chunk in chunks]
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        on_status(f"Embedding chunks {start + 1}-{stop} of {len(texts)}...")
        matrices.append(embedder.embed_documents(texts[start:stop]))
    return np.vstack(matrices)


def _check_embedder(embedder) -> None:
    if embedder.name != EMBEDDING_MODEL_NAME or embedder.max_input_tokens != EMBEDDING_MAX_INPUT_TOKENS:
        raise AppError(
            "The embedding model is not the local model this app expects. "
            "Refusing to mix vectors from different models."
        )


def _source_payload(hit: SearchHit) -> dict:
    chunk = hit.chunk
    return {
        "source_id": hit.source_id,
        "path": chunk.path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "commit": chunk.commit,
        "url": chunk.url,
        "text": chunk.text,
        "score": round(hit.score, 4),
    }


def _merge_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    merged = dict(left)
    for key, value in right.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def _merge_notices(existing: list[str], extra: list[str]) -> list[str]:
    merged = list(existing)
    for notice in extra:
        if notice not in merged:
            merged.append(notice)
    return merged
