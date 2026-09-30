"""Store chunk vectors in FAISS and the original text beside them.

FAISS only stores numbers. The words, file path, line range, and commit live
in a JSON file so a search result can be turned back into a source link.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chat_github.chunking import ChunkRecord, chunk_file
from chat_github.config import (
    CACHE_ROOT,
    CHUNK_SETTINGS_VERSION,
    EMBEDDING_MAX_INPUT_TOKENS,
    EMBEDDING_MODEL_NAME,
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_TOTAL_BYTES,
    OVERLAP_LINES,
    TOP_K,
)
from chat_github.errors import AppError, EmptyIndexError
from chat_github.github_loader import RemoteFile


@dataclass
class SearchHit:
    source_id: str
    score: float
    chunk: ChunkRecord


@dataclass
class RepositoryIndex:
    owner: str
    repo: str
    commit: str
    default_branch: str
    embedding_model: str
    max_input_tokens: int
    files_indexed: int
    chunks: list[ChunkRecord]
    faiss_index: object
    skipped_summary: dict[str, int] = field(default_factory=dict)
    incomplete: bool = False
    incomplete_reasons: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    loaded_from_cache: bool = False
    cache_path: Path | None = None

    @property
    def identity(self) -> str:
        return f"{self.owner}/{self.repo}@{self.commit}"


def cache_directory(
    owner: str,
    repo: str,
    commit: str,
    *,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    max_input_tokens: int = EMBEDDING_MAX_INPUT_TOKENS,
    cache_root: Path = CACHE_ROOT,
) -> Path:
    fingerprint = settings_fingerprint(embedding_model, max_input_tokens)
    return (
        cache_root
        / _safe_part(owner)
        / _safe_part(repo)
        / f"{commit}-{fingerprint}"
    )


def settings_fingerprint(embedding_model: str, max_input_tokens: int) -> str:
    payload = json.dumps(
        {
            "version": CHUNK_SETTINGS_VERSION,
            "model": embedding_model,
            "max_input_tokens": max_input_tokens,
            "overlap_lines": OVERLAP_LINES,
            "max_file_bytes": MAX_FILE_BYTES,
            "max_total_bytes": MAX_TOTAL_BYTES,
            "max_files": MAX_FILES,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_is_ready(directory: Path) -> bool:
    return all((directory / name).is_file() for name in ("manifest.json", "chunks.json", "index.faiss"))


def build_chunks(
    files: list[RemoteFile],
    *,
    owner: str,
    repo: str,
    commit: str,
    count_tokens,
    max_tokens: int,
    overlap_lines: int = OVERLAP_LINES,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for remote_file in files:
        chunks.extend(
            chunk_file(
                remote_file.path,
                remote_file.text,
                owner=owner,
                repo=repo,
                commit=commit,
                count_tokens=count_tokens,
                max_tokens=max_tokens,
                overlap_lines=overlap_lines,
            )
        )
    for chunk in chunks:
        if count_tokens(chunk.text) > max_tokens:
            raise AppError(
                f"{chunk.path} produced a chunk over the embedding limit. "
                "Indexing stopped instead of truncating it."
            )
    return chunks


def build_index(
    *,
    owner: str,
    repo: str,
    commit: str,
    default_branch: str,
    chunks: list[ChunkRecord],
    vectors: np.ndarray,
    embedding_model: str,
    max_input_tokens: int,
    files_indexed: int,
    skipped_summary: dict[str, int],
    incomplete: bool,
    incomplete_reasons: list[str],
    notices: list[str],
) -> RepositoryIndex:
    if not chunks:
        raise EmptyIndexError(_empty_message(skipped_summary))
    if len(vectors) != len(chunks):
        raise AppError("The embedding step returned a different number of vectors than chunks.")
    faiss_index = _new_faiss_index(vectors)
    return RepositoryIndex(
        owner=owner,
        repo=repo,
        commit=commit,
        default_branch=default_branch,
        embedding_model=embedding_model,
        max_input_tokens=max_input_tokens,
        files_indexed=files_indexed,
        chunks=chunks,
        faiss_index=faiss_index,
        skipped_summary=skipped_summary,
        incomplete=incomplete,
        incomplete_reasons=incomplete_reasons,
        notices=notices,
    )


def save_index(index: RepositoryIndex, directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "owner": index.owner,
        "repo": index.repo,
        "commit": index.commit,
        "default_branch": index.default_branch,
        "embedding_model": index.embedding_model,
        "max_input_tokens": index.max_input_tokens,
        "settings": settings_fingerprint(index.embedding_model, index.max_input_tokens),
    }
    payload = {
        **manifest,
        "files_indexed": index.files_indexed,
        "skipped_summary": index.skipped_summary,
        "incomplete": index.incomplete,
        "incomplete_reasons": index.incomplete_reasons,
        "notices": index.notices,
        "chunks": [
            {
                "path": chunk.path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "text": chunk.text,
                "commit": chunk.commit,
                "owner": chunk.owner,
                "repo": chunk.repo,
            }
            for chunk in index.chunks
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (directory / "chunks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    faiss = _import_faiss()
    faiss.write_index(index.faiss_index, str(directory / "index.faiss"))


def load_index(directory: Path) -> RepositoryIndex:
    if not cache_is_ready(directory):
        raise AppError("The saved index is incomplete. Click Load repository to build it again.")
    try:
        payload = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        faiss = _import_faiss()
        faiss_index = faiss.read_index(str(directory / "index.faiss"))
    except (OSError, json.JSONDecodeError) as exc:
        shutil.rmtree(directory, ignore_errors=True)
        raise AppError(
            "The saved index could not be read and was removed. Click Load repository again."
        ) from exc

    chunks = [
        ChunkRecord(
            path=item["path"],
            start_line=int(item["start_line"]),
            end_line=int(item["end_line"]),
            text=item["text"],
            commit=item["commit"],
            owner=item["owner"],
            repo=item["repo"],
        )
        for item in payload["chunks"]
    ]
    if faiss_index.ntotal != len(chunks):
        shutil.rmtree(directory, ignore_errors=True)
        raise AppError(
            "The saved vectors did not match the saved chunks, so the index was removed. "
            "Click Load repository again."
        )
    return RepositoryIndex(
        owner=payload["owner"],
        repo=payload["repo"],
        commit=payload["commit"],
        default_branch=payload["default_branch"],
        embedding_model=payload["embedding_model"],
        max_input_tokens=int(payload["max_input_tokens"]),
        files_indexed=int(payload["files_indexed"]),
        chunks=chunks,
        faiss_index=faiss_index,
        skipped_summary=dict(payload.get("skipped_summary") or {}),
        incomplete=bool(payload.get("incomplete")),
        incomplete_reasons=list(payload.get("incomplete_reasons") or []),
        notices=list(payload.get("notices") or []),
        loaded_from_cache=True,
        cache_path=directory,
    )


def search(index: RepositoryIndex, query_vector: np.ndarray, *, top_k: int = TOP_K) -> list[SearchHit]:
    """Return the closest chunks. Source links come from chunk metadata."""
    if not index.chunks:
        return []
    faiss = _import_faiss()
    vector = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
    if vector.shape[1] != index.faiss_index.d:
        raise AppError(
            "The question was embedded with a different vector size than the index. "
            "Click Refresh repository so both use the same local model."
        )
    k = min(top_k, len(index.chunks))
    scores, ids = index.faiss_index.search(vector, k)
    hits: list[SearchHit] = []
    for score, row_id in zip(scores[0], ids[0]):
        if int(row_id) < 0:
            continue
        chunk = index.chunks[int(row_id)]
        hits.append(
            SearchHit(
                source_id=f"S{len(hits) + 1}",
                score=float(score),
                chunk=chunk,
            )
        )
    return hits


def _new_faiss_index(vectors: np.ndarray):
    faiss = _import_faiss()
    matrix = np.ascontiguousarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise AppError(
            "faiss-cpu is not installed. Run pip install -r requirements.txt inside the virtual environment."
        ) from exc
    return faiss


def _safe_part(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return cleaned or "unknown"


def _empty_message(skipped_summary: dict[str, int]) -> str:
    if skipped_summary:
        return (
            "No source or documentation files were indexed. "
            "Everything found was skipped because of the file filters or size limits."
        )
    return "No source or documentation files were found on the default branch."
