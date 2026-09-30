"""Split repository text into chunks the embedding model can accept whole.

all-MiniLM-L6-v2 truncates anything past 256 tokens. These functions refuse to
emit a chunk that the supplied token counter says is too long.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from chat_github.github_loader import github_blob_url

# Definitions may be indented. Markdown headings are only treated as boundaries
# when they start at column 0, so a Python comment inside a function stays put.
_CODE_BOUNDARY = re.compile(
    r"^(?:def |async def |class |function |export default function |"
    r"export function |export class |pub fn |fn |func )"
)
_HEADING = re.compile(r"^#+\s+\S")


@dataclass(frozen=True)
class ChunkRecord:
    path: str
    start_line: int
    end_line: int
    text: str
    commit: str
    owner: str
    repo: str

    @property
    def url(self) -> str:
        return github_blob_url(
            self.owner,
            self.repo,
            self.commit,
            self.path,
            self.start_line,
            self.end_line,
        )


def chunk_file(
    path: str,
    text: str,
    *,
    owner: str,
    repo: str,
    commit: str,
    count_tokens,
    max_tokens: int,
    overlap_lines: int,
) -> list[ChunkRecord]:
    """Split one file. Line numbers refer to the original file, starting at 1."""
    if max_tokens < 8:
        raise ValueError("The embedding token limit is too small to chunk text safely.")

    lines = list(enumerate(text.splitlines(), start=1))
    if not lines:
        return []

    chunks: list[list[tuple[int, str]]] = []
    pending: list[tuple[int, str]] = []

    def emit(block: list[tuple[int, str]]) -> None:
        if block and _join(block).strip():
            chunks.append(block)

    for unit in _units(lines):
        if count_tokens(_join(unit)) > max_tokens:
            emit(pending)
            pending = []
            for piece in _split_to_fit(unit, count_tokens, max_tokens, overlap_lines):
                emit(piece)
            continue

        candidate = pending + unit
        if pending and count_tokens(_join(candidate)) > max_tokens:
            previous = pending
            emit(previous)
            pending = _with_overlap(previous, unit, count_tokens, max_tokens, overlap_lines)
        else:
            pending = candidate

    emit(pending)
    return [
        ChunkRecord(
            path=path,
            start_line=block[0][0],
            end_line=block[-1][0],
            text=_join(block),
            commit=commit,
            owner=owner,
            repo=repo,
        )
        for block in chunks
    ]


def _is_boundary(text: str) -> bool:
    if _CODE_BOUNDARY.match(text.lstrip()):
        return True
    return _HEADING.match(text) is not None


def _units(lines: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    units: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for line_no, text in lines:
        if current and _is_boundary(text):
            units.append(current)
            current = [(line_no, text)]
        else:
            current.append((line_no, text))
    if current:
        units.append(current)
    return units


def _split_to_fit(
    lines: list[tuple[int, str]],
    count_tokens,
    max_tokens: int,
    overlap_lines: int,
) -> list[list[tuple[int, str]]]:
    pieces: list[list[tuple[int, str]]] = []
    expanded = _expand_long_lines(lines, count_tokens, max_tokens)
    start = 0
    total = len(expanded)
    while start < total:
        end = start
        last_good = None
        while end < total:
            candidate = expanded[start : end + 1]
            if count_tokens(_join(candidate)) <= max_tokens:
                last_good = end
                end += 1
            else:
                break
        if last_good is None:
            raise ValueError("A single token still exceeds the embedding model limit.")
        piece = expanded[start : last_good + 1]
        pieces.append(piece)
        next_start = last_good + 1 - max(overlap_lines, 0)
        if next_start <= start:
            next_start = last_good + 1
        start = next_start
    return pieces


def _expand_long_lines(
    lines: list[tuple[int, str]],
    count_tokens,
    max_tokens: int,
) -> list[tuple[int, str]]:
    expanded: list[tuple[int, str]] = []
    for line_no, text in lines:
        if count_tokens(text) <= max_tokens:
            expanded.append((line_no, text))
            continue
        remaining = text
        while remaining:
            if count_tokens(remaining) <= max_tokens:
                expanded.append((line_no, remaining))
                break
            best = _longest_prefix(remaining, count_tokens, max_tokens)
            expanded.append((line_no, remaining[:best]))
            remaining = remaining[best:]
    return expanded


def _longest_prefix(text: str, count_tokens, max_tokens: int) -> int:
    lo = 1
    hi = len(text) - 1
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if count_tokens(text[:mid]) <= max_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best < 1:
        raise ValueError("A single character exceeds the embedding model limit.")
    return best


def _with_overlap(
    previous: list[tuple[int, str]],
    upcoming: list[tuple[int, str]],
    count_tokens,
    max_tokens: int,
    overlap_lines: int,
) -> list[tuple[int, str]]:
    if overlap_lines <= 0 or not previous:
        return upcoming
    tail = previous[-overlap_lines:]
    for drop in range(0, len(tail) + 1):
        combined = tail[drop:] + upcoming
        if count_tokens(_join(combined)) <= max_tokens:
            return combined
    return upcoming


def _join(lines: list[tuple[int, str]]) -> str:
    return "\n".join(text for _, text in lines)
