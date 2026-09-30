from chat_github.chunking import chunk_file
from chat_github.github_loader import github_blob_url


def _line_tokens(text: str) -> int:
    return max(1, len(text.splitlines()) * 10)


def test_chunks_keep_function_boundaries_and_line_metadata():
    source = "def alpha():\n    return 1\n\ndef beta():\n    return 2\n"
    chunks = chunk_file(
        "src/app.py",
        source,
        owner="octo",
        repo="demo",
        commit="abc123def4567890",
        count_tokens=_line_tokens,
        max_tokens=30,
        overlap_lines=0,
    )

    beta = next(chunk for chunk in chunks if "def beta():" in chunk.text)
    alpha = next(chunk for chunk in chunks if "def alpha():" in chunk.text)
    assert alpha.start_line == 1
    assert beta.start_line == 4
    assert beta.end_line == 5
    assert beta.text == "def beta():\n    return 2"
    assert beta.commit == "abc123def4567890"
    assert beta.path == "src/app.py"

    original_lines = source.splitlines()
    for chunk in chunks:
        expected = "\n".join(original_lines[chunk.start_line - 1 : chunk.end_line])
        assert chunk.text == expected
        assert _line_tokens(chunk.text) <= 30
        assert chunk.url == github_blob_url(
            "octo",
            "demo",
            chunk.commit,
            chunk.path,
            chunk.start_line,
            chunk.end_line,
        )
        assert f"L{chunk.start_line}-L{chunk.end_line}" in chunk.url


def test_chunks_stay_inside_the_token_limit_when_a_function_is_long():
    lines = ["def huge():"] + [f"    value_{index} = {index}" for index in range(12)]
    source = "\n".join(lines)
    max_tokens = 30
    chunks = chunk_file(
        "src/huge.py",
        source,
        owner="octo",
        repo="demo",
        commit="commitsha",
        count_tokens=_line_tokens,
        max_tokens=max_tokens,
        overlap_lines=2,
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert _line_tokens(chunk.text) <= max_tokens
        assert chunk.path == "src/huge.py"
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


def test_a_single_long_line_is_split_without_exceeding_the_limit():
    source = " ".join(f"word{index}" for index in range(40))

    def word_tokens(text: str) -> int:
        return max(1, len(text.split()))

    chunks = chunk_file(
        "README.md",
        source,
        owner="octo",
        repo="demo",
        commit="commitsha",
        count_tokens=word_tokens,
        max_tokens=8,
        overlap_lines=0,
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert word_tokens(chunk.text) <= 8
        assert chunk.start_line == 1
        assert chunk.end_line == 1
        assert chunk.text in source
