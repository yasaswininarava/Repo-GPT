from chat_github.errors import InvalidRepositoryUrl
from chat_github.github_loader import (
    TreeEntry,
    classify_path,
    decode_text,
    parse_github_url,
    select_paths,
)
import pytest


def test_parse_accepts_common_public_urls():
    parsed = parse_github_url("https://github.com/pallets/flask")
    assert (parsed.owner, parsed.repo, parsed.url_branch) == ("pallets", "flask", None)

    parsed = parse_github_url("https://github.com/pallets/flask.git")
    assert parsed.repo == "flask"

    parsed = parse_github_url("https://github.com/pallets/flask/")
    assert parsed.repo == "flask"

    parsed = parse_github_url("https://github.com/pallets/my.repo")
    assert parsed.repo == "my.repo"

    parsed = parse_github_url("https://github.com/pallets/flask/tree/dev/src")
    assert parsed.url_branch == "dev"
    assert parsed.repo == "flask"

    parsed = parse_github_url("git@github.com:pallets/flask.git")
    assert (parsed.owner, parsed.repo) == ("pallets", "flask")


def test_parse_rejects_non_repository_urls():
    for value in ["", "   ", "https://gitlab.com/pallets/flask", "https://github.com/pallets", "not a url"]:
        with pytest.raises(InvalidRepositoryUrl):
            parse_github_url(value)

    with pytest.raises(InvalidRepositoryUrl):
        parse_github_url("https://github.com/pallets/flask/issues/1")


def test_filter_skips_binaries_dependencies_secrets_and_locks():
    entries = [
        TreeEntry(".env", 12),
        TreeEntry(".env.local", 12),
        TreeEntry("certs/private_key.pem", 40),
        TreeEntry("node_modules/pkg/index.js", 40),
        TreeEntry("logo.png", 40),
        TreeEntry("demo.mp4", 40),
        TreeEntry("package-lock.json", 40),
        TreeEntry("src/app.py", 40),
        TreeEntry("README.md", 40),
        TreeEntry("Dockerfile", 40),
    ]
    selection = select_paths(entries)
    assert [item.path for item in selection.kept] == ["Dockerfile", "README.md", "src/app.py"]
    assert selection.skipped["secret_file"] == 3
    assert selection.skipped["dependency_directory"] == 1
    assert selection.skipped["binary_or_media"] == 2
    assert selection.skipped["generated_file"] == 1
    assert selection.incomplete is False


def test_filter_marks_size_and_count_limits():
    huge = TreeEntry("src/huge.py", 200_000)
    assert classify_path(huge.path, huge.size) == "too_large"

    entries = [TreeEntry(f"src/file_{index}.py", 10) for index in range(5)]
    selection = select_paths(entries, max_files=2, max_total_bytes=1_000)
    assert len(selection.kept) == 2
    assert selection.incomplete is True
    assert selection.skipped["file_cap"] == 3


def test_total_byte_cap_stops_the_selection():
    entries = [TreeEntry(f"src/file_{index}.py", 100) for index in range(4)]
    selection = select_paths(entries, max_files=10, max_total_bytes=250)
    assert len(selection.kept) == 2
    assert selection.incomplete is True
    assert selection.skipped["byte_cap"] == 2


def test_decode_text_rejects_binary_and_returns_source_as_text():
    assert decode_text(b"def hello():\n    return 1\n") == "def hello():\n    return 1\n"
    assert decode_text(b"\xff\xfe not utf-8") is None
    assert decode_text(b"has\0null") is None
    assert decode_text(b"   \n") is None


def test_loader_source_does_not_execute_repository_code():
    from pathlib import Path

    source = Path("chat_github/github_loader.py").read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "os.system" not in source
    assert "exec(" not in source
    assert "eval(" not in source
