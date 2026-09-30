"""Fetch a public repository without ever running its code.

The loader reads text over HTTP and returns strings. It does not import,
eval, or subprocess anything that came from the repository.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import quote, urljoin, urlparse

import requests

from chat_github.config import (
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_TOTAL_BYTES,
)
from chat_github.errors import (
    GitHubRateLimitError,
    InvalidRepositoryUrl,
    NetworkError,
    RepositoryAccessError,
)

GITHUB_API = "https://api.github.com"
RAW_HOST = "https://raw.githubusercontent.com"
USER_AGENT = "repo-gpt-learning-app"

_HTTPS_REPO = re.compile(
    r"^https?://(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)"
    r"(?:\.git)?"
    r"(?:/(?:tree|blob)/(?P<branch>[^/]+)(?:/.*)?)?"
    r"/?(?:\?[^#]*)?(?:#.*)?$",
    re.IGNORECASE,
)
_SSH_REPO = re.compile(
    r"^git@github\.com:"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)"
    r"(?:\.git)?$",
    re.IGNORECASE,
)

SKIP_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    ".next",
    ".nuxt",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".eggs",
    "__pycache__",
    "node_modules",
    "bower_components",
    "venv",
    "site-packages",
    "dist-packages",
    "vendor",
    "dist",
    "build",
    "target",
    "coverage",
    "Pods",
    "DerivedData",
}

GENERATED_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "cargo.lock",
    "composer.lock",
    "gemfile.lock",
    "go.sum",
    "pipfile.lock",
}

TEXT_EXTENSIONS = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".cjs",
    ".java",
    ".kt",
    ".kts",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".h",
    ".hh",
    ".hpp",
    ".cpp",
    ".cc",
    ".cs",
    ".swift",
    ".scala",
    ".sql",
    ".sh",
    ".bash",
    ".ps1",
    ".r",
    ".lua",
    ".md",
    ".mdx",
    ".markdown",
    ".rst",
    ".txt",
    ".ipynb",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".html",
    ".css",
    ".scss",
    ".vue",
    ".svelte",
    ".xml",
    ".gradle",
    ".dockerfile",
}

EXTENSIONLESS_NAMES = {
    "readme",
    "dockerfile",
    "makefile",
    "license",
    "gemfile",
    "rakefile",
    "procfile",
}

MEDIA_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".bmp",
    ".svg",
    ".mp4",
    ".webm",
    ".mov",
    ".avi",
    ".mkv",
    ".mp3",
    ".wav",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".pyc",
    ".pyo",
    ".class",
    ".jar",
    ".wasm",
    ".bin",
    ".pkl",
    ".pickle",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".min.js",
    ".min.css",
    ".map",
}

SECRET_NAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}

SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".kdbx"}


@dataclass(frozen=True)
class ParsedRepo:
    owner: str
    repo: str
    url_branch: str | None = None


@dataclass(frozen=True)
class TreeEntry:
    path: str
    size: int | None = None
    mode: str = "100644"
    entry_type: str = "blob"


@dataclass
class PathSelection:
    kept: list[TreeEntry] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    incomplete: bool = False
    incomplete_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RepoMetadata:
    owner: str
    repo: str
    default_branch: str
    commit: str


@dataclass(frozen=True)
class RemoteFile:
    path: str
    text: str


@dataclass
class DownloadResult:
    files: list[RemoteFile]
    skipped: dict[str, int]
    incomplete: bool
    incomplete_reasons: list[str]
    files_considered: int


def parse_github_url(url: str) -> ParsedRepo:
    """Extract owner and repository name from a GitHub URL or clone URL."""
    text = (url or "").strip()
    if not text:
        raise InvalidRepositoryUrl(
            "Enter a public GitHub repository URL, such as https://github.com/owner/repo."
        )

    https_match = _HTTPS_REPO.match(text)
    ssh_match = _SSH_REPO.match(text)
    if https_match:
        owner = https_match.group("owner")
        repo = https_match.group("repo")
        branch = https_match.group("branch")
    elif ssh_match:
        owner = ssh_match.group("owner")
        repo = ssh_match.group("repo")
        branch = None
    else:
        raise InvalidRepositoryUrl(
            "That is not a GitHub repository URL. Use a link like "
            "https://github.com/owner/repo. This app does not accept other hosts."
        )

    if owner in {".", ".."} or repo in {".", ".."}:
        raise InvalidRepositoryUrl(
            "The owner or repository name in that URL is not valid."
        )
    return ParsedRepo(owner=owner, repo=repo, url_branch=branch)


def classify_path(path: str, size: int | None) -> str | None:
    """Return a skip reason, or None when the path is worth reading."""
    parts = [part for part in path.replace("\\", "/").split("/") if part]
    if not parts:
        return "empty"
    if any(part in SKIP_DIRECTORIES for part in parts[:-1]) or parts[-1] in SKIP_DIRECTORIES:
        return "dependency_directory"

    name = parts[-1]
    lower_name = name.lower()
    if _is_secret(lower_name):
        return "secret_file"
    if lower_name in GENERATED_NAMES or lower_name.endswith(".min.js") or lower_name.endswith(".min.css"):
        return "generated_file"

    suffix = _suffix(lower_name)
    if suffix in MEDIA_EXTENSIONS or lower_name.endswith(".map"):
        return "binary_or_media"
    if suffix not in TEXT_EXTENSIONS and lower_name not in EXTENSIONLESS_NAMES:
        return "binary_or_media"
    if size is not None and size > MAX_FILE_BYTES:
        return "too_large"
    if size == 0:
        return "empty"
    return None


def select_paths(
    entries: list[TreeEntry],
    *,
    max_files: int = MAX_FILES,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> PathSelection:
    """Choose which tree entries to download, in stable path order."""
    selection = PathSelection()
    total_bytes = 0
    ordered = sorted(entries, key=lambda item: item.path.lower())

    for entry in ordered:
        if entry.entry_type != "blob":
            continue
        if entry.mode == "120000":
            _count(selection.skipped, "symlink")
            continue

        reason = classify_path(entry.path, entry.size)
        if reason is None and entry.size is not None and entry.size > max_file_bytes:
            reason = "too_large"
        if reason is not None:
            _count(selection.skipped, reason)
            continue

        next_size = entry.size or 0
        if len(selection.kept) >= max_files:
            _count(selection.skipped, "file_cap")
            selection.incomplete = True
            _add_reason(
                selection.incomplete_reasons,
                f"Stopped after {max_files} files so indexing stays small enough for a laptop.",
            )
            continue
        if total_bytes + next_size > max_total_bytes:
            _count(selection.skipped, "byte_cap")
            selection.incomplete = True
            _add_reason(
                selection.incomplete_reasons,
                "Stopped because the selected text reached the total size limit. "
                "The index does not cover the whole repository.",
            )
            continue

        selection.kept.append(entry)
        total_bytes += next_size
    return selection


def decode_text(content: bytes) -> str | None:
    """Return UTF-8 text, or None for binary content. Never executes it."""
    if not content or b"\0" in content:
        return None
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text.strip():
        return None
    return text


def github_blob_url(
    owner: str,
    repo: str,
    commit: str,
    path: str,
    start_line: int,
    end_line: int,
) -> str:
    """Build a GitHub link from stored metadata. The model does not invent this."""
    quoted_path = "/".join(quote(part) for part in path.split("/"))
    if start_line == end_line:
        anchor = f"L{start_line}"
    else:
        anchor = f"L{start_line}-L{end_line}"
    return f"https://github.com/{owner}/{repo}/blob/{commit}/{quoted_path}#{anchor}"


def fetch_repo_metadata(owner: str, repo: str, token: str | None) -> RepoMetadata:
    repo_payload = _github_json(f"/repos/{owner}/{repo}", token)
    default_branch = repo_payload.get("default_branch")
    if not isinstance(default_branch, str) or not default_branch:
        raise RepositoryAccessError(
            "GitHub did not report a default branch for this repository."
        )
    branch_payload = _github_json(
        f"/repos/{owner}/{repo}/branches/{quote(default_branch, safe='')}",
        token,
    )
    commit = ((branch_payload.get("commit") or {}).get("sha"))
    if not isinstance(commit, str) or not commit:
        raise RepositoryAccessError(
            "GitHub did not report the latest commit on the default branch."
        )
    return RepoMetadata(
        owner=owner,
        repo=repo,
        default_branch=default_branch,
        commit=commit,
    )


def fetch_tree(owner: str, repo: str, commit: str, token: str | None) -> tuple[list[TreeEntry], bool]:
    payload = _github_json(
        f"/repos/{owner}/{repo}/git/trees/{commit}?recursive=1",
        token,
    )
    entries: list[TreeEntry] = []
    for item in payload.get("tree") or []:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if not isinstance(path, str):
            continue
        size = item.get("size")
        entries.append(
            TreeEntry(
                path=path,
                size=size if isinstance(size, int) else None,
                mode=str(item.get("mode") or "100644"),
                entry_type=str(item.get("type") or "blob"),
            )
        )
    return entries, bool(payload.get("truncated"))


def download_files(
    owner: str,
    repo: str,
    commit: str,
    entries: list[TreeEntry],
    *,
    token: str | None,
    on_status,
) -> tuple[list[RemoteFile], dict[str, int]]:
    """Download file text from raw.githubusercontent.com. Does not run it."""
    files: list[RemoteFile] = []
    skipped: dict[str, int] = {}
    session = requests.Session()
    for index, entry in enumerate(entries, start=1):
        if index == 1 or index % 10 == 0 or index == len(entries):
            on_status(f"Downloading file {index} of {len(entries)}...")
        url = f"{RAW_HOST}/{owner}/{repo}/{commit}/{_quote_path(entry.path)}"
        try:
            response = _get(session, url, _plain_headers(), allow_token=False)
        except NetworkError:
            raise
        except requests.RequestException as exc:
            raise NetworkError(
                "The download from GitHub stopped because of a network problem. "
                "Check your connection and try again."
            ) from exc

        if response.status_code == 404:
            _count(skipped, "undecodable")
            continue
        if response.status_code in (403, 429):
            raise GitHubRateLimitError(_rate_limit_message(token))
        if response.status_code != 200:
            raise RepositoryAccessError(
                f"GitHub returned HTTP {response.status_code} while downloading {entry.path}."
            )
        if len(response.content) > MAX_FILE_BYTES:
            _count(skipped, "too_large")
            continue
        text = decode_text(response.content)
        if text is None:
            _count(skipped, "undecodable")
            continue
        files.append(RemoteFile(path=entry.path, text=text))
    return files, skipped


def _github_json(path: str, token: str | None) -> dict:
    session = requests.Session()
    headers = _api_headers(token)
    try:
        response = _get(session, GITHUB_API + path, headers, allow_token=bool(token))
    except requests.RequestException as exc:
        raise NetworkError(
            "Could not reach the GitHub API. Check your internet connection and try again."
        ) from exc
    return _interpret_github_response(response, token)


def _interpret_github_response(response: requests.Response, token: str | None) -> dict:
    remaining = response.headers.get("X-RateLimit-Remaining")
    if response.status_code in (403, 429) and (
        remaining == "0" or "rate limit" in response.text.lower()
    ):
        raise GitHubRateLimitError(_rate_limit_message(token))
    if response.status_code == 404:
        raise RepositoryAccessError(
            "That repository was not found, or it is not public. "
            "This app only loads public repositories on the default branch."
        )
    if response.status_code == 409:
        raise RepositoryAccessError("That repository is empty, so there is nothing to index.")
    if response.status_code in (401, 403):
        raise RepositoryAccessError(
            "GitHub refused access to that repository. Confirm that it is public. "
            "If you set GITHUB_TOKEN, check that the token is valid."
        )
    if response.status_code != 200:
        raise RepositoryAccessError(
            f"GitHub returned HTTP {response.status_code} while loading the repository."
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RepositoryAccessError("GitHub returned a response that was not JSON.") from exc
    if not isinstance(payload, dict):
        raise RepositoryAccessError("GitHub returned an unexpected response.")
    return payload


def _get(session: requests.Session, url: str, headers: dict[str, str], *, allow_token: bool):
    current = url
    current_headers = dict(headers)
    for _ in range(3):
        response = session.get(
            current,
            headers=current_headers,
            timeout=30,
            allow_redirects=False,
        )
        if response.status_code not in {301, 302, 303, 307, 308}:
            return response
        location = response.headers.get("Location")
        if not location:
            raise NetworkError("GitHub returned a redirect without a destination.")
        next_url = urljoin(current, location)
        if allow_token and urlparse(next_url).netloc != urlparse(current).netloc:
            current_headers.pop("Authorization", None)
            allow_token = False
        current = next_url
    raise NetworkError("GitHub redirected too many times, so the request was stopped.")


def _api_headers(token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _plain_headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT}


def _rate_limit_message(token: str | None) -> str:
    if token:
        return (
            "GitHub rate limit reached, even with GITHUB_TOKEN. "
            "Wait until the limit resets, then try again. This app does not retry in a loop."
        )
    return (
        "GitHub rate limit reached. Unauthenticated requests allow about 60 REST calls "
        "per hour. Add GITHUB_TOKEN to your local .env file for a higher limit "
        "(about 5,000 requests per hour), then restart the app. Do not paste the token into chat."
    )


def _is_secret(lower_name: str) -> bool:
    if lower_name in SECRET_NAMES or lower_name.startswith(".env"):
        return True
    if any(lower_name.endswith(suffix) for suffix in SECRET_SUFFIXES):
        return True
    return "id_rsa" in lower_name or "private_key" in lower_name or "serviceaccount" in lower_name


def _suffix(lower_name: str) -> str:
    if "." not in lower_name:
        return ""
    return "." + lower_name.rsplit(".", 1)[1]


def _quote_path(path: str) -> str:
    return "/".join(quote(part) for part in path.split("/"))


def _count(summary: dict[str, int], reason: str) -> None:
    summary[reason] = summary.get(reason, 0) + 1


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)
