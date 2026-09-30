"""Errors that the interface can show directly to a learner."""


class AppError(Exception):
    """A problem the user can act on. The message must not contain secrets."""


class InvalidRepositoryUrl(AppError):
    """The text entered is not a usable public GitHub repository URL."""


class RepositoryAccessError(AppError):
    """GitHub could not provide this public repository."""


class GitHubRateLimitError(AppError):
    """GitHub refused the request because the rate limit was reached."""


class EmptyIndexError(AppError):
    """Nothing indexable was left after filtering."""


class MissingCredentialError(AppError):
    """A required local environment variable is missing."""


class GeminiQuotaError(AppError):
    """Gemini refused the request because a free-tier quota was exhausted."""


class GeminiAuthError(AppError):
    """Gemini rejected the API key."""


class NetworkError(AppError):
    """A remote service could not be reached."""


class AnswerError(AppError):
    """An answer failed after retrieval. Sources are the excerpts already found."""

    def __init__(self, message: str, sources: list[dict] | None = None):
        super().__init__(message)
        self.sources = sources or []
