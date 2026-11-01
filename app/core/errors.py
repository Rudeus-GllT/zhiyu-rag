from __future__ import annotations


class AppError(Exception):
    """A domain error that is safe to show to API clients."""

    def __init__(self, message: str, *, code: str = "bad_request", status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class NotFoundError(AppError):
    def __init__(self, message: str = "资源不存在"):
        super().__init__(message, code="not_found", status_code=404)


class ConflictError(AppError):
    def __init__(self, message: str):
        super().__init__(message, code="conflict", status_code=409)


class UnsupportedFileError(AppError):
    def __init__(self, message: str):
        super().__init__(message, code="unsupported_file", status_code=415)


class LLMError(AppError):
    def __init__(self, message: str):
        super().__init__(message, code="llm_error", status_code=502)


class IndexError_(AppError):
    def __init__(self, message: str):
        super().__init__(message, code="index_error", status_code=500)
