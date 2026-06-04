"""Global exception handlers for the chord-engine API.

Maps exceptions to structured ``ErrorResponse`` JSON bodies with
appropriate HTTP status codes.  Registered in ``api/main.py``.
"""

import logging
import traceback as tb
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT, HTTP_500_INTERNAL_SERVER_ERROR

from api.models.response import ErrorResponse

logger = logging.getLogger("chord-engine")


# ---------------------------------------------------------------------------
# RequestValidationError  →  422
# ---------------------------------------------------------------------------


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic / FastAPI request validation failures.

    Returns a flat ``ErrorResponse`` with a human-readable summary of
    the fields that failed validation.
    """
    errors: list[dict[str, Any]] = exc.errors()

    # Build a concise summary: "field (reason), field (reason), ..."
    parts: list[str] = []
    for err in errors:
        loc = " → ".join(str(p) for p in err.get("loc", []))
        msg = err.get("msg", "unknown error")
        if loc:
            parts.append(f"{loc}: {msg}")
        else:
            parts.append(msg)

    detail = "; ".join(parts) if parts else "Request validation failed"

    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_CONTENT,
        content=ErrorResponse(error="validation_error", detail=detail).model_dump(),
    )


# ---------------------------------------------------------------------------
# HTTPException  →  same status code
# ---------------------------------------------------------------------------


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Handle Starlette / FastAPI HTTP exceptions.

    Attempts to preserve a ``detail`` dict if the caller already
    returned an ``ErrorResponse``-shaped body; otherwise wraps the
    detail string in a generic ``ErrorResponse``.
    """
    detail: Any = exc.detail

    if isinstance(detail, dict) and "error" in detail and "detail" in detail:
        # The caller already built an ErrorResponse dict — pass through.
        return JSONResponse(status_code=exc.status_code, content=detail)

    # Wrap plain-string details.
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error",
            detail=str(detail) if detail else "HTTP error",
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Generic Exception  →  500 (logged)
# ---------------------------------------------------------------------------


async def catch_all_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception and return a generic 500 response.

    The full traceback is logged server-side but **not** exposed to the
    client in the JSON response.
    """
    logger.error(
        "Unhandled exception on %s %s\n%s",
        request.method,
        request.url.path,
        "".join(tb.format_exception(type(exc), exc, exc.__traceback__)),
    )

    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            detail="An unexpected error occurred.",
        ).model_dump(),
    )
