"""FastAPI application entry point for the Chord Engine API.

Run the development server with::

    uvicorn api.main:app --reload --port 8000

For Hugging Face Spaces deployment, the Dockerfile uses port 7860.

The full OpenAPI documentation is available at ``/api/docs`` (Swagger UI)
and ``/api/redoc`` (ReDoc).
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.middleware.errors import (
    catch_all_handler,
    http_exception_handler,
    validation_error_handler,
)
from api.routes import health, jobs
from config import API
from api.services.job_store import job_store

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("chord-engine")

# ---------------------------------------------------------------------------
# Periodic cleanup
# ---------------------------------------------------------------------------


async def cleanup_loop() -> None:
    """Run periodically and purge jobs older than the configured max age.

    Never raises — errors are logged and the loop continues.
    """
    interval = API["cleanup_interval_seconds"]
    max_age = API["max_job_age_minutes"]
    while True:
        try:
            await asyncio.sleep(interval)
            count = job_store.cleanup_old_jobs(max_age_minutes=max_age)
            if count > 0:
                logger.info("Cleanup removed %d expired job(s).", count)
        except Exception:
            logger.exception("Cleanup loop encountered an error — continuing.")


# ---------------------------------------------------------------------------
# Lifespan — replaces the deprecated ``on_event`` API
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Handle application startup / shutdown lifecycle."""
    # -- startup -----------------------------------------------------------
    from api import _app_start_time as app_start

    # Re-bind the module-level start time so it reflects the actual server
    # boot moment rather than the import time of ``api.__init__``.
    # We use a plain attribute on the app object so the health endpoint can
    # read it without a circular import.
    application.state.start_time = time.time()

    cleanup_task = asyncio.create_task(cleanup_loop())
    logger.info("Chord Engine API started.")

    yield

    # -- shutdown ----------------------------------------------------------
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("Chord Engine API shutting down.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Chord Engine API",
    description="Async REST API for audio chord recognition.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# -- CORS ------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Exception handlers ----------------------------------------------------

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

app.add_exception_handler(Exception, catch_all_handler)
# More-specific handlers registered after the generic one so they
# take precedence when the exception type matches.
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)

# -- Routers ---------------------------------------------------------------

app.include_router(health.router)
app.include_router(jobs.router)


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    """Redirect browser visitors to the Swagger UI."""
    return RedirectResponse(url="/api/docs")
