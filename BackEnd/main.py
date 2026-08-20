"""FastAPI application entry point.

Run locally with: ``uvicorn BackEnd.main:app --reload``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from BackEnd.app.api.routes import close_online_pipeline, router
from BackEnd.app.retrieval_tools.text import close_text_search


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        # Close shared clients/pools only once when uvicorn shuts down.
        await close_text_search()
        close_online_pipeline()


app = FastAPI(
    title="4US-1F AIC Online Retrieval API",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


__all__ = ["app"]
