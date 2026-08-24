"""FastAPI routes for the online retrieval pipeline."""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from BackEnd.app.Database.postgre_manager import PostgreManager
from BackEnd.app.api.config import (
    API_PREFIX,
    RETRIEVAL_V2_CORPUS_STATS_PATH,
    RETRIEVAL_V2_ENABLED,
    RETRIEVAL_V2_VIDEO_INDEX_PATH,
    SELECTIVE_VERIFIER_ENABLED,
)
from BackEnd.app.api.models import HealthResponse, QueryRequest, QueryResponse
from BackEnd.app.api.pipeline import OnlinePipeline
from BackEnd.app.retrieval_v2.readiness import inspect_retrieval_artifacts


logger = logging.getLogger(__name__)
router = APIRouter(prefix=API_PREFIX)


@lru_cache(maxsize=1)
def _build_online_pipeline() -> OnlinePipeline:
    """Create one DB/model orchestration object for the whole API process."""

    return OnlinePipeline(
        PostgreManager(),
        selective_verifier_enabled=SELECTIVE_VERIFIER_ENABLED,
        retrieval_v2_enabled=RETRIEVAL_V2_ENABLED,
    )


def get_online_pipeline() -> OnlinePipeline:
    """FastAPI dependency kept public so tests can replace the real pipeline."""

    try:
        return _build_online_pipeline()
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Online pipeline is not configured.",
        ) from error


def close_online_pipeline() -> None:
    """Dispose the cached SQLAlchemy engine during application shutdown."""

    if _build_online_pipeline.cache_info().currsize:
        _build_online_pipeline().db_mng.engine.dispose()
        _build_online_pipeline.cache_clear()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    readiness = inspect_retrieval_artifacts(
        RETRIEVAL_V2_CORPUS_STATS_PATH,
        RETRIEVAL_V2_VIDEO_INDEX_PATH,
    )
    return HealthResponse(
        selective_verifier_enabled=SELECTIVE_VERIFIER_ENABLED,
        retrieval_v2_enabled=RETRIEVAL_V2_ENABLED,
        retrieval_v2_corpus_stats_ready=readiness.corpus_stats_ready,
        retrieval_v2_video_index_ready=readiness.video_index_ready,
        retrieval_v2_degraded_reasons=(
            list(readiness.degraded_reasons) if RETRIEVAL_V2_ENABLED else []
        ),
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    pipeline: OnlinePipeline = Depends(get_online_pipeline),
) -> QueryResponse:
    """Run one prompt through the complete online proposal pipeline."""

    try:
        return await pipeline.execute(request)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except Exception as error:
        logger.exception("Online query failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Online query failed. Check server logs for the failing stage.",
        ) from error


@router.get(
    "/frames/{frame_id}",
    name="get_frame_image",
    response_class=FileResponse,
)
async def get_frame_image(
    frame_id: str,
    pipeline: OnlinePipeline = Depends(get_online_pipeline),
) -> FileResponse:
    """Serve only media paths resolved from canonical frame IDs in PostgreSQL."""

    try:
        image_path = pipeline.frame_resolver.media_path(frame_id)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    return FileResponse(image_path)


__all__ = ["close_online_pipeline", "get_online_pipeline", "router"]
