import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.core.errors import AppError
from app.db.session import dispose_engine, init_db


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    await init_db()
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    application = FastAPI(
        title=settings.app_name,
        version="2.0.0",
        description=(
            "Interview scheduling RAG backend powered by Google Gemini, Qdrant, "
            "SQL metadata storage, and Redis chat memory."
        ),
        lifespan=lifespan,
    )
    application.include_router(router)

    @application.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "detail": str(exc)},
        )

    return application


app = create_app()
