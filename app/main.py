import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.health import router as health_router
from app.core.config import BASE_DIR, Settings
from app.core.logging import configure_logging
from app.db.models import ModelVersion
from app.db.session import Base, build_engine, build_session_factory
from app.web import initialize_state, router
from ml.inference import ModelStore


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logger = configure_logging(settings)
        engine = build_engine(settings.database_url)
        application.state.engine = engine
        application.state.session_factory = build_session_factory(engine)
        Base.metadata.create_all(engine)
        try:
            application.state.models = ModelStore(settings.artifact_dir)
        except (ValueError, OSError):
            logger.error("Ошибка загрузки ML-моделей: анализ недоступен")
            application.state.models = ModelStore(settings.log_dir / "missing-models")
        initialize_state(application)
        store = application.state.models
        application.state.models_loaded = {
            "sms": store.sms is not None,
            "url": store.url is not None,
        }
        with application.state.session_factory() as session:
            for kind, metadata in (("sms", store.sms_meta), ("url", store.url_meta)):
                for old in session.scalars(
                    select(ModelVersion).where(ModelVersion.model_type == kind)
                ):
                    old.is_active = False
                if metadata:
                    version = session.scalar(
                        select(ModelVersion).where(
                            ModelVersion.model_type == kind,
                            ModelVersion.version == metadata["version"],
                        )
                    )
                    if version is None:
                        version = ModelVersion(model_type=kind, version=metadata["version"])
                        session.add(version)
                    version.algorithm = metadata.get(
                        "algorithm", metadata.get("selected", "unknown")
                    )
                    version.metrics_json = (
                        metadata.get("test_metrics")
                        or metadata.get("test")
                        or metadata.get("validation_metrics")
                        or {}
                    )
                    version.threshold = metadata.get(
                        "threshold", metadata.get("thresholds", {}).get("t2", 0.5)
                    )
                    version.trained_at = metadata["trained_at"]
                    version.is_active = True
            session.commit()
        logger.info("Приложение запущено")
        if not all(application.state.models_loaded.values()):
            logger.warning(
                "Одна или обе ML-модели отсутствуют: анализ недоступен для соответствующего типа"
            )
        else:
            logger.info("Обе ML-модели загружены один раз при старте")
        try:
            yield
        finally:
            engine.dispose()
            logger.info("Приложение остановлено")

    application = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
    application.state.settings = settings
    application.include_router(health_router)
    application.include_router(router)
    application.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    @application.exception_handler(StarletteHTTPException)
    async def http_error(request, exception):
        detail = exception.detail
        if not isinstance(detail, dict):
            detail = {"code": "HTTP_ERROR", "message": str(detail)}
        logging.getLogger("sms_guard").warning(
            "http_error status=%s code=%s", exception.status_code, detail.get("code")
        )
        return JSONResponse({"error": detail}, status_code=exception.status_code)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exception):
        return JSONResponse(
            {"error": {"code": "VALIDATION_ERROR", "message": "Проверьте параметры запроса"}},
            status_code=400,
        )

    return application


app = create_app()
