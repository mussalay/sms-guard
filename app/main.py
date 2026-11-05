from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.v1.health import router as health_router
from app.core.config import BASE_DIR, Settings
from app.core.logging import configure_logging
from app.db.session import build_engine, build_session_factory


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logger = configure_logging(settings)
        engine = build_engine(settings.database_url)
        application.state.engine = engine
        application.state.session_factory = build_session_factory(engine)
        application.state.models_loaded = {"sms": False, "url": False}
        logger.info("Приложение запущено")
        logger.warning("ML-модели ещё не подключены: анализ недоступен (этап 2)")
        try:
            yield
        finally:
            engine.dispose()
            logger.info("Приложение остановлено")

    application = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
    application.state.settings = settings
    application.include_router(health_router)
    application.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=BASE_DIR / "templates")

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(request: Request):
        return templates.TemplateResponse(
            request=request, name="index.html", context={"app_name": settings.app_name}
        )

    return application


app = create_app()
