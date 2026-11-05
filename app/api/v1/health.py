from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1", tags=["health"])


class HealthResponse(BaseModel):
    status: str
    models: dict[str, bool]


@router.get("/health/", response_model=HealthResponse)
def health(request: Request):
    # Liveness of the scaffold, not readiness of the future analysis service.
    return HealthResponse(status="ok", models=request.app.state.models_loaded)
