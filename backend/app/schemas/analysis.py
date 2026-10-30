from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    inference_ready: bool
    classifier_loaded: bool
    segmentation_loaded: bool
    classifier_checkpoint_present: bool
    segmentation_checkpoint_present: bool
    message: str
    model_load_error: str | None = None
    segmentation_load_error: str | None = None


class QualityResponse(BaseModel):
    width: int
    height: int
    format: str
    blur_score: float
    is_blurry: bool


class ClassResponse(BaseModel):
    supported_plant: str
    classes: list[str]
    status: str
    message: str


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Actionable error message.")
