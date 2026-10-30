from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .preprocessing.quality import InvalidImageError, decode_image
from .schemas.analysis import ClassResponse, HealthResponse, QualityResponse
from .services.models import ModelUnavailableError, model_registry
from .services.screening import screen_plant_image


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_registry.load()
    yield


app = FastAPI(
    title=settings.app_name,
    description="CPU-first plant disease and visible affected-area analysis API.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, object]:
    """Return operational and checkpoint status without claiming untrained models."""
    classifier_checkpoint = settings.model_dir / "classifier.pt"
    segmentation_checkpoint = settings.model_dir / "segmentation.pt"
    model_health = model_registry.health()
    return {
        "status": "ok",
        "models_loaded": all(model_health.values()),
        "inference_ready": model_health["classifier_loaded"],
        **model_health,
        "classifier_checkpoint_present": classifier_checkpoint.exists(),
        "segmentation_checkpoint_present": segmentation_checkpoint.exists(),
        "model_load_error": model_registry.model_load_error,
        "segmentation_load_error": model_registry.segmentation_load_error,
        "message": (
            "Ready for classification and segmentation."
            if model_health["classifier_loaded"] and model_health["segmentation_loaded"]
            else "Classifier ready; segmentation is unavailable, so exact lesion masks and affected-area estimates are not produced."
            if model_health["classifier_loaded"]
            else f"Classifier is unavailable: {model_registry.model_load_error}"
            if model_registry.model_load_error
            else "A trained classifier checkpoint is required for inference."
        ),
    }


@app.get("/classes", response_model=ClassResponse)
def classes() -> dict[str, object]:
    """Return the class mapping saved by the training workflow."""
    plant_names = sorted({
        class_name.partition("___")[0].replace("_", " ").replace(",", "").strip()
        for class_name in (model_registry.class_names or [])
    })
    return {
        "supported_plant": ", ".join(plant_names) if plant_names else "",
        "classes": model_registry.class_names or [],
        "status": "ready" if model_registry.class_names else "not_configured",
        "message": (
            "Classes loaded from the trained model mapping."
            if model_registry.class_names
            else "Classes will be discovered during dataset preparation."
        ),
    }


@app.post("/quality", response_model=QualityResponse)
async def quality(file: UploadFile = File(...)) -> QualityResponse:
    contents = await file.read()
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
    try:
        _, result = decode_image(contents)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return QualityResponse(**result.__dict__)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    contents = await file.read()
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
    try:
        image, quality_result = decode_image(contents)
        prediction = model_registry.predict_class(image)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"quality": quality_result.__dict__, "model": prediction}


@app.post("/screen-plant")
async def screen_plant(
    file: UploadFile = File(...),
    mode: str = Form(default="whole_plant"),
) -> dict[str, object]:
    contents = await file.read()
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
    try:
        image, quality_result = decode_image(contents)
        screening = screen_plant_image(image, model_registry, mode)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"quality": quality_result.__dict__, "screening": screening}
