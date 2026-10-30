from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw

from .models import ModelRegistry, ModelUnavailableError


MIN_CANDIDATE_SCORE = 0.60
MAX_WINDOWS = 36
MAX_CANDIDATES = 8
SOURCES = [
    {
        "name": "American Phytopathological Society",
        "title": "APS Education Center: Learning Plant Pathology",
        "role": "Plant pathology education and reviewed disease-profile resources; general reference, not a crop-specific treatment prescription.",
        "url": "https://www.apsnet.org/edcenter/Pages/default.aspx",
    },
    {
        "name": "Food and Agriculture Organization of the United Nations",
        "title": "Plant Production and Protection",
        "role": "Institutional guidance on sustainable crop production and plant protection; general reference.",
        "url": "https://www.fao.org/plant-production-protection/en/",
    },
    {
        "name": "USDA Animal and Plant Health Inspection Service",
        "title": "Plant Pests and Diseases",
        "role": "Official U.S. plant-health and pest information; consult local authorities for regulated or unusual pests.",
        "url": "https://www.aphis.usda.gov/plant-pests-diseases",
    },
]


@dataclass(frozen=True)
class Candidate:
    box: dict[str, int]
    image: str
    predicted_class: str
    confidence: float
    confidence_calibrated: bool
    needs_review: bool
    explanation: str
    visible_evidence: str
    possible_causes: list[str]
    severity: str
    potential_impact: str
    immediate_actions: list[str]
    biological_treatment: str
    chemical_treatment: str
    sources: list[dict[str, str]]


def _image_data_url(image: Image.Image, max_dimension: int = 720) -> str:
    image = image.convert("RGB").copy()
    image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=78, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _friendly_label(label: str) -> tuple[str, str]:
    plant, _, condition = label.partition("___")
    plant = plant.replace("_", " ").replace(",", "").replace("(including sour)", "").strip()
    condition = condition.replace("_", " ").strip()
    return plant or "unknown plant", condition or "unknown class"


def _possible_cause(label: str) -> list[str]:
    normalized = label.lower()
    categories = (
        ("spider mite", "The model class is associated with a mite pest."),
        ("bacterial", "The model class is associated with a bacterial disease label."),
        ("virus", "The model class is associated with a viral disease label."),
        ("mildew", "The model class is associated with a mildew disease label."),
        ("rust", "The model class is associated with a rust disease label."),
        ("blight", "The model class is associated with a blight disease label."),
        ("scab", "The model class is associated with a scab disease label."),
        ("rot", "The model class is associated with a rot disease label."),
        ("leaf spot", "The model class is associated with a leaf-spot disease label."),
        ("leaf mold", "The model class is associated with a leaf-mold disease label."),
        ("esca", "The model class is associated with an Esca disease label."),
        ("scorch", "The model class is associated with a leaf-scorch disease label."),
        ("citrus greening", "The model class is associated with a citrus-greening disease label."),
    )
    for token, description in categories:
        if token in normalized:
            return [description]
    return ["The model selected a disease class, but the causal agent is not established by this image-only result."]


def _candidate_explanation(
    label: str,
    score: float,
    confidence_calibrated: bool = False,
) -> tuple[str, str]:
    plant, condition = _friendly_label(label)
    explanation = (
        f"The classifier's top class pairs the dataset plant label {plant} with the disease label {condition}. "
        "This is a PlantVillage-trained model prediction, not independent botanical identification or a confirmed diagnosis."
    )
    score_description = (
        "temperature-scaled score using held-out PlantVillage calibration data"
        if confidence_calibrated
        else "uncalibrated model score"
    )
    visible_evidence = (
        f"This crop was selected because its {score_description} for the named class was {score:.1%}. "
        "The system does not segment or independently verify spots, lesions, insects, or other symptoms."
    )
    return explanation, visible_evidence


def _guidance(label: str) -> tuple[list[str], str, str]:
    immediate = [
        "Keep this observation separate from healthy plants when practical, and photograph the whole plant plus close views of both leaf surfaces.",
        "Record the plant species, location, recent weather, watering, and when symptoms first appeared; inspect nearby plants for similar symptoms.",
        "Clean hands and tools between plants. Avoid moving potentially infested plant material until the cause is clearer.",
        "Ask a local university/cooperative extension service or plant diagnostic clinic to confirm the diagnosis before treating.",
    ]
    if "spider_mites" in label.lower() or "spider mite" in label.lower():
        immediate.insert(1, "Inspect leaf undersides with a hand lens for moving mites, fine webbing, and stippling; the classifier output alone does not confirm mites.")

    biological = (
        "No biological control is prescribed from this image-only prediction. A qualified local adviser should first confirm the pest/pathogen and whether a locally registered biological control fits."
    )
    chemical = (
        "No pesticide is selected from this result. If a professional confirms a target, use only a product registered for that crop and location, follow its label and pre-harvest/re-entry intervals, and comply with local regulations."
    )
    return immediate, biological, chemical


def _windows(image: Image.Image, mode: str) -> list[tuple[int, int, int, int]]:
    width, height = image.size
    short_side = min(width, height)
    fraction = 0.52 if mode == "single_leaf" else 0.38
    side = min(short_side, max(128, round(short_side * fraction)))
    stride = max(64, round(side * 0.62))

    def positions(length: int) -> list[int]:
        if length <= side:
            return [0]
        result = list(range(0, length - side + 1, stride))
        end = length - side
        if result[-1] != end:
            result.append(end)
        return result

    xs, ys = positions(width), positions(height)

    def evenly_sample(values: list[int], limit: int) -> list[int]:
        if len(values) <= limit:
            return values
        indices = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
        return [values[index] for index in indices]

    max_per_axis = int(MAX_WINDOWS**0.5)
    xs = evenly_sample(xs, max_per_axis)
    ys = evenly_sample(ys, max_per_axis)
    return [(x, y, x + side, y + side) for y in ys for x in xs]


def _intersection_over_union(a: dict[str, int], b: dict[str, int]) -> float:
    left = max(a["left"], b["left"])
    top = max(a["top"], b["top"])
    right = min(a["right"], b["right"])
    bottom = min(a["bottom"], b["bottom"])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (a["right"] - a["left"]) * (a["bottom"] - a["top"])
    area_b = (b["right"] - b["left"]) * (b["bottom"] - b["top"])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _overlay(image: Image.Image, candidates: list[Candidate]) -> str:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    for index, candidate in enumerate(candidates, start=1):
        box = candidate.box
        draw.rectangle(
            (box["left"], box["top"], box["right"], box["bottom"]),
            outline=(232, 90, 55),
            width=max(3, round(min(image.size) / 300)),
        )
        draw.text((box["left"] + 5, box["top"] + 5), str(index), fill=(255, 255, 255), stroke_width=2, stroke_fill=(120, 25, 15))
    return _image_data_url(result, max_dimension=1600)


def screen_plant_image(
    image: Image.Image,
    model_registry: ModelRegistry,
    mode: str = "whole_plant",
) -> dict[str, Any]:
    if mode not in {"single_leaf", "whole_plant"}:
        raise ValueError("Mode must be 'single_leaf' or 'whole_plant'.")
    if model_registry.classifier is None or not model_registry.class_names:
        raise RuntimeError("A trained classifier checkpoint and class mapping are required for image screening.")

    candidates: list[Candidate] = []
    checked = 0
    overall_prediction = model_registry.predict_class(image) if mode == "single_leaf" else None
    for left, top, right, bottom in _windows(image, mode):
        crop = image.crop((left, top, right, bottom))
        if min(crop.size) < 128:
            continue
        checked += 1
        prediction = model_registry.predict_class(crop)
        label = str(prediction["predicted_class"])
        confidence = float(prediction["confidence"])
        if "healthy" in label.lower() or confidence < MIN_CANDIDATE_SCORE:
            continue

        confidence_calibrated = bool(
            prediction.get(
                "confidence_calibrated",
                getattr(model_registry, "calibration_report", None),
            )
        )
        needs_review = bool(
            prediction.get(
                "needs_review",
                confidence < float(getattr(model_registry, "review_threshold", MIN_CANDIDATE_SCORE)),
            )
        )
        explanation, evidence = _candidate_explanation(label, confidence, confidence_calibrated)
        actions, biological, chemical = _guidance(label)
        box = {"left": left, "top": top, "right": right, "bottom": bottom}
        candidates.append(
            Candidate(
                box=box,
                image=_image_data_url(crop),
                predicted_class=label,
                confidence=confidence,
                confidence_calibrated=confidence_calibrated,
                needs_review=needs_review,
                explanation=explanation,
                visible_evidence=evidence,
                possible_causes=_possible_cause(label),
                severity="Not assessed from image",
                potential_impact="Cannot be estimated reliably from this crop or classifier score.",
                immediate_actions=actions,
                biological_treatment=biological,
                chemical_treatment=chemical,
                sources=SOURCES,
            )
        )

    candidates.sort(key=lambda item: item.confidence, reverse=True)
    deduplicated: list[Candidate] = []
    for candidate in candidates:
        if any(
            candidate.predicted_class == existing.predicted_class
            and _intersection_over_union(candidate.box, existing.box) >= 0.55
            for existing in deduplicated
        ):
            continue
        deduplicated.append(candidate)
        if len(deduplicated) == MAX_CANDIDATES:
            break

    confidence_calibrated = bool(getattr(model_registry, "calibration_report", None))
    review_required = bool(
        (overall_prediction and overall_prediction.get("needs_review"))
        or any(candidate.needs_review for candidate in deduplicated)
    )
    explainability = None
    explainability_error = None
    gradcam = getattr(model_registry, "gradcam", None)
    if callable(gradcam):
        attribution_image = image
        attribution_class = (
            str(overall_prediction["predicted_class"])
            if overall_prediction
            else None
        )
        if mode == "whole_plant" and deduplicated:
            top_region = deduplicated[0]
            box = top_region.box
            attribution_image = image.crop(
                (box["left"], box["top"], box["right"], box["bottom"])
            )
            attribution_class = top_region.predicted_class
        if attribution_class is not None:
            try:
                explainability = gradcam(attribution_image, attribution_class)
            except (ModelUnavailableError, RuntimeError, ValueError) as exc:
                explainability_error = str(exc)
    elif mode == "single_leaf":
        explainability_error = "Grad-CAM is unavailable for the active model."

    calibration_note = (
        "Scores are temperature-scaled using a held-out PlantVillage calibration split; this does not establish field-photo accuracy or calibrated probabilities for out-of-domain images."
        if confidence_calibrated
        else "Confidence scores are uncalibrated; predictions are best matches among the trained labels and are not an OOD detector."
    )
    limitations = [
        f"Only candidate regions whose model score met the screening threshold are shown. A region crop is not a lesion mask; the exact affected pixels are not localized. {calibration_note}",
        "The classifier was trained on PlantVillage leaf images. Natural-background, whole-plant, unsupported-species, or unknown-disease photos may be outside its training domain and are not reliably detected as out-of-distribution.",
        "No candidates does not prove the plant is healthy. Severity, spread, and biological or chemical treatment are not determined by this model.",
        "The listed references are general diagnostic resources, not disease-specific treatment trials. Follow local extension advice and pesticide labels.",
    ]
    if explainability_error:
        limitations.append(f"Grad-CAM attribution was not produced: {explainability_error}")
    return {
        "mode": mode,
        "regions_checked": checked,
        "candidate_regions": [candidate.__dict__ for candidate in deduplicated],
        "candidate_count": len(deduplicated),
        "overlay": _overlay(image, deduplicated),
        "overall_prediction": overall_prediction,
        "explainability": explainability,
        "candidate_score_threshold": MIN_CANDIDATE_SCORE,
        "confidence_calibrated": confidence_calibrated,
        "review_threshold": float(getattr(model_registry, "review_threshold", MIN_CANDIDATE_SCORE)),
        "review_required": review_required,
        "unknown_detection_validated": False,
        "limitations": limitations,
        "sources": SOURCES,
        "professional_attribution": (
            "The cited pages are institutional resources. No individual scientist or publication author is attributed because the current sources are general references and do not support a disease-specific attribution."
        ),
    }


def screen_whole_plant(image: Image.Image, model_registry: ModelRegistry, grid_size: int = 3) -> dict[str, Any]:
    """Compatibility wrapper for earlier callers of whole-plant screening."""
    del grid_size
    return screen_plant_image(image, model_registry, "whole_plant")
