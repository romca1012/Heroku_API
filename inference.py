# src/inference.py
# ============================================================
# 🚀 Inférence classification + mapping métier (Smartbuilders)
# - Charge best_classifier.pt (PyTorch)
# - Prétraitement ImageNet
# - Softmax → probas classes
# - Mapping vers état métier via score = poids_classe × proba
# - Visu optionnelle (légende sur l'image)
# ============================================================

import os
import io
import cv2
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Union
import torchvision.transforms as T
import torch.nn.functional as F

# -------------------------
# Config globale
# -------------------------
BASE_DIR = Path("/Users/lorinkakahoun/Appli/sb")
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Classes (ordre identique à l'entraînement)
CLASS_NAMES = [
    "Longitudinal cracks",  # 0
    "Transverse cracks",    # 1
    "Alligator cracks",     # 2
    "Potholes"              # 3
]

NUM_CLASSES = 4

# IMG_SIZE utilisé pour la classification (modifiable via env)
IMG_SIZE = int(os.environ.get("SB_IMG_SIZE", "224"))

# Poids de risque par classe (ajuste si besoin)
CLASS_WEIGHTS = {
    "Longitudinal cracks": 0.45,   # usure légère, souvent longitudinal
    "Transverse cracks":   0.55,   # un cran au-dessus
    "Alligator cracks":    0.85,   # structurellement plus sévère
    "Potholes":            0.90    # impact direct, urgent
}

# Seuils de décision sur le score = poids * proba
# Modifie pour calibrer selon tes métriques terrain
THRESHOLDS = {
    "good": 0.35,   # < 0.35  → Bon état
    "light": 0.55,  # [0.35, 0.55) → Usure légère
    "medium": 0.75  # [0.55, 0.75) → Dégradation moyenne
    # ≥ 0.75 → Dégradation sévère
}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")


def get_transforms(img_size: int = IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])


def load_classifier(model_path: Union[str, Path]) -> torch.nn.Module:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # cas entraînement avec dict
        state_dict = ckpt["state_dict"]
        model_name = ckpt.get("model_name", "resnet50")
        img_size = ckpt.get("img_size", IMG_SIZE)
    else:
        # cas torch.save(model.state_dict())
        state_dict = ckpt
        model_name = os.environ.get("SB_MODEL", "resnet50")
        img_size = IMG_SIZE

    # Reconstruit l’architecture minimale pour chargement
    import torchvision.models as models
    if model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, NUM_CLASSES)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, NUM_CLASSES)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, NUM_CLASSES)
    else:
        # resnet50 par défaut
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, NUM_CLASSES)

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()
    return model


def _open_image(image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
    if isinstance(image_input, (str, Path)):
        return Image.open(image_input).convert("RGB")
    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    raise TypeError("image_input must be path/bytes/PIL.Image")


def _map_to_state(class_name: str, prob: float) -> Dict[str, Any]:
    weight = CLASS_WEIGHTS.get(class_name, 0.5)
    score = float(weight * prob)

    if score < THRESHOLDS["good"]:
        state = "Bon état"
        intervention = "aucune"
        priority = "none"
    elif score < THRESHOLDS["light"]:
        state = "Usure légère"
        intervention = "entretien préventif recommandé"
        priority = "low"
    elif score < THRESHOLDS["medium"]:
        state = "Dégradation moyenne"
        intervention = "maintenance nécessaire bientôt"
        priority = "medium"
    else:
        state = "Dégradation sévère / panne imminente"
        intervention = "intervention urgente"
        priority = "high"

    return {
        "state": state,
        "intervention": intervention,
        "priority": priority,
        "score": round(score, 4),
        "weight_used": weight,
    }


def predict_image(
    image_input: Union[str, Path, bytes, Image.Image],
    model_path: Union[str, Path] = MODELS_DIR / "best_classifier.pt",
    return_visual: bool = False,
) -> Dict[str, Any]:
    """
    Prédit l'état d'une infrastructure à partir d'une image.
    """
    # 1) Charge modèle
    model = load_classifier(model_path)

    # 2) Prétraitement
    pil_img = _open_image(image_input)
    transform = get_transforms(IMG_SIZE)
    x = transform(pil_img).unsqueeze(0).to(DEVICE)

    # 3) Inference
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_id]
    pred_prob = float(probs[pred_id])

    # 4) Mapping métier
    summary = _map_to_state(pred_name, pred_prob)

    result = {
        "predictions": [
            {"class_id": pred_id, "class_name": pred_name, "confidence": round(pred_prob, 4)}
        ],
        "summary": summary
    }

    # 5) (Optionnel) visuel
    if return_visual:
        # On dessine juste un bandeau texte (pas de bbox en classification)
        img = np.array(pil_img)[:, :, ::-1].copy()  # RGB->BGR pour OpenCV
        label = f"{summary['state']} | {pred_name} ({pred_prob:.2f})"
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        out_path = RESULTS_DIR / "pred_visual.png"
        cv2.imwrite(str(out_path), img)
        result["visualization_path"] = str(out_path)

    return result


def visualize_predictions(image_path: Union[str, Path], predictions: Dict[str, Any], output_path: Union[str, Path]):
    """
    Dessine un bandeau de résumé(state + classe + confiance) sur l'image.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    pred = predictions["predictions"][0]
    summary = predictions["summary"]

    label = f"{summary['state']} | {pred['class_name']} ({pred['confidence']:.2f})"
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    color = (0, 255, 0) if summary["priority"] in ("none", "low") else (0, 165, 255) if summary["priority"] == "medium" else (0, 0, 255)
    cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
