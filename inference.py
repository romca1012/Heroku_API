from pathlib import Path
from PIL import Image
import torch
import numpy as np
import io
from ultralytics import YOLO


# --- Chemins ---
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Classes métier ---
CLASS_NAMES = [
    "Bon état – aucune intervention nécessaire",
    "Usure légère – entretien préventif recommandé",
    "Dégradation moyenne – maintenance nécessaire bientôt",
    "Dégradation sévère – intervention urgente"
]

# --- Modèle YOLO ---
YOLO_MODEL_PATH = r"C:\Users\AbdoulayeDIALLO\Desktop\Hackathon_data_science\YOLOv8_Small_RDD.pt"

print("🧠 Chargement du modèle YOLOv8...")
model = YOLO(YOLO_MODEL_PATH)
print(f"✅ Modèle chargé : {Path(YOLO_MODEL_PATH).name}")


def _open_image(image_bytes):
    """Ouvre une image à partir de bytes"""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _map_to_state(detected_classes):
    """
    Map YOLOv8 class detections vers une catégorie métier
    """
    if not detected_classes:
        return {"state": CLASS_NAMES[0]}  # Bon état

    detected_lower = [c.lower() for c in detected_classes]

    if any("pothole" in c for c in detected_lower):
        return {"state": CLASS_NAMES[3]}  # Dégradation sévère
    elif any("alligator" in c or "block" in c for c in detected_lower):
        return {"state": CLASS_NAMES[2]}  # Dégradation moyenne
    elif any("transverse" in c or "longitudinal" in c for c in detected_lower):
        return {"state": CLASS_NAMES[1]}  # Usure légère
    else:
        return {"state": CLASS_NAMES[0]}  # Bon état


def predict_image(image_bytes):
    """
    Effectue une prédiction avec YOLOv8 et renvoie un dictionnaire structuré.
    """
    image = _open_image(image_bytes)

    # Prédiction
    results = model(image)

    # Extraire les classes détectées
    boxes = results[0].boxes
    detected_classes = []
    confidences = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            detected_classes.append(model.names[cls_id])
            confidences.append(conf)

    # Mapper vers un état
    summary = _map_to_state(detected_classes)

    # Retourner le résultat formaté
    return {
        "predictions": [
            {"class_name": cls, "confidence": round(conf, 3)}
            for cls, conf in zip(detected_classes, confidences)
        ],
        "summary": summary
    }
