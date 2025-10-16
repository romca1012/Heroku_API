from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import io


import requests
##

# Base dir = dossier API
BASE_DIR = Path(__file__).parent

# Dossiers
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Config modèle
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
CLASS_NAMES = ["Longitudinal cracks", "Transverse cracks", "Alligator cracks", "Potholes"]

def get_transforms(img_size=IMG_SIZE):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])



#########
def download_model(url, dest_path):
    dest_path = Path(dest_path)
    if dest_path.exists():
        return
    print(f"Downloading model from {url} …")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Model downloaded.")


def load_classifier(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        GITHUB_MODEL_URL = "https://github.com/romca1012/Heroku_API/releases/download/v1.1/best_classifier.1.pt"
        download_model(GITHUB_MODEL_URL, model_path)

    ckpt = torch.load(model_path, map_location='cpu')
    import torchvision.models as models
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(ckpt if isinstance(ckpt, dict) else ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model

def _open_image(image_bytes):
    return Image.open(image_bytes if isinstance(image_bytes, Path) else io.BytesIO(image_bytes)).convert("RGB")

def _map_to_state(class_name, prob):
    # mapping métier simplifié
    if prob < 0.35: return {"state":"Bon état"}
    if prob < 0.55: return {"state":"Usure légère"}
    if prob < 0.75: return {"state":"Dégradation moyenne"}
    return {"state":"Dégradation sévère / panne imminente"}

def predict_image(image_bytes, model_path=MODELS_DIR / "best_classifier.pt"):
    model = load_classifier(model_path)
    pil_img = _open_image(image_bytes)
    x = get_transforms()(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_id]
    return {
        "predictions": [{"class_id": pred_id, "class_name": pred_name, "confidence": float(probs[pred_id])}],
        "summary": _map_to_state(pred_name, float(probs[pred_id]))
    }



# def load_classifier(model_path):
#     model_path = Path(model_path)
#     if not model_path.exists():
#         raise FileNotFoundError(f"Model not found: {model_path}")
#     ckpt = torch.load(model_path, map_location=DEVICE)
#     import torchvision.models as models
#     model = models.resnet50(weights=None)
#     model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
#     model.load_state_dict(ckpt if isinstance(ckpt, dict) else ckpt["state_dict"])
#     model.to(DEVICE).eval()
#     return model

# def _open_image(image_bytes):
#     return Image.open(image_bytes if isinstance(image_bytes, Path) else io.BytesIO(image_bytes)).convert("RGB")

# def _map_to_state(class_name, prob):
#     # mapping métier simplifié
#     if prob < 0.35: return {"state":"Bon état"}
#     if prob < 0.55: return {"state":"Usure légère"}
#     if prob < 0.75: return {"state":"Dégradation moyenne"}
#     return {"state":"Dégradation sévère / panne imminente"}

# def predict_image(image_bytes, model_path=MODELS_DIR / "best_classifier.pt"):
#     model = load_classifier(model_path)
#     pil_img = _open_image(image_bytes)
#     x = get_transforms()(pil_img).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         logits = model(x)
#         probs = F.softmax(logits, dim=1).cpu().numpy()[0]
#     pred_id = int(np.argmax(probs))
#     pred_name = CLASS_NAMES[pred_id]
#     return {
#         "predictions": [{"class_id": pred_id, "class_name": pred_name, "confidence": float(probs[pred_id])}],
#         "summary": _map_to_state(pred_name, float(probs[pred_id]))
#     }
