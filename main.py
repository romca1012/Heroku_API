from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from PIL import Image
import io, os, sqlite3, uuid, hashlib
from pathlib import Path
from starlette.staticfiles import StaticFiles
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from . import inference as inf
import torch
import numpy as np

#################PREDICTION
# Chemin du modèle (par défaut dans API/models/best_classifier.pt)
# MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_classifier.pt")
BASE_DIR = Path(__file__).parent
MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "models" / "best_classifier.pt"))

# # Charge le modèle une seule fois au démarrage
# MODEL = None
# try:
#     # load_classifier est défini dans inference.py
#     MODEL = inf.load_classifier(MODEL_PATH)
#     print(f"Model loaded from {MODEL_PATH}")
# except Exception as e:
#     # Ne crash pas l'app si le modèle est absent — on garde fallback
#     print(f"Warning: impossible de charger le modèle: {e}")
#     MODEL = None
###TESTTTTTTTTTT
MODEL = inf.load_classifier(MODEL_PATH)  # inf.load_classifier téléchargera automatiquement depuis GitHub si le .pt est absent

###########################################


app = FastAPI(title="InfraPredict Minimal API")

# --- CORS RMC ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
DB_PATH = "infra.db"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# --- Création/Migration de la table minimale ---
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inferences (
                id TEXT PRIMARY KEY,
                image_url TEXT NOT NULL,
                address TEXT NOT NULL,
                prediction TEXT NOT NULL,
                date TEXT,
                longitude FLOAT,
                latitude FLOAT,
                created_at TEXT
            )
        """)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(inferences);").fetchall()]
        if "longitude" not in cols:
            conn.execute("ALTER TABLE inferences ADD COLUMN longitude FLOAT")
        if "latitude" not in cols:
            conn.execute("ALTER TABLE inferences ADD COLUMN latitude FLOAT")
        if "created_at" not in cols:
            conn.execute("ALTER TABLE inferences ADD COLUMN created_at TEXT")
init_db()


def predict_with_model(image_bytes: bytes):
    """
    Preprocess the image, run the loaded MODEL (already loaded at startup),
    compute probs and mapping métier using functions from inference.py.
    Returns a dict similar to inference.predict_image() result.
    Falls back to the old placeholder if MODEL is None.
    """
    # fallback placeholder if MODEL missing
    if MODEL is None:
        # ancienne logique placeholder
        h = hashlib.sha256(image_bytes).digest()
        classes = [
            "Bon état",
            "Usure légère",
            "Dégradation moyenne",
            "Dégradation sévère / panne imminente",
        ]
        return {"summary": {"state": classes[h[0] % len(classes)]}, "predictions": []}

    # use inference module helpers
    pil_img = inf._open_image(image_bytes)  # returns PIL.Image
    transform = inf.get_transforms(inf.IMG_SIZE)
    x = transform(pil_img).unsqueeze(0).to(inf.DEVICE)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_name = inf.CLASS_NAMES[pred_id]
    pred_prob = float(probs[pred_id])
    summary = inf._map_to_state(pred_name, pred_prob)

    return {
        "predictions": [
            {"class_id": pred_id, "class_name": pred_name, "confidence": round(pred_prob, 4)}
        ],
        "summary": summary
    }


# # --- Fonction placeholder (à remplacer plus tard par le vrai modèle) ---
# def placeholder_predict(image_bytes: bytes) -> str:
#     h = hashlib.sha256(image_bytes).digest()
#     classes = [
#         "Bon état",
#         "Usure légère",
#         "Dégradation moyenne",
#         "Dégradation sévère / panne imminente",
#     ]
#     return classes[h[0] % len(classes)]

# --- Routes ---
@app.get("/")
def root():
    return {"message": "API maintenance prédictive prête !"}

@app.post("/predict_image")
async def predict_image(
    request: Request,
    address: str = Form(...),
    date: str = Form(...),
    longitude: float = Form(...),
    latitude: float = Form(...),
    file: UploadFile = File(...)
):
    if not (file.content_type and file.content_type.startswith("image/")):
        return JSONResponse(status_code=400, content={"message": "Le fichier doit être une image."})

    image_bytes = await file.read()
    try:
        Image.open(io.BytesIO(image_bytes)).verify()
    except Exception:
        return JSONResponse(status_code=400, content={"message": "Image invalide."})

    # Sauvegarder image
    uid = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1].lower() or ".png"
    filename = f"{uid}{ext}"
    disk_path = UPLOAD_DIR / filename
    with open(disk_path, "wb") as f:
        f.write(image_bytes)

    # URL publique
    base = str(request.base_url).rstrip("/")
    image_url = f"{base}/uploads/{filename}"

    # # Prédiction (placeholder)
    # prediction = placeholder_predict(image_bytes)

    result = predict_with_model(image_bytes)
    # mapping to your previous single "prediction" field:
    # we want the "state" string
    prediction = result.get("summary", {}).get("state", "Unknown")

    # Date d'enregistrement (UTC)
    created_at = datetime.utcnow().isoformat()

    # Enregistrer dans la base
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO inferences (id, image_url, address, prediction, date, longitude, latitude, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (uid, image_url, address, prediction, date, longitude, latitude, created_at)
        )

    return {
        "id": uid,
        "image_url": image_url,
        "address": address,
        "prediction": prediction,
        "date": date,
        "longitude": longitude,
        "latitude": latitude,
        "prediction_raw": result,
        "created_at": created_at
    }

@app.get("/inferences")
def list_inferences():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM inferences ORDER BY rowid DESC").fetchall()
        return [dict(r) for r in rows]

@app.get("/inferences/{id}")
def get_inference(id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM inferences WHERE id = ?", (id,)).fetchone()
        if not row:
            return JSONResponse(status_code=404, content={"message": "Inference introuvable."})
        return dict(row)
@app.delete("/inferences/{id}")
def delete_inference(id: str):
    """
    Supprime une inférence de la base ainsi que l'image associée dans /uploads.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT image_url FROM inferences WHERE id = ?", (id,)).fetchone()
        if not row:
            return JSONResponse(status_code=404, content={"message": "Inference introuvable."})

        # Extraire le nom du fichier depuis image_url
        image_url = row["image_url"]
        filename = os.path.basename(image_url)
        file_path = UPLOAD_DIR / filename

        # Supprimer l'image du disque si elle existe
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                print(f"⚠️ Erreur suppression fichier : {e}")

        # Supprimer l'entrée en base
        conn.execute("DELETE FROM inferences WHERE id = ?", (id,))
        conn.commit()

    return {"message": f"Inference {id} supprimée avec succès."}

