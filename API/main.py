from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from PIL import Image
import io, os, sqlite3, uuid, hashlib
from pathlib import Path
from starlette.staticfiles import StaticFiles
from datetime import datetime

app = FastAPI(title="InfraPredict Minimal API")

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
                image_path TEXT NOT NULL,
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

# --- Fonction placeholder (à remplacer plus tard par le vrai modèle) ---
def placeholder_predict(image_bytes: bytes) -> str:
    h = hashlib.sha256(image_bytes).digest()
    classes = [
        "Bon état",
        "Usure légère",
        "Dégradation moyenne",
        "Dégradation sévère / panne imminente",
    ]
    return classes[h[0] % len(classes)]

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

    # Prédiction (placeholder)
    prediction = placeholder_predict(image_bytes)

    # Date d'enregistrement (UTC)
    created_at = datetime.utcnow().isoformat()

    # Enregistrer dans la base
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO inferences (id, image_path, address, prediction, date, longitude, latitude, created_at) "
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
