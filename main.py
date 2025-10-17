from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from PIL import Image
import io, os, sqlite3, uuid
from pathlib import Path
from starlette.staticfiles import StaticFiles
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

################# CHARGEMENT DU MODÈLE YOLO #################
BASE_DIR = Path(__file__).parent
MODEL_PATH = r"C:\Users\AbdoulayeDIALLO\Desktop\Hackathon_data_science\YOLOv8_Small_RDD.pt"

print("🧠 Chargement du modèle YOLOv8...")
try:
    MODEL = YOLO(MODEL_PATH)
    print(f"✅ Modèle YOLOv8 chargé : {os.path.basename(MODEL_PATH)}")
except Exception as e:
    print(f"⚠️ Erreur lors du chargement du modèle : {e}")
    MODEL = None
#############################################################

app = FastAPI(title="InfraPredict YOLOv8 API")

# --- CORS ---
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

# --- Création de la table SQLite ---
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
init_db()

# --- Mapping des classes YOLO vers les états métiers ---
CLASS_MAPPING = {
    "Bon_etat": "✅ Bon état",
    "Usure_legere": "⚠️ Usure légère",
    "transverse_crack": "⚠️ Usure légère",
    "Degradation_moyenne": "🛠️ Dégradation moyenne",
    "Degradation_severe": "🚧 Dégradation sévère",
    "potholes": "🚧 Dégradation sévère"
}

# --- Fonction de prédiction avec YOLOv8 ---
def predict_with_yolo(image_bytes: bytes):
    if MODEL is None:
        return {"summary": {"state": "⚠️ Modèle non chargé"}, "predictions": []}

    # Enregistrer temporairement l'image
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(image_bytes)

    # Exécuter la prédiction
    results = MODEL(temp_path)
    boxes = results[0].boxes
    # results[0].save(filename="yolo-r-0.jpg")

    detected_classes = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls)
            class_name = MODEL.names[cls_id]
            print("result box::::::", class_name)
            mapped = CLASS_MAPPING.get(class_name, f"🔎 Classe inconnue : {class_name}")
            detected_classes.append(mapped)
    else:
        detected_classes.append("Aucune anomalie détectée")

    # Déterminer la gravité maximale
    if any("Dégradation sévère" in c for c in detected_classes) or any("Classe inconnue" in c for c in detected_classes):
        summary_state = "Dégradation sévère"
    elif any("Dégradation moyenne" in c for c in detected_classes):
        summary_state = "🛠️ Dégradation moyenne"
    elif any("Usure légère" in c for c in detected_classes):
        summary_state = "Usure légère"
    else:
        summary_state = "Bon état"

    return {
        "predictions": detected_classes,
        "summary": {"state": summary_state}
    }

# --- Routes ---
@app.get("/")
def root():
    return {"message": "🚗 API YOLOv8 - Détection des dégradations routières prête !"}

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

    # Sauvegarde sur disque
    uid = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1].lower() or ".png"
    filename = f"{uid}{ext}"
    disk_path = UPLOAD_DIR / filename
    with open(disk_path, "wb") as f:
        f.write(image_bytes)

    base = str(request.base_url).rstrip("/")
    image_url = f"{base}/uploads/{filename}"

    # Prédiction avec YOLO
    result = predict_with_yolo(image_bytes)
    prediction = result.get("summary", {}).get("state")

    # ⚠️ Assure que prediction n'est jamais None pour respecter NOT NULL
    if not prediction:
        prediction = "Aucune anomalie détectée"

    # Enregistrement en base
    created_at = datetime.utcnow().isoformat()
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
        "predictions_detail": result["predictions"],
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

@app.delete("/inferences/{id}")
def delete_inference(id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT image_url FROM inferences WHERE id = ?", (id,)).fetchone()
        if not row:
            return JSONResponse(status_code=404, content={"message": "Inference introuvable."})

        image_url = row["image_url"]
        filename = os.path.basename(image_url)
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                print(f"⚠️ Erreur suppression fichier : {e}")

        conn.execute("DELETE FROM inferences WHERE id = ?", (id,))
        conn.commit()

    return {"message": f"Inference {id} supprimée avec succès."}
