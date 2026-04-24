# main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf
import json, random, os




app = FastAPI()


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CONFIG
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "skin_tone_savedmodel_op.keras")
PALETTE_PATH = os.path.join(BASE_DIR, "model", "palettes_50_colors.json")
IMG_SIZE = (224, 224)



# Load model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded")

# Load palettes
with open(PALETTE_PATH, "r") as f:
    palettes = json.load(f)

# Class mapping
idx2class = {0: 'Black', 1: 'Brown', 2: 'White'}




# --------------------------
# FACE DETECTION (FAST)
# --------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)




def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None, False

    h_img, w_img = image.shape[:2]

    # Pick largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # 🔴 1. Reject too small faces
    if w < w_img * 0.2 or h < h_img * 0.2:
        return None, False

    # 🔴 2. Center check (soft)
    cx = x + w / 2
    cy = y + h / 2

    if not (w_img * 0.2 < cx < w_img * 0.8 and h_img * 0.2 < cy < h_img * 0.8):
        return None, False

    # 🔴 3. Face should occupy enough area
    face_area = w * h
    img_area = w_img * h_img

    if face_area / img_area < 0.05:
        return None, False

    # ✅ Crop with controlled padding
    pad_w = int(0.2 * w)
    pad_h = int(0.15 * h)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_img, x + w + pad_w)
    y2 = min(h_img, y + h + pad_h)

    face = image[y1:y2, x1:x2]

    return face, True




# --------------------------
# PREPROCESS
# --------------------------
def preprocess(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return img


# --------------------------
# PREDICTION
# --------------------------
def predict_image(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    face, detected = detect_face(image)

    if not detected:
        return None, 0.0, {}, False

    img = preprocess(face)
    inp = np.expand_dims(img, 0)

    probs = model.predict(inp)[0]
    idx = int(np.argmax(probs))
    label = idx2class[idx]

    image = cv2.resize(image, (300, 300))

    return label, float(probs[idx]), {
        idx2class[i]: float(probs[i]) for i in range(len(probs))
    }, True


# --------------------------
# PALETTE
# --------------------------
def get_palette(label):
    key = next((k for k in palettes if k.lower() == label.lower()), None)
    if key is None:
        key = list(palettes.keys())[0]
    return random.sample(palettes[key], 6)


# --------------------------
# API ROUTE
# --------------------------
@app.post("/predict")

async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()

    label, prob, all_probs, face_detected = predict_image(img_bytes)

    if not face_detected:
        return {
            "face_detected": False,
            "colors": []
        }
    
    print("API HIT")

    colors = get_palette(label)

    return {
        "face_detected": True,
        "predicted_skin_tone": label,
        "prob": prob,
        "all_probs": all_probs,
        "colors": colors
    }


