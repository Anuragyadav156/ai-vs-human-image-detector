from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import traceback

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load your trained model
model = tf.keras.models.load_model("my_model.keras")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((299, 299))  # Match your model input size
    img_array = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# Serve the index page (HTML form)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API endpoint to handle prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("🛬 /predict endpoint hit")

    try:
        contents = await file.read()
        print(f"📸 Received image of size: {len(contents)} bytes")
        image = preprocess_image(contents)
        print("✅ Image preprocessed")

        prediction = model.predict(image)[0][0]
        print(f"🧠 Prediction result: {prediction}")

        label = "AI Generated" if prediction > 0.5 else "Human Created"
        confidence = round(float(prediction if prediction > 0.5 else 1 - prediction), 4)

        return JSONResponse(content={
            "label": label,
            "confidence": confidence
        })

    except Exception as e:
        print("❌ Prediction failed:", e)
        return JSONResponse(content={
            "error": "Prediction failed",
            "detail": str(e)
        }, status_code=500)
