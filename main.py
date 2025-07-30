from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="my_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((299, 299))  # Match your model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess_image(contents)

        # Set the tensor to the input data
        interpreter.set_tensor(input_details[0]['index'], image)

        # Run inference
        interpreter.invoke()

        # Get prediction result
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        label = "AI Generated" if prediction > 0.5 else "Human Created"
        confidence = round(float(prediction if prediction > 0.5 else 1 - prediction), 4)

        return JSONResponse(content={
            "label": label,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(content={
            "error": "Prediction failed",
            "detail": str(e)
        }, status_code=500)
