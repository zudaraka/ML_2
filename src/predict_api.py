"""
predict_api.py
- Loads outputs/model.keras (Keras native format) or outputs/model.h5 if present
- Loads outputs/class_map.json to map index -> class_name
- /health endpoint included
- /predict accepts form-file upload and returns {"class": "<name>", "confidence": 0.92}
"""
import os, json
from fastapi import FastAPI, UploadFile, File
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import uvicorn

MODEL_KERAS = "outputs/model.keras"
MODEL_H5 = "outputs/model.h5"  # older files
CLASS_MAP = "outputs/class_map.json"

app = FastAPI()
model = None
class_map = None

def find_and_load_model():
    global model
    if os.path.exists(MODEL_KERAS):
        model = load_model(MODEL_KERAS)
        print(f"Loaded model from {MODEL_KERAS}")
    elif os.path.exists(MODEL_H5):
        model = load_model(MODEL_H5)
        print(f"Loaded model from {MODEL_H5}")
    else:
        model = None
        print("No saved model found in outputs/")

def load_class_map():
    global class_map
    if os.path.exists(CLASS_MAP):
        with open(CLASS_MAP, "r") as f:
            class_map = json.load(f)  # keys are string indices
        # ensure keys are ints in memory
        class_map = {int(k): v for k, v in class_map.items()}
        print(f"Loaded class map from {CLASS_MAP}: {class_map}")
    else:
        class_map = None
        print("No class_map.json found in outputs/")

def preprocess_image_bytes(path, target_size=(224,224)):
    img = load_img(path, target_size=target_size)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.on_event("startup")
def startup():
    find_and_load_model()
    load_class_map()

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "class_map_loaded": class_map is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded."}
    contents = await file.read()
    tmp = "tmp_input.jpg"
    with open(tmp, "wb") as f:
        f.write(contents)
    x = preprocess_image_bytes(tmp, target_size=(224,224))
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    name = class_map.get(idx, str(idx)) if class_map else str(idx)
    return {"class": name, "class_index": idx, "confidence": conf}

if __name__ == "__main__":
    uvicorn.run("src.predict_api:app", host="0.0.0.0", port=8000, reload=False)
