# src/predict_api.py
"""
Prediction API (final)
- Loads outputs/model.keras or outputs/model.h5
- Loads outputs/class_map.json (converts keys to ints)
- /health endpoint included
- /predict accepts form-file uploads (images) and returns:
  {"class":"<name>", "class_index": 0, "confidence": 0.8181}
- Improvements: TF log level suppression, logging, rounded confidence,
  robust tempfile cleanup and better error messages.
"""

import os
import json
import tempfile
import traceback
import logging
from typing import Optional

# Reduce TensorFlow logging noise early
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=info, 2=warning, 3=error

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("predict_api")

# Paths
MODEL_KERAS = "outputs/model.keras"
MODEL_H5 = "outputs/model.h5"
CLASS_MAP = "outputs/class_map.json"

app = FastAPI(title="Prediction API")
model: Optional[tf.keras.Model] = None
class_map: Optional[dict] = None


def find_and_load_model():
    global model
    try:
        if os.path.exists(MODEL_KERAS):
            model = tf.keras.models.load_model(MODEL_KERAS)
            logger.info("Loaded model from %s", MODEL_KERAS)
        elif os.path.exists(MODEL_H5):
            model = tf.keras.models.load_model(MODEL_H5)
            logger.info("Loaded model from %s", MODEL_H5)
        else:
            model = None
            logger.warning("No saved model found in outputs/")
    except Exception as e:
        model = None
        logger.exception("Error loading model: %s", e)
        traceback.print_exc()


def load_class_map():
    global class_map
    try:
        if os.path.exists(CLASS_MAP):
            with open(CLASS_MAP, "r") as f:
                loaded = json.load(f)
            # convert keys to ints if they are strings
            class_map = {int(k): v for k, v in loaded.items()}
            logger.info("Loaded class map from %s: %s", CLASS_MAP, class_map)
        else:
            class_map = None
            logger.warning("No class_map.json found in outputs/")
    except Exception as e:
        class_map = None
        logger.exception("Error loading class_map: %s", e)
        traceback.print_exc()


def preprocess_image_file(path: str, target_size=(224, 224)) -> np.ndarray:
    """
    Load image from path, convert to RGB and return batched numpy array scaled to [0,1].
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.on_event("startup")
def startup_event():
    find_and_load_model()
    load_class_map()


@app.get("/", response_class=HTMLResponse)
def read_root():
    status = "OK" if model is not None else "MODEL NOT LOADED"
    html = f"<html><body><h1>Prediction API</h1><p>Status: {status}</p><p>Visit <a href='/docs'>/docs</a> for the interactive API.</p></body></html>"
    return HTMLResponse(content=html)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "class_map_loaded": class_map is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        logger.error("Predict called but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Basic content-type check (client-supplied header)
    if not (file.content_type and file.content_type.startswith("image")):
        logger.warning("Upload rejected due to content-type=%s", file.content_type)
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    tmp_name = None
    try:
        # write to a secure temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp_name = tmp.name
            content = await file.read()
            tmp.write(content)

        # validate it's an image
        try:
            _ = Image.open(tmp_name)
            _.close()
        except UnidentifiedImageError:
            logger.warning("Uploaded file is not a valid image")
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

        # preprocess and predict
        x = preprocess_image_file(tmp_name, target_size=(224, 224))
        preds = model.predict(x)

        preds = np.asarray(preds)
        if preds.ndim == 1:
            # single-output model (binary/logit style) — treat value as confidence for class 1
            conf = float(preds[0])
            idx = 1 if conf >= 0.5 else 0
        elif preds.ndim == 2 and preds.shape[0] == 1:
            probs = preds[0]
            idx = int(np.argmax(probs))
            conf = float(np.max(probs))
        else:
            logger.error("Unexpected prediction shape: %s", preds.shape)
            raise HTTPException(status_code=500, detail=f"Unexpected prediction shape: {preds.shape}")

        # round confidence for nicer output
        conf = round(conf, 4)
        name = class_map.get(idx, str(idx)) if class_map else str(idx)

        logger.info("Predicted class %s (idx=%d) conf=%s", name, idx, conf)
        return JSONResponse({"class": name, "class_index": idx, "confidence": conf})

    except HTTPException:
        # re-raise FastAPI HTTP errors
        raise
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # cleanup temp file
        try:
            if tmp_name and os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            logger.debug("Failed to remove temp file %s", tmp_name)


@app.get("/favicon.ico")
def favicon():
    # silent 204 for browsers
    return Response(status_code=204)


if __name__ == "__main__":
    # Run directly for development/demo:
    # source venv/bin/activate
    # python -m uvicorn src.predict_api:app --host 0.0.0.0 --port 8000
    import uvicorn
    uvicorn.run("src.predict_api:app", host="0.0.0.0", port=8000, reload=False)
