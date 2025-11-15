# Dockerfile - Use for building a Linux container (CPU TF)
FROM python:3.9-slim

WORKDIR /app

# system deps that sometimes help TensorFlow (optional small set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# copy requirements for docker build
COPY requirements.docker.txt .

# upgrade pip and install
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.docker.txt

# copy app code
COPY src/ src/
# copy outputs/model and class_map if you want the model baked into the image.
# If you do not want to bake model, remove the next line and mount outputs at runtime.
COPY outputs/ outputs/

EXPOSE 8000

CMD ["uvicorn", "src.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
