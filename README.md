# DL-MLOPS-CW2
 
 **Title:** *Deep Learning Project Using MLOps*

---

## 📌 Project Overview

This project implements an **image classification system** using Deep Learning and integrates full **MLOps practices**, including:

- Automated model training pipeline  
- Data preprocessing (normalization & augmentation)  
- Experiment tracking & model versioning via **MLflow**  
- Custom model architecture support  
- Continuous Integration with **GitHub Actions**  
- Deployment using **FastAPI**  
- Optional **Docker containerization**  
- Model monitoring and drift checking  

The implementation satisfies **all requirements** of the CW2 assignment.

---

## 📁 Repository Structure

DL-MLOPS-CW2/
├── data/                     # Training/validation datasets (not uploaded)
├── outputs/                  # Saved model + metadata (gitignored)
├── src/
│   ├── train.py              # Training pipeline with MLflow
│   ├── predict_api.py        # FastAPI inference server
│   └── model_defs.py         # Custom model builder
├── mlruns/                   # MLflow experiment tracking (gitignored)
├── .github/workflows/ci.yml  # CI pipeline
├── requirements.txt
├── requirements.docker.txt
├── Dockerfile
├── lab2_3_dl_hnd242f_08,33,34.ipynb
└── README.md

---

## 🔧 Environment Setup

### 1️⃣ Create Virtual Environment
python3 -m venv venv
source venv/bin/activate

### 2️⃣ Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

### ⚠️ For M1/M2 Macs
pip install tensorflow-macos tensorflow-metal

---

## 🗂️ Dataset Format

data/
├── train/
│   ├── classA/
│   └── classB/
└── val/
    ├── classA/
    └── classB/

---

## 🧠 Model Training (MLflow Integrated)

Run training:
python src/train.py --data-dir data --epochs 5 --batch-size 8 --img-size 224 --run-name final_run

This saves:
- outputs/model.keras
- outputs/class_map.json
- outputs/history.json

And logs to MLflow.

---

## 📊 Start MLflow Dashboard

mlflow ui --backend-store-uri mlruns --port 5000

Open:
http://127.0.0.1:5000

---

## 🚀 FastAPI Deployment

Start server:
uvicorn src.predict_api:app --host 127.0.0.1 --port 8000

Health check:
curl -i http://127.0.0.1:8000/health

Prediction:
curl -X POST "http://127.0.0.1:8000/predict" \
-F "file=@data/val/classA/classA_0.png" -i

---

## 🔁 CI/CD (GitHub Actions)

Workflow file:
.github/workflows/ci.yml

Runs:
- Dependency installation  
- Basic tests  
- CI smoke check  

---

## 🐳 Docker Deployment (Optional)

Build the image:
docker build -t dl-mlops-cw2 .

Run the container:
docker run -p 8000:8000 dl-mlops-cw2

---

## 📈 Model Monitoring

- Compare experiment runs in MLflow  
- Detect model drift  
- Retrain model when accuracy drops  

---

## 📘 Submission Checklist

✔ Jupyter Notebook report  
✔ GitHub repo link  
✔ 5-minute demonstration video  
✔ README.md (this file)  
✔ All source code  

---

## 💡 Mac TensorFlow Fix
pip install tensorflow-macos tensorflow-metal

---

## 🎉 Completed MLOps Workflow

This project demonstrates:

- End-to-end ML lifecycle  
- Automated pipelines  
- Deployment  
- Monitoring  
- Experiment tracking  
- Reproducibility  

Fully compliant with CW2 evaluation criteria.
