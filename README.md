# DL-MLOps-CW2

**National Institute of Business Management**  
Higher National Diploma in Data Science 24.2f  
Machine Learning 02 – Course Work 2  

**Title:** *Deep Learning Project using MLOps*

---

## 📌 Project Summary

This project implements a Deep Learning image classification model and deploys it using MLOps principles.

It includes:

- Model training pipeline  
- Data preprocessing  
- MLflow experiment tracking & model versioning  
- CI workflow  
- FastAPI model deployment  
- Optional Dockerization  
- Model monitoring & experiment logging  

This satisfies all requirements of the CW2 assignment.

---

## 📁 Repository Structure

```
DL-MLOPS-CW2/
├── data/                     # train/val image folders (not pushed to GitHub)
├── outputs/                  # model.keras, class_map.json, history.json (gitignored)
├── src/
│   ├── train.py              # training + MLflow pipeline
│   ├── predict_api.py        # FastAPI model server
│   └── model_defs.py         # custom model builder (optional)
├── mlruns/                   # MLflow experiment logs
├── .github/workflows/ci.yml  # GitHub Actions CI workflow
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔧 Setup Instructions

### 1️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **For macOS M1/M2 users:**

```bash
pip install tensorflow-macos tensorflow-metal
```

---

## 🗂️ Dataset Structure

Your dataset should follow this structure:

```
data/
├── train/
│   ├── classA/
│   └── classB/
└── val/
    ├── classA/
    └── classB/
```

---

## 🧠 Model Training (with MLflow)

Run training:

```bash
source venv/bin/activate
python src/train.py --data-dir data --epochs 5 --batch-size 8 --img-size 224 --run-name final_run
```

This will:

✔ Train the model  
✔ Save model → `outputs/model.keras`  
✔ Save class mapping → `outputs/class_map.json`  
✔ Log experiments into **MLflow**

---

## 📊 View MLflow Dashboard

Start MLflow UI:

```bash
mlflow ui --backend-store-uri mlruns --port 5000
```

Open in browser:  
👉 http://127.0.0.1:5000

You will see:

- Training metrics  
- Parameters  
- Model versions  
- Artifacts  

---

## 🚀 Run FastAPI Model Server

### Start API

```bash
python -m uvicorn src.predict_api:app --host 127.0.0.1 --port 8000
```

### Health Check

```bash
curl -i http://127.0.0.1:8000/health
```

### Make a Prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-F "file=@data/val/classA/classA_0.png" -i
```

---

## 🔁 CI/CD Pipeline (GitHub Actions)

Included workflow:

```
.github/workflows/ci.yml
```

It performs:

- Code checkout  
- Dependency installation  
- Quick CI smoke test  

---

## 🐳 Docker (Optional Containerization)

Build the Docker image:

```bash
docker build -t dl-mlops-cw2 .
```

Run the container:

```bash
docker run -p 8000:8000 dl-mlops-cw2
```

---

## 📈 Model Monitoring

- MLflow is used to track all experiments  
- Compare validation accuracy across runs to detect model drift  
- Re-train model if accuracy drops on new data  

---

## 📘 What To Submit

✔ Jupyter Notebook report  
✔ GitHub repository link  
✔ 5-minute demonstration video  
✔ This README.md  
✔ All code + workflows  

---

## 📞 Notes

If TensorFlow gives macOS errors:

```bash
pip install tensorflow-macos tensorflow-metal
```

---
