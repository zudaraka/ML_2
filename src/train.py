# src/train.py
"""
train.py
- Tries to import `build_model(input_shape, num_classes)` from src/model_defs.py
  (or root model_defs.py) if present (so it can use your exact notebook architecture).
- Otherwise falls back to a MobileNetV2-based model.
- Trains, logs to MLflow, saves model and class_map.json in outputs/.
"""
import os
import json
import mlflow
import mlflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import argparse

OUTPUT_DIR = "outputs"
MODEL_KERAS = os.path.join(OUTPUT_DIR, "model.keras")
CLASS_MAP_PATH = os.path.join(OUTPUT_DIR, "class_map.json")

def try_import_user_model():
    """
    Try to import a user-provided build_model function from src/model_defs.py
    or model_defs.py at repo root. If found, return it; otherwise return None.
    """
    import importlib, sys, os
    candidates = ["src.model_defs", "model_defs"]
    for name in candidates:
        try:
            spec = importlib.import_module(name)
            if hasattr(spec, "build_model"):
                print(f"Using build_model from {name}")
                return spec.build_model
        except Exception:
            continue
    return None

def default_build_model(input_shape=(224,224,3), num_classes=2):
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def get_generators(data_dir, img_size=(224,224), batch_size=32):
    """
    Returns train and validation generators.
    Uses MobileNetV2's preprocess_input so inputs match pretrained base expectations.
    Adds light augmentation for the training generator to help small datasets.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # training augmentation + MobileNetV2 preprocessing
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        horizontal_flip=True,
        zoom_range=0.10
    )

    # validation: only preprocessing
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    return train_gen, val_gen

def save_class_map(generator, path=CLASS_MAP_PATH):
    # generator.class_indices maps class_name -> index
    class_indices = generator.class_indices
    # invert to index -> class_name
    inv = {str(v): k for k, v in class_indices.items()}
    with open(path, "w") as f:
        json.dump(inv, f, indent=2)
    print(f"Saved class map to {path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data", help="root data folder with train/val")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--run-name", type=str, default="dl_mlop_run")
    return p.parse_args()

def main():
    args = parse_args()
    data_dir = args.data_dir
    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mlflow.set_experiment("DL_MLOps_CW2")
    with mlflow.start_run(run_name=args.run_name):
        train_gen, val_gen = get_generators(data_dir, img_size=img_size, batch_size=batch_size)
        num_classes = train_gen.num_classes
        print(f"Found {num_classes} classes: {train_gen.class_indices}")

        # try user model
        user_builder = try_import_user_model()
        if user_builder is not None:
            try:
                model = user_builder(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
                print("Using user-provided model builder.")
            except Exception as e:
                print("Failed to build using user builder; falling back to default. Error:", e)
                model = default_build_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
        else:
            model = default_build_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)

        # Log params
        mlflow.log_param("img_size", args.img_size)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("run_name", args.run_name)

        # train
        history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs)

        # save model (native keras format)
        model.save(MODEL_KERAS)
        print(f"Saved model to {MODEL_KERAS}")

        # save mapping class index -> name
        save_class_map(train_gen, path=CLASS_MAP_PATH)

        # log metrics and model to mlflow
        final_val_acc = float(history.history.get("val_accuracy", [0.0])[-1])
        final_val_loss = float(history.history.get("val_loss", [0.0])[-1])
        mlflow.log_metric("val_accuracy", final_val_acc)
        mlflow.log_metric("val_loss", final_val_loss)
        mlflow.keras.log_model(model, "model")

        # save small JSON of history (optional)
        try:
            with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
                json.dump(history.history, f)
            mlflow.log_artifact(os.path.join(OUTPUT_DIR, "history.json"))
        except Exception:
            pass

if __name__ == "__main__":
    main()
