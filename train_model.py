"""
Train the 7-layer model on extracted CSV data (pushup_train.csv).

Outputs:
- saved model file: model_pushup_7layer.h5
- saved scaler: scaler_pushup.save (joblib)
- a small JSON summary printed to stdout

Adjust paths / hyperparams at top of file.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import joblib
from model import build_model

CSV_PATH = "D:/Fyp/First/serious/pushup_train_khuari.csv"   # CSV produced by extractor
MODEL_OUT = "./serious/model_pushup_7layer_khuari_hateehoe.h5"
SCALER_OUT = "./serious/scaler_pushup_7layer_khuari_hateehoe.save"
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 80

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # drop rows with NaNs if any
    df = df.dropna()
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values
    # map labels: 'C' -> 0, 'L' -> 1
    y = np.array([0 if str(lbl).upper().startswith("C") else 1 for lbl in y_raw], dtype=np.int32)
    return X, y

def train():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found. Run extractor first.")
    X, y = load_data(CSV_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_OUT)
    input_dim = X_train.shape[1]
    model = build_model(input_dim, n_classes=2, dropout_rate=0.2)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True),
    ]
    # For binary output, use y as-is
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)
    # save final model if checkpoint didn't already
    model.save(MODEL_OUT)
    # report simple metrics
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Saved model to {MODEL_OUT}")
    print(f"Test loss: {loss:.4f}  Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()