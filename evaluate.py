"""
Load saved model and scaler, evaluate on a CSV (can be separate test CSV or same train CSV split).

Prints classification report and confusion matrix, and returns (y_true, y_pred) if called programmatically.
"""
import os
import json
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
from tensorflow.keras.models import load_model

MODEL_IN ="./serious/model_pushup_7layer_khuari_hateehoe.h5"
SCALER_IN = "./serious/scaler_pushup_7layer_khuari_hateehoe.save"
CSV_IN = "D:/Fyp/First/serious/pushup_train_khuari.csv"  # or point to a separate test CSV

def load_data(csv_path):
    df = pd.read_csv(csv_path).dropna()
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values
    y = np.array([0 if str(lbl).upper().startswith("C") else 1 for lbl in y_raw], dtype=np.int32)
    return X, y, y_raw

def evaluate(csv_in=CSV_IN, model_in=MODEL_IN, scaler_in=SCALER_IN):
    if not os.path.exists(model_in) or not os.path.exists(scaler_in):
        raise FileNotFoundError("Missing model or scaler. Run training first.")
    X, y, y_raw = load_data(csv_in)
    scaler = joblib.load(scaler_in)
    model = load_model(model_in)
    Xs = scaler.transform(X)
    preds = model.predict(Xs)
    # binary model outputs a probability; convert to 0/1
    y_pred = (preds.ravel() >= 0.5).astype(int)
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=["C (correct)", "L (incorrect)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    # --- ROC curve ---
    try:
        y_scores = preds.ravel()
        fpr, tpr, _ = roc_curve(y, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_out = os.path.join(os.path.dirname(model_in) or '.', 'roc_curve.png')
        plt.savefig(roc_out)
        print(f"Saved ROC curve to: {roc_out}")
        plt.close()
    except Exception as e:
        print(f"Could not create ROC curve: {e}")

    # --- Loss curve (if training history available) ---
    hist = _find_and_load_history(model_in)
    if hist is None:
        print("No training history file found; skipping loss plot.")
    else:
        try:
            plt.figure()
            epochs = range(1, len(hist['loss']) + 1)
            plt.plot(epochs, hist['loss'], 'b-', label='Training loss')
            if 'val_loss' in hist:
                plt.plot(epochs, hist['val_loss'], 'r--', label='Validation loss')
            plt.title('Model loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            loss_out = os.path.join(os.path.dirname(model_in) or '.', 'loss_curve.png')
            plt.savefig(loss_out)
            print(f"Saved loss curve to: {loss_out}")
            plt.close()
        except Exception as e:
            print(f"Could not create loss plot: {e}")

    return y, y_pred


def _find_and_load_history(model_in):
    """Try to find a training history near the model file in common formats.

    Looks for files like: <model>.history.json, <model>_history.json, history.json,
    <model>_history.npy, history.npy, <model>_history.pkl, history.pkl
    Returns a dict with keys 'loss' and optionally 'val_loss', or None if not found.
    """
    base_dir = os.path.dirname(model_in) or '.'
    model_base = os.path.splitext(os.path.basename(model_in))[0]
    candidates = [
        os.path.join(base_dir, model_base + '.history.json'),
        os.path.join(base_dir, model_base + '_history.json'),
        os.path.join(base_dir, 'history.json'),
        os.path.join(base_dir, model_base + '_history.npy'),
        os.path.join(base_dir, 'history.npy'),
        os.path.join(base_dir, model_base + '_history.pkl'),
        os.path.join(base_dir, 'history.pkl'),
    ]
    # also glob any file with 'history' in name in that directory
    candidates += glob.glob(os.path.join(base_dir, '*history*'))

    for fname in candidates:
        if not os.path.exists(fname):
            continue
        try:
            if fname.endswith('.json'):
                with open(fname, 'r') as f:
                    data = json.load(f)
                # Keras history objects saved as {'loss': [...], 'val_loss': [...], ...}
                if isinstance(data, dict) and 'loss' in data:
                    return data
            elif fname.endswith('.npy'):
                arr = np.load(fname, allow_pickle=True)
                # could be an array of dict or dict saved as object
                if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                    obj = arr.tolist()[0]
                    if isinstance(obj, dict) and 'loss' in obj:
                        return obj
                elif isinstance(arr, dict) and 'loss' in arr:
                    return arr
            elif fname.endswith('.pkl') or fname.endswith('.pickle'):
                with open(fname, 'rb') as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict) and 'loss' in obj:
                    return obj
            else:
                # try to load as json first
                try:
                    with open(fname, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, dict) and 'loss' in data:
                        return data
                except Exception:
                    pass
        except Exception:
            # ignore and try next
            continue
    return None

if __name__ == "__main__":
    evaluate()