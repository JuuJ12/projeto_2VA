
import os
import time
import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from src.preprocessing import preprocess
from src.features import extract_features

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "models")



def build_feature_matrix(
    df: pd.DataFrame,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    n = len(df)

    for i, row in df.iterrows():
        try:
            imgs = preprocess(row["path"])
            feat = extract_features(imgs["bgr"])
            X_list.append(feat)
            y_list.append(row["label_idx"])
        except Exception as e:
            print(f"[classical] AVISO: erro na imagem {row['path']}: {e}")
            continue

        if verbose and (len(X_list) % 100 == 0 or len(X_list) == n):
            print(f"  [{len(X_list):>4}/{n}] features extraídas...", end="\r")

    if verbose:
        print()

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y



def train_classical(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models_dir: str = MODELS_DIR,
) -> dict:
    os.makedirs(models_dir, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    print("[classical] Treinando k-NN (k=5)...")
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    knn.fit(X_scaled, y_train)
    print(f"  Concluído em {time.time()-t0:.1f}s")

    print("[classical] Treinando SVM (kernel RBF, C=10)...")
    t0 = time.time()
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=False, random_state=42)
    svm.fit(X_scaled, y_train)
    print(f"  Concluído em {time.time()-t0:.1f}s")

    models = {"scaler": scaler, "knn": knn, "svm": svm}

    for name, obj in models.items():
        path = os.path.join(models_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    print(f"[classical] Modelos salvos em: {models_dir}")

    return models


def load_classical_models(models_dir: str = MODELS_DIR) -> dict:
    models = {}
    for name in ("scaler", "knn", "svm"):
        path = os.path.join(models_dir, f"{name}.pkl")
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models



def predict_classical(
    models: dict,
    X_test: np.ndarray,
) -> dict:
    scaler = models["scaler"]
    X_scaled = scaler.transform(X_test)

    t0 = time.time()
    knn_pred = models["knn"].predict(X_scaled)
    knn_time_ms = (time.time() - t0) * 1000

    t0 = time.time()
    svm_pred = models["svm"].predict(X_scaled)
    svm_time_ms = (time.time() - t0) * 1000

    return {
        "knn_pred": knn_pred,
        "svm_pred": svm_pred,
        "knn_time_ms": knn_time_ms,
        "svm_time_ms": svm_time_ms,
    }
