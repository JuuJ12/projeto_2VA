"""
classical.py
============
Pipeline clássico: extração de features + treino e avaliação de k-NN e SVM.

Fluxo:
    1. Lê imagens via data_utils e preprocessing
    2. Extrai vetor de features (features.py) para cada imagem
    3. Aplica StandardScaler (fit no treino, transform em val/teste)
    4. Treina k-NN e SVM com scikit-learn
    5. Salva modelos em results/models/

Decisões registradas:
    ADR-0003 — subconjunto 500/classe
    ADR-0004 — k-NN e SVM via scikit-learn (não do zero)
    GLOSSARY — Pipeline clássico, Vetor de features, Tempo de inferência
"""

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

# Caminho padrão para salvar modelos
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "models")


# ---------------------------------------------------------------------------
# Construção da matriz de features
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Itera sobre todas as imagens de um split e extrai o vetor de features
    de cada uma, montando a matriz X e o vetor de rótulos y.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com colunas [path, label, label_idx] (saída de data_utils).
    verbose : bool
        Se True, imprime progresso a cada 100 imagens.

    Retorna
    -------
    X : np.ndarray float32, shape (n_imgs, n_features)
    y : np.ndarray int,     shape (n_imgs,)
    """
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
        print()  # nova linha após o progresso

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def train_classical(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models_dir: str = MODELS_DIR,
) -> dict:
    """
    Treina k-NN e SVM com StandardScaler ajustado no conjunto de treino.

    Parâmetros
    ----------
    X_train : np.ndarray (n, d) — features brutas do conjunto de treino.
    y_train : np.ndarray (n,)  — rótulos inteiros.
    models_dir : str            — diretório para salvar os modelos.

    Retorna
    -------
    dict com chaves:
        "scaler" – StandardScaler ajustado
        "knn"    – KNeighborsClassifier treinado
        "svm"    – SVC treinado
    """
    os.makedirs(models_dir, exist_ok=True)

    # 1. Normalização: fit APENAS no treino (evita data leakage)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # 2. k-NN (k=5, métrica euclidiana, pesos uniformes)
    print("[classical] Treinando k-NN (k=5)...")
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    knn.fit(X_scaled, y_train)
    print(f"  Concluído em {time.time()-t0:.1f}s")

    # 3. SVM (kernel RBF, C=10, gamma='scale')
    print("[classical] Treinando SVM (kernel RBF, C=10)...")
    t0 = time.time()
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=False, random_state=42)
    svm.fit(X_scaled, y_train)
    print(f"  Concluído em {time.time()-t0:.1f}s")

    models = {"scaler": scaler, "knn": knn, "svm": svm}

    # 4. Persiste no disco
    for name, obj in models.items():
        path = os.path.join(models_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    print(f"[classical] Modelos salvos em: {models_dir}")

    return models


def load_classical_models(models_dir: str = MODELS_DIR) -> dict:
    """
    Carrega modelos clássicos salvos em disco.

    Retorna
    -------
    dict com chaves "scaler", "knn", "svm".
    """
    models = {}
    for name in ("scaler", "knn", "svm"):
        path = os.path.join(models_dir, f"{name}.pkl")
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models


# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------

def predict_classical(
    models: dict,
    X_test: np.ndarray,
) -> dict:
    """
    Aplica scaler e gera predições de k-NN e SVM no conjunto de teste.

    Mede separadamente o tempo de inferência de cada classificador
    (extração de features já feita em X_test).

    Parâmetros
    ----------
    models : dict — saída de train_classical ou load_classical_models.
    X_test : np.ndarray (n, d) — features brutas do conjunto de teste.

    Retorna
    -------
    dict com chaves:
        "knn_pred"       – np.ndarray (n,) predições k-NN
        "svm_pred"       – np.ndarray (n,) predições SVM
        "knn_time_ms"    – tempo de inferência k-NN em milissegundos
        "svm_time_ms"    – tempo de inferência SVM em milissegundos
    """
    scaler = models["scaler"]
    X_scaled = scaler.transform(X_test)

    # k-NN
    t0 = time.time()
    knn_pred = models["knn"].predict(X_scaled)
    knn_time_ms = (time.time() - t0) * 1000

    # SVM
    t0 = time.time()
    svm_pred = models["svm"].predict(X_scaled)
    svm_time_ms = (time.time() - t0) * 1000

    return {
        "knn_pred": knn_pred,
        "svm_pred": svm_pred,
        "knn_time_ms": knn_time_ms,
        "svm_time_ms": svm_time_ms,
    }
