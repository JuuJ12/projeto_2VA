"""
cnn.py
======
Pipeline CNN: MobileNetV2 com transfer learning (backbone congelado).

Arquitetura (ADR-0002):
    MobileNetV2 pré-treinado no ImageNet (backbone congelado)
    → embedding 1280-dim
    → nn.Dropout(0.2)
    → nn.Linear(1280, 10)

Treino:
    Adam lr=1e-3, 20 épocas, early stopping (paciência 5 no val loss)
    Augmentation no treino: flip horizontal, rotação ±10°
    Batch size 32, imagens 224×224 (ADR-0001)

Decisões registradas:
    ADR-0001 — 224×224 uniforme
    ADR-0002 — transfer learning, backbone congelado
    GLOSSARY — CNN, Transfer learning, Backbone, Tempo de inferência
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from src.data_utils import CLASS_NAMES

# Diretório padrão para salvar o modelo
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_cnn.pt")


# ---------------------------------------------------------------------------
# Dataset customizado
# ---------------------------------------------------------------------------

class GameScreenshotDataset(Dataset):
    """
    Dataset PyTorch para screenshots de jogos.

    Lê imagens do disco sob demanda (sem pré-carregar tudo na RAM),
    aplicando as transformações definidas em `transform`.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        """
        Parâmetros
        ----------
        df        : DataFrame com colunas [path, label_idx].
        transform : torchvision.transforms a aplicar.
        """
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        label = int(row["label_idx"])
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Transformações (augmentation no treino, apenas normalização em val/teste)
# ---------------------------------------------------------------------------

# Média e desvio padrão do ImageNet (necessários para pesos pré-treinados)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 10) -> nn.Module:
    """
    Constrói o modelo MobileNetV2 com backbone congelado.

    - Baixa pesos pré-treinados no ImageNet (primeira execução faz download)
    - Congela TODOS os parâmetros do backbone
    - Substitui o classificador original por: Dropout(0.2) → Linear(1280, 10)

    Parâmetros
    ----------
    num_classes : int — número de classes (padrão: 10 jogos).

    Retorna
    -------
    nn.Module com apenas o classifier treinável.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Congela backbone (features)
    for param in model.features.parameters():
        param.requires_grad = False

    # Substitui o classificador (in_features = 1280 no MobileNetV2)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )

    return model


def count_trainable_params(model: nn.Module) -> int:
    """Retorna o número de parâmetros treináveis do modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict:
    """
    Cria DataLoaders para treino, validação e teste.

    Parâmetros
    ----------
    train_df, val_df, test_df : DataFrames do split.
    batch_size : int (padrão: 32).
    num_workers : int — workers para carregamento paralelo.
                  0 = processo principal (mais seguro no Windows).

    Retorna
    -------
    dict com chaves "train", "val", "test" → DataLoader.
    """
    loaders = {
        "train": DataLoader(
            GameScreenshotDataset(train_df, TRAIN_TRANSFORM),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            GameScreenshotDataset(val_df, EVAL_TRANSFORM),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            GameScreenshotDataset(test_df, EVAL_TRANSFORM),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
    return loaders


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def train_cnn(
    model: nn.Module,
    loaders: dict,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
    save_path: str = BEST_MODEL_PATH,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict]:
    """
    Loop de treino com early stopping no val loss.

    Parâmetros
    ----------
    model     : nn.Module (saída de build_model).
    loaders   : dict com "train" e "val" DataLoaders.
    epochs    : número máximo de épocas (padrão: 20).
    lr        : learning rate do Adam (padrão: 1e-3).
    patience  : épocas sem melhora antes de parar (padrão: 5).
    save_path : caminho para salvar o melhor modelo.
    device    : torch.device. Se None, detecta automaticamente (CUDA > CPU).

    Retorna
    -------
    (model_best, history)
        model_best : modelo com os melhores pesos (estado do best val loss)
        history    : dict com listas "train_loss", "val_loss",
                     "train_acc", "val_acc"
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cnn] Dispositivo: {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # --- Fase de treino ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t_epoch = time.time()

        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # --- Fase de validação ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in loaders["val"]:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                val_loss    += loss.item() * images.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t_epoch
        print(
            f"  Época {epoch:>3}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_state, save_path)
            print(f"    ✓ Melhor modelo salvo (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"    Sem melhora há {epochs_no_improve}/{patience} épocas.")
            if epochs_no_improve >= patience:
                print(f"[cnn] Early stopping na época {epoch}.")
                break

    # Restaura melhor estado
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model, history


# ---------------------------------------------------------------------------
# Inferência
# ---------------------------------------------------------------------------

def predict_cnn(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> tuple[np.ndarray, float]:
    """
    Executa inferência no conjunto de teste e mede o tempo.

    Parâmetros
    ----------
    model  : nn.Module (melhor estado carregado).
    loader : DataLoader do conjunto de teste.
    device : torch.device. Se None, detecta automaticamente.

    Retorna
    -------
    (y_pred, inference_time_ms)
        y_pred             : np.ndarray (n,) de rótulos preditos
        inference_time_ms  : tempo total de inferência em milissegundos
                             (apenas o forward pass, sem carregar dados do disco)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    total_time = 0.0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            t0     = time.time()
            outputs = model(images)
            total_time += time.time() - t0
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    y_pred = np.concatenate(all_preds)
    return y_pred, total_time * 1000  # ms


def load_cnn_model(
    save_path: str = BEST_MODEL_PATH,
    num_classes: int = 10,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Carrega o melhor modelo salvo durante o treino.

    Retorna
    -------
    nn.Module em modo eval, no device especificado.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes)
    state = torch.load(save_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def plot_training_history(
    history: dict,
    out_path: str | None = None,
) -> None:
    """
    Plota as curvas de loss e acurácia durante o treino da CNN.

    Parâmetros
    ----------
    history  : dict retornado por train_cnn.
    out_path : caminho para salvar a figura. Se None, usa FIGURES_DIR.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_path is None:
        figures_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "results", "figures"
        )
        os.makedirs(figures_dir, exist_ok=True)
        out_path = os.path.join(figures_dir, "cnn_training_history.png")

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Treino")
    ax1.plot(epochs, history["val_loss"],   label="Validação")
    ax1.set(xlabel="Época", ylabel="Loss", title="Loss × Época")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Treino")
    ax2.plot(epochs, history["val_acc"],   label="Validação")
    ax2.set(xlabel="Época", ylabel="Acurácia", title="Acurácia × Época")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[cnn] Histórico de treino salvo em: {out_path}")
