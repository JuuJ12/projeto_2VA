"""
visualize.py
============
Geração de gráficos comparativos e análises visuais das características
extraídas por jogo, atendendo ao item 7 da proposta do projeto:
"Avaliação e visualização: acurácia, precisão, recall, F1 e matriz de confusão,
com gráficos das características por jogo."
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from src.data_utils import CLASS_NAMES, load_splits
from src.preprocessing import preprocess
from src.features import (
    hsv_histogram,
    lbp_histogram,
    glcm_features,
    fourier_radial,
    gradient_features,
    HSV_H_BINS,
)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "figures")


def generate_feature_visualizations(train_df: pd.DataFrame, n_samples_per_class: int = 30) -> None:
    """
    Gera gráficos das distribuições de características para cada um dos 10 jogos:
    1. Perfil médio de cor (Histograma de Matiz / Hue HSV)
    2. Comparação de densidade de bordas (Sobel) e contraste (GLCM)
    3. Perfil de frequência espacial radial (Fourier)
    4. Amostras visuais pré-processadas (grid 2x5)
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"[visualize] Coletando amostras ({n_samples_per_class}/classe) para gráficos comparativos...")

    hsv_by_class = {c: [] for c in CLASS_NAMES}
    fourier_by_class = {c: [] for c in CLASS_NAMES}
    edge_densities = {c: [] for c in CLASS_NAMES}
    glcm_contrasts = {c: [] for c in CLASS_NAMES}
    sample_images = {}

    for c in CLASS_NAMES:
        class_df = train_df[train_df["label"] == c].head(n_samples_per_class)
        for idx, row in class_df.iterrows():
            prep = preprocess(row["path"])
            bgr = prep["bgr"]
            gray = prep["gray"]

            if c not in sample_images:
                sample_images[c] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # HSV Hue
            hsv = hsv_histogram(bgr)
            hsv_by_class[c].append(hsv[:HSV_H_BINS])

            # Fourier
            four = fourier_radial(gray)
            fourier_by_class[c].append(four)

            # Gradient stats
            grad = gradient_features(gray)
            edge_densities[c].append(grad[2])  # edge_density

            # GLCM contrast (primeira propriedade)
            glcm = glcm_features(gray)
            glcm_contrasts[c].append(np.mean(glcm[:4]))

    # -------------------------------------------------------------
    # 1. Grid com 1 amostra de cada jogo após pré-processamento
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, c in enumerate(CLASS_NAMES):
        ax = axes[i // 5, i % 5]
        ax.imshow(sample_images[c])
        ax.set_title(c, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Amostras Pré-processadas (224×224 com Suavização Gaussiana)", fontsize=14, y=1.02)
    fig.tight_layout()
    grid_path = os.path.join(FIGURES_DIR, "sample_games_grid.png")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Amostras salvas em: {grid_path}")

    # -------------------------------------------------------------
    # 2. Perfil médio de Matiz (Hue) por jogo
    # -------------------------------------------------------------
    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
    hue_angles = np.linspace(0, 180, HSV_H_BINS)
    for i, c in enumerate(CLASS_NAMES):
        ax = axes[i % 5, i // 5]
        mean_hue = np.mean(hsv_by_class[c], axis=0)
        ax.plot(hue_angles, mean_hue, color="#2b5c8f", lw=2)
        ax.fill_between(hue_angles, mean_hue, color="#4C72B0", alpha=0.3)
        ax.set_title(c, fontsize=10, fontweight="bold")
        ax.set_ylabel("Frequência")
        if i % 5 == 4:
            ax.set_xlabel("Matiz (0 a 180°)")
    fig.suptitle("Distribuição Média de Matiz (HSV) por Jogo", fontsize=14, y=0.99)
    fig.tight_layout()
    hsv_path = os.path.join(FIGURES_DIR, "feature_hsv_distributions.png")
    fig.savefig(hsv_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Gráficos de HSV salvos em: {hsv_path}")

    # -------------------------------------------------------------
    # 3. Densidade de Bordas vs Contraste de Textura
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    mean_edges = [np.mean(edge_densities[c]) for c in CLASS_NAMES]
    mean_contrasts = [np.mean(glcm_contrasts[c]) for c in CLASS_NAMES]

    y_pos = np.arange(len(CLASS_NAMES))
    ax1.barh(y_pos, mean_edges, color="#c44e52")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(CLASS_NAMES)
    ax1.invert_yaxis()
    ax1.set_xlabel("Densidade Média de Bordas (Sobel)")
    ax1.set_title("Densidade de Bordas por Jogo")

    ax2.barh(y_pos, mean_contrasts, color="#55a868")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(CLASS_NAMES)
    ax2.invert_yaxis()
    ax2.set_xlabel("Contraste Médio de Textura (GLCM)")
    ax2.set_title("Contraste de Textura por Jogo")

    fig.tight_layout()
    edge_path = os.path.join(FIGURES_DIR, "feature_edge_texture_comparison.png")
    fig.savefig(edge_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Comparação de bordas e textura salva em: {edge_path}")

    # -------------------------------------------------------------
    # 4. Espectro Radial de Fourier por Jogo
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    bands = np.arange(1, 9)
    for c in CLASS_NAMES:
        mean_four = np.mean(fourier_by_class[c], axis=0)
        ax.plot(bands, mean_four, marker="o", label=c, alpha=0.8)
    ax.set_xlabel("Faixa Radial de Frequência (1=Baixas / Suaves → 8=Altas / Detalhes)")
    ax.set_ylabel("Fração de Energia")
    ax.set_title("Assinatura de Frequência Espacial (Fourier 2D) por Jogo")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    four_path = os.path.join(FIGURES_DIR, "feature_fourier_spectrum.png")
    fig.savefig(four_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Gráfico de Fourier salvo em: {four_path}")


if __name__ == "__main__":
    splits_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "splits", "splits.json")
    train_df, _, _ = load_splits(splits_file)
    generate_feature_visualizations(train_df)
