"""
features.py
===========
Extração de features visuais para o pipeline clássico.

Vetor final por imagem (~147 dims):
    HSV histograma  → 64  dims  (OpenCV)
    LBP uniforme    → 59  dims  (scikit-image)
    GLCM            → 16  dims  (scikit-image)
    Fourier radial  →  8  dims  (np.fft  — NumPy puro)
    ─────────────────────────────────────────
    Total           ≈ 147 dims

Decisões registradas:
    ADR-0004 — Fourier via np.fft conta como "do zero"; LBP e GLCM via libs.
    GLOSSARY — Vetor de features, Fourier radial.
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


# ---------------------------------------------------------------------------
# Parâmetros fixos (alinhados com o plano acordado na entrevista)
# ---------------------------------------------------------------------------

# HSV histograma
HSV_H_BINS = 32
HSV_S_BINS = 32

# LBP
LBP_P = 8        # número de vizinhos
LBP_R = 1        # raio em pixels
LBP_METHOD = "uniform"

# GLCM
GLCM_DISTANCES = [1]
GLCM_ANGLES    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0°, 45°, 90°, 135°
GLCM_PROPS     = ["contrast", "homogeneity", "energy", "correlation"]  # 4 × 4 ângulos = 16

# Fourier radial
FOURIER_N_BANDS = 8


# ---------------------------------------------------------------------------
# 1. Histograma HSV
# ---------------------------------------------------------------------------

def hsv_histogram(img_bgr: np.ndarray) -> np.ndarray:
    """
    Calcula e normaliza o histograma dos canais H e S da imagem em HSV.

    Justificativa: cada jogo tem paleta de cores própria; H e S capturam
    cor independente de brilho (canal V ignorado para maior robustez a
    variações de iluminação).

    Parâmetros
    ----------
    img_bgr : np.ndarray (H, W, 3) uint8, espaço BGR.

    Retorna
    -------
    np.ndarray float32, shape (HSV_H_BINS + HSV_S_BINS,) = (64,), L1-normalizado.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [HSV_H_BINS], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [HSV_S_BINS], [0, 256])

    hist = np.concatenate([hist_h.flatten(), hist_s.flatten()])

    # Normalização L1: divide pela soma total de pixels
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. LBP (Local Binary Pattern)
# ---------------------------------------------------------------------------

def lbp_histogram(img_gray: np.ndarray) -> np.ndarray:
    """
    Calcula o histograma LBP uniforme da imagem em escala de cinza.

    LBP compara cada pixel com seus P vizinhos em raio R; padrões "uniformes"
    (com no máximo 2 transições 0→1 ou 1→0) descrevem microestruturas como
    bordas, cantos e texturas planas. Distingue pixel art de gráficos realistas.

    Parâmetros
    ----------
    img_gray : np.ndarray (H, W) uint8.

    Retorna
    -------
    np.ndarray float32, shape (LBP_P + 2,) = (10,) para P=8.
    Nota: skimage retorna P+2 bins para método 'uniform'.
    """
    lbp = local_binary_pattern(img_gray, P=LBP_P, R=LBP_R, method=LBP_METHOD)

    n_bins = LBP_P + 2  # 10 bins para P=8
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


# ---------------------------------------------------------------------------
# 3. GLCM (Gray-Level Co-occurrence Matrix)
# ---------------------------------------------------------------------------

def glcm_features(img_gray: np.ndarray) -> np.ndarray:
    """
    Extrai propriedades de textura da GLCM em 4 direções.

    Propriedades calculadas: contraste, homogeneidade, energia, correlação.
    Para cada propriedade, a média sobre os 4 ângulos é retornada, resultando
    em 4 valores por propriedade × 4 propriedades = 16 features no total.

    Parâmetros
    ----------
    img_gray : np.ndarray (H, W) uint8.

    Retorna
    -------
    np.ndarray float32, shape (16,) — 4 props × 4 ângulos.
    """
    # GLCM espera uint8; reduz para 64 níveis para diminuir custo computacional
    img_reduced = (img_gray // 4).astype(np.uint8)  # 256 → 64 níveis

    glcm = graycomatrix(
        img_reduced,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=64,
        symmetric=True,
        normed=True,
    )  # shape: (64, 64, n_dist, n_angles)

    features = []
    for prop in GLCM_PROPS:
        # graycoprops retorna shape (n_dist, n_angles); aqui (1, 4)
        values = graycoprops(glcm, prop)[0]  # shape (4,), um valor por ângulo
        features.extend(values.tolist())

    return np.array(features, dtype=np.float32)  # (16,)


# ---------------------------------------------------------------------------
# 4. Fourier radial — DO ZERO com np.fft
# ---------------------------------------------------------------------------

def fourier_radial(img_gray: np.ndarray, n_bands: int = FOURIER_N_BANDS) -> np.ndarray:
    """
    Calcula a energia do espectro de Fourier 2D agrupada em faixas radiais.

    Passos (todos em NumPy puro):
      1. np.fft.fft2  → transforma a imagem para o domínio da frequência
      2. np.fft.fftshift → centraliza o espectro (DC no centro)
      3. Magnitude ao quadrado → espectro de potência
      4. Divide o espectro em `n_bands` anéis concêntricos igualmente
         espaçados da frequência zero até a frequência de Nyquist
      5. Soma a energia de cada anel → vetor de n_bands valores

    Justificativa: jogos com pixel art têm energia concentrada em baixas
    frequências; gráficos 3D realistas distribuem energia por toda a faixa.

    Parâmetros
    ----------
    img_gray : np.ndarray (H, W) uint8.
    n_bands  : int — número de faixas radiais. Padrão: 8.

    Retorna
    -------
    np.ndarray float32, shape (n_bands,), L1-normalizado.
    """
    # 1 & 2: FFT 2D e centralização
    f_transform = np.fft.fft2(img_gray.astype(np.float64))
    f_shifted   = np.fft.fftshift(f_transform)

    # 3: espectro de potência (magnitude²)
    power_spectrum = np.abs(f_shifted) ** 2

    # 4: grade de distâncias radiais ao centro
    H, W = img_gray.shape
    cy, cx = H // 2, W // 2
    y_idx, x_idx = np.ogrid[:H, :W]
    radius = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

    # Frequência de Nyquist = metade da menor dimensão
    max_radius = min(cy, cx)

    # 5: soma de energia por faixa radial
    band_energy = np.zeros(n_bands, dtype=np.float64)
    band_edges  = np.linspace(0, max_radius, n_bands + 1)

    for i in range(n_bands):
        mask = (radius >= band_edges[i]) & (radius < band_edges[i + 1])
        band_energy[i] = power_spectrum[mask].sum()

    # Normalização L1
    total = band_energy.sum()
    if total > 0:
        band_energy /= total

    return band_energy.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Vetor de features completo
# ---------------------------------------------------------------------------

def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extrai e concatena todos os descritores em um único vetor de features.

    Ordem e dimensões:
        [0:64]   HSV histograma  (64 dims)
        [64:74]  LBP uniforme    (10 dims)   ← P+2 = 10 para P=8
        [74:90]  GLCM            (16 dims)
        [90:98]  Fourier radial  ( 8 dims)
        ─────────────────────────────────
        Total:                   98 dims

    Parâmetros
    ----------
    img_bgr : np.ndarray (H, W, 3) uint8, já redimensionado e pré-processado.

    Retorna
    -------
    np.ndarray float32, shape (98,) — sem normalização global
    (StandardScaler é aplicado no nível do dataset em classical.py).
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    hsv_feat     = hsv_histogram(img_bgr)       # (64,)
    lbp_feat     = lbp_histogram(img_gray)      # (10,)
    glcm_feat    = glcm_features(img_gray)      # (16,)
    fourier_feat = fourier_radial(img_gray)     # ( 8,)

    return np.concatenate([hsv_feat, lbp_feat, glcm_feat, fourier_feat])


def feature_names() -> list[str]:
    """
    Retorna os nomes de cada dimensão do vetor de features.
    Útil para visualizações e análise de importância.
    """
    names = []
    names += [f"hsv_h_{i}" for i in range(HSV_H_BINS)]
    names += [f"hsv_s_{i}" for i in range(HSV_S_BINS)]
    names += [f"lbp_{i}" for i in range(LBP_P + 2)]
    for prop in GLCM_PROPS:
        for angle_deg in [0, 45, 90, 135]:
            names += [f"glcm_{prop}_{angle_deg}"]
    names += [f"fourier_band_{i}" for i in range(FOURIER_N_BANDS)]
    return names
