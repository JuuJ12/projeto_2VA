"""
preprocessing.py
================
Pré-processamento de imagens para o pipeline clássico e a CNN.

Inclui:
  - Leitura e redimensionamento para 224×224 (ADR-0001)
  - Convolução Gaussiana implementada DO ZERO em NumPy (ADR-0004)
  - Conversão de espaço de cores (BGR → cinza, BGR → HSV)

Nenhuma função deste módulo usa cv2.GaussianBlur nem cv2.filter2D;
a suavização gaussiana é feita inteiramente com NumPy.
"""

import cv2
import numpy as np

# Resolução padrão de entrada para ambos os pipelines (ADR-0001)
DEFAULT_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Leitura e redimensionamento
# ---------------------------------------------------------------------------

def load_and_resize(
    path: str,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> np.ndarray:
    """
    Lê uma imagem do disco com OpenCV e redimensiona para `size`.

    Parâmetros
    ----------
    path : str
        Caminho absoluto do arquivo PNG/JPG.
    size : (largura, altura)
        Dimensões de saída em pixels. Padrão: (224, 224).

    Retorna
    -------
    np.ndarray uint8, shape (altura, largura, 3), espaço de cores BGR.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir ou não puder ser lido pelo OpenCV.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"OpenCV não conseguiu ler: {path}")
    # INTER_AREA é preferível para redução de resolução (anti-aliasing)
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img_resized


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Converte imagem BGR para escala de cinza (uint8)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def to_hsv(img_bgr: np.ndarray) -> np.ndarray:
    """Converte imagem BGR para HSV (uint8)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)


# ---------------------------------------------------------------------------
# Convolução Gaussiana — IMPLEMENTAÇÃO DO ZERO em NumPy
# ---------------------------------------------------------------------------

def gaussian_kernel(ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Constrói um kernel Gaussiano 2D de tamanho `ksize × ksize`.

    A fórmula aplicada é:
        G(x, y) = exp(-(x² + y²) / (2 σ²))
    normalizada para que a soma dos elementos seja 1.

    Parâmetros
    ----------
    ksize : int
        Tamanho do kernel (deve ser ímpar). Padrão: 5.
    sigma : float
        Desvio padrão da Gaussiana. Padrão: 1.0.

    Retorna
    -------
    np.ndarray float64, shape (ksize, ksize), soma = 1.0.
    """
    if ksize % 2 == 0:
        raise ValueError(f"ksize deve ser ímpar; recebido: {ksize}")

    # Gera coordenadas centralizadas em zero
    half = ksize // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)   # [-2, -1, 0, 1, 2] para ksize=5
    xx, yy = np.meshgrid(ax, ax)                          # grades 2D de coordenadas x e y

    # Aplica a fórmula Gaussiana
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()                                 # normaliza para soma = 1
    return kernel


def apply_gaussian(
    img: np.ndarray,
    ksize: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Aplica filtro Gaussiano a uma imagem usando convolução 2D implementada
    inteiramente em NumPy (sem cv2.GaussianBlur nem scipy).

    A estratégia é usar `np.convolve` de forma separável: como o kernel
    Gaussiano 2D é separável (G₂D = Gx · Gy^T), aplica-se primeiro um
    kernel 1D nas linhas e depois nas colunas — O(N·k) em vez de O(N·k²).

    Parâmetros
    ----------
    img : np.ndarray
        Imagem de entrada. Pode ser cinza (H, W) ou colorida (H, W, 3).
        Tipo: uint8 ou float.
    ksize : int
        Tamanho do kernel (ímpar). Padrão: 5.
    sigma : float
        Desvio padrão. Padrão: 1.0.

    Retorna
    -------
    np.ndarray float64 com o mesmo shape de `img`, valores no intervalo
    [0.0, 255.0].
    """
    # Kernel 1D separável (raiz do kernel 2D)
    half = ksize // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    kernel_1d = np.exp(-(ax**2) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()

    img_float = img.astype(np.float64)

    def _convolve_channel(channel: np.ndarray) -> np.ndarray:
        """Aplica convolução 1D separável em um canal 2D."""
        # Convolução nas linhas (eixo horizontal)
        # np.apply_along_axis aplica np.convolve linha a linha
        smoothed = np.apply_along_axis(
            lambda row: np.convolve(row, kernel_1d, mode="same"),
            axis=1,
            arr=channel,
        )
        # Convolução nas colunas (eixo vertical)
        smoothed = np.apply_along_axis(
            lambda col: np.convolve(col, kernel_1d, mode="same"),
            axis=0,
            arr=smoothed,
        )
        return smoothed

    if img_float.ndim == 2:
        # Imagem em escala de cinza
        return _convolve_channel(img_float)
    else:
        # Imagem colorida: processa cada canal independentemente
        channels = [_convolve_channel(img_float[:, :, c]) for c in range(img_float.shape[2])]
        return np.stack(channels, axis=2)


# ---------------------------------------------------------------------------
# Pipeline de pré-processamento completo
# ---------------------------------------------------------------------------

def preprocess(
    path: str,
    size: tuple[int, int] = DEFAULT_SIZE,
    apply_blur: bool = True,
    ksize: int = 5,
    sigma: float = 1.0,
) -> dict:
    """
    Pré-processa uma imagem do disco e retorna um dicionário com todas
    as representações necessárias para extração de features.

    Parâmetros
    ----------
    path : str
        Caminho do arquivo.
    size : (largura, altura)
        Resolução de saída. Padrão: (224, 224).
    apply_blur : bool
        Se True, aplica suavização Gaussiana antes das features.
    ksize, sigma : parâmetros do filtro Gaussiano.

    Retorna
    -------
    dict com chaves:
        "bgr"  – imagem BGR uint8 (224×224×3)
        "hsv"  – imagem HSV uint8 (224×224×3)
        "gray" – imagem cinza uint8 (224×224), pós-blur se apply_blur=True
    """
    bgr = load_and_resize(path, size)

    if apply_blur:
        # Suavização do zero em NumPy; recorta para uint8 para features de textura
        blurred = apply_gaussian(bgr, ksize=ksize, sigma=sigma)
        bgr_for_features = np.clip(blurred, 0, 255).astype(np.uint8)
    else:
        bgr_for_features = bgr

    return {
        "bgr": bgr_for_features,
        "hsv": to_hsv(bgr_for_features),
        "gray": to_gray(bgr_for_features),
    }
