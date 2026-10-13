
import cv2
import numpy as np

DEFAULT_SIZE = (224, 224)



def load_and_resize(
    path: str,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"OpenCV não conseguiu ler: {path}")
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img_resized


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def to_hsv(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)



def gaussian_kernel(ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    if ksize % 2 == 0:
        raise ValueError(f"ksize deve ser ímpar; recebido: {ksize}")

    half = ksize // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def apply_gaussian(
    img: np.ndarray,
    ksize: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    half = ksize // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    kernel_1d = np.exp(-(ax**2) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()

    img_float = img.astype(np.float64)

    def _convolve_channel(channel: np.ndarray) -> np.ndarray:
        smoothed = np.apply_along_axis(
            lambda row: np.convolve(row, kernel_1d, mode="same"),
            axis=1,
            arr=channel,
        )
        smoothed = np.apply_along_axis(
            lambda col: np.convolve(col, kernel_1d, mode="same"),
            axis=0,
            arr=smoothed,
        )
        return smoothed

    if img_float.ndim == 2:
        return _convolve_channel(img_float)
    else:
        channels = [_convolve_channel(img_float[:, :, c]) for c in range(img_float.shape[2])]
        return np.stack(channels, axis=2)



def preprocess(
    path: str,
    size: tuple[int, int] = DEFAULT_SIZE,
    apply_blur: bool = True,
    ksize: int = 5,
    sigma: float = 1.0,
) -> dict:
    bgr = load_and_resize(path, size)

    if apply_blur:
        blurred = apply_gaussian(bgr, ksize=ksize, sigma=sigma)
        bgr_for_features = np.clip(blurred, 0, 255).astype(np.uint8)
    else:
        bgr_for_features = bgr

    return {
        "bgr": bgr_for_features,
        "hsv": to_hsv(bgr_for_features),
        "gray": to_gray(bgr_for_features),
    }
