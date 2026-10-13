
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops



HSV_H_BINS = 32
HSV_S_BINS = 32

LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"

GLCM_DISTANCES = [1]
GLCM_ANGLES    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS     = ["contrast", "homogeneity", "energy", "correlation"]

FOURIER_N_BANDS = 8



def hsv_histogram(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [HSV_H_BINS], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [HSV_S_BINS], [0, 256])

    hist = np.concatenate([hist_h.flatten(), hist_s.flatten()])

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.astype(np.float32)



def lbp_histogram(img_gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(img_gray, P=LBP_P, R=LBP_R, method=LBP_METHOD)

    n_bins = LBP_P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist



def glcm_features(img_gray: np.ndarray) -> np.ndarray:
    img_reduced = (img_gray // 4).astype(np.uint8)

    glcm = graycomatrix(
        img_reduced,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=64,
        symmetric=True,
        normed=True,
    )

    features = []
    for prop in GLCM_PROPS:
        values = graycoprops(glcm, prop)[0]
        features.extend(values.tolist())

    return np.array(features, dtype=np.float32)



def fourier_radial(img_gray: np.ndarray, n_bands: int = FOURIER_N_BANDS) -> np.ndarray:
    f_transform = np.fft.fft2(img_gray.astype(np.float64))
    f_shifted   = np.fft.fftshift(f_transform)

    power_spectrum = np.abs(f_shifted) ** 2

    H, W = img_gray.shape
    cy, cx = H // 2, W // 2
    y_idx, x_idx = np.ogrid[:H, :W]
    radius = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

    max_radius = min(cy, cx)

    band_energy = np.zeros(n_bands, dtype=np.float64)
    band_edges  = np.linspace(0, max_radius, n_bands + 1)

    for i in range(n_bands):
        mask = (radius >= band_edges[i]) & (radius < band_edges[i + 1])
        band_energy[i] = power_spectrum[mask].sum()

    total = band_energy.sum()
    if total > 0:
        band_energy /= total

    return band_energy.astype(np.float32)



def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    hsv_feat     = hsv_histogram(img_bgr)
    lbp_feat     = lbp_histogram(img_gray)
    glcm_feat    = glcm_features(img_gray)
    fourier_feat = fourier_radial(img_gray)

    return np.concatenate([hsv_feat, lbp_feat, glcm_feat, fourier_feat])


def feature_names() -> list[str]:
    names = []
    names += [f"hsv_h_{i}" for i in range(HSV_H_BINS)]
    names += [f"hsv_s_{i}" for i in range(HSV_S_BINS)]
    names += [f"lbp_{i}" for i in range(LBP_P + 2)]
    for prop in GLCM_PROPS:
        for angle_deg in [0, 45, 90, 135]:
            names += [f"glcm_{prop}_{angle_deg}"]
    names += [f"fourier_band_{i}" for i in range(FOURIER_N_BANDS)]
    return names

