"""
Image quality metrics for fog enhancement evaluation.
PSNR and SSIM computed between enhanced output and clean reference.
If no reference is available, only structural sharpness is reported.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    Higher is better; >30 dB is generally considered good quality.

    Args:
        img1, img2: uint8 numpy arrays (same shape)
    """
    return float(psnr_skimage(img1, img2, data_range=255))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index (0–1).
    Closer to 1 = more similar structure to reference.

    Args:
        img1, img2: uint8 numpy arrays (same shape)
    """
    # Convert to grayscale for SSIM
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return float(ssim_skimage(g1, g2, data_range=255))


def compute_brisque_proxy(image_bgr: np.ndarray) -> float:
    """
    No-reference quality proxy: Laplacian variance (sharpness).
    Used when no clean reference is available (real foggy images).
    Higher variance = sharper = less foggy.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def print_metrics(foggy: np.ndarray, enhanced: np.ndarray, reference: np.ndarray = None):
    """Print a comparison table of quality metrics."""
    print("\n" + "=" * 50)
    print("  IMAGE QUALITY METRICS")
    print("=" * 50)

    fog_sharp = compute_brisque_proxy(foggy)
    enh_sharp = compute_brisque_proxy(enhanced)
    print(f"  Sharpness (Laplacian var):")
    print(f"    Foggy input  : {fog_sharp:>8.2f}")
    print(f"    Enhanced     : {enh_sharp:>8.2f}  ({'+' if enh_sharp > fog_sharp else ''}{enh_sharp - fog_sharp:.2f})")

    if reference is not None:
        ref = cv2.resize(reference, (enhanced.shape[1], enhanced.shape[0]))
        psnr_val = compute_psnr(enhanced, ref)
        ssim_val = compute_ssim(enhanced, ref)
        print(f"  PSNR (vs clean ref)  : {psnr_val:>8.2f} dB")
        print(f"  SSIM (vs clean ref)  : {ssim_val:>8.4f}")
    else:
        print("  [PSNR / SSIM skipped — no clean reference provided]")
        print("  Tip: place a clear version of the scene as 'reference.png'")

    print("=" * 50 + "\n")