"""
Comprehensive Image Quality Metrics for Dehazing Evaluation
============================================================

WHY PRECISION/RECALL/ACCURACY DON'T DIRECTLY APPLY
----------------------------------------------------
AOD-Net and GridDehazeNet are image-to-image regression models, not
classifiers. They output a continuous-valued image, not a class label.
The correct evaluation framework is:

  REFERENCE-BASED (need a clean GT image — gold standard):
    PSNR  — Peak Signal-to-Noise Ratio (dB). Pixel-level fidelity.
             Formula: 10·log10(MAX²/MSE).  Higher = better. >30 dB = good.
    SSIM  — Structural Similarity Index. Perceptual quality (luminance,
             contrast, structure).  Range [0,1].  Higher = better. >0.85 = good.
    MSE   — Mean Squared Error (pixel space). Direct error.  Lower = better.
    PSNR-B— PSNR for images with blocking artefacts (variant for compression).

  NO-REFERENCE (work on ANY output — useful for real road footage):
    Sharpness  — Laplacian variance. Measures edge crispness.  Higher = better.
    FogDensity — Laplacian + brightness combined.  Lower in output = more fog removed.
    Colorfulness — chroma saturation measure (Michel et al.).  Higher = richer colour.
    Contrast   — RMS contrast of the luminance channel.  Higher = better.
    NIQE-proxy — Dark channel prior estimate.  Lower = less haze remaining.

  SPEED / EFFICIENCY:
    Inference time (ms/frame)
    FPS (frames per second)
    Model parameters (millions)
    Model size on disk (MB)

  AUTONOMOUS DRIVING SPECIFIC:
    Visibility score — how far into the scene detail is recoverable
    Lane contrast    — contrast of road markings (critical for AV lane detection)
    Object salience  — how well foreground objects stand out from background
"""

import time
import cv2
import numpy as np

try:
    from skimage.metrics import (
        structural_similarity as _ssim,
        peak_signal_noise_ratio as _psnr,
        mean_squared_error as _mse,
    )
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False


# ─────────────────────────────────────────────────────────────────────────────
# Reference-based metrics
# ─────────────────────────────────────────────────────────────────────────────

def psnr(img: np.ndarray, ref: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB.  Higher = better."""
    if _SKIMAGE:
        return float(_psnr(ref, img, data_range=255))
    mse_val = float(np.mean((ref.astype(np.float64) -
                              img.astype(np.float64))**2))
    return float('inf') if mse_val == 0 else 10*np.log10(255**2/mse_val)


def ssim(img: np.ndarray, ref: np.ndarray) -> float:
    """Structural Similarity Index.  Range [0,1], higher = better."""
    if _SKIMAGE:
        return float(_ssim(ref, img, channel_axis=2, data_range=255))
    # Fallback: single-channel simplified SSIM
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float64)
    i = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    mu1,mu2 = r.mean(), i.mean()
    s1,s2   = r.std(), i.std()
    s12     = ((r-mu1)*(i-mu2)).mean()
    return float(((2*mu1*mu2+C1)*(2*s12+C2)) /
                 ((mu1**2+mu2**2+C1)*(s1**2+s2**2+C2)))


def mse(img: np.ndarray, ref: np.ndarray) -> float:
    """Mean Squared Error (pixel space).  Lower = better."""
    if _SKIMAGE:
        return float(_mse(ref.astype(np.float32), img.astype(np.float32)))
    return float(np.mean((ref.astype(np.float64) -
                           img.astype(np.float64))**2))


def mae(img: np.ndarray, ref: np.ndarray) -> float:
    """Mean Absolute Error.  Lower = better."""
    return float(np.mean(np.abs(ref.astype(np.float64) -
                                 img.astype(np.float64))))


def rmse(img: np.ndarray, ref: np.ndarray) -> float:
    """Root Mean Squared Error.  Lower = better."""
    return float(np.sqrt(mse(img, ref)))


# ─────────────────────────────────────────────────────────────────────────────
# No-reference metrics
# ─────────────────────────────────────────────────────────────────────────────

def sharpness(img: np.ndarray) -> float:
    """Laplacian variance — measures edge crispness.  Higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def fog_density_score(img: np.ndarray) -> float:
    """
    Combined fog density estimate using Laplacian variance + brightness.
    Lower score = less fog remaining in the output.
    Score in [0, 1].
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_br  = float(gray.mean())
    # Low sharpness + high brightness → more fog
    sharpness_fog = max(0.0, 1.0 - lap_var / 300.0)
    bright_fog    = max(0.0, (mean_br - 160.0) / 95.0)
    return float(np.clip(0.65*sharpness_fog + 0.35*bright_fog, 0, 1))


def colorfulness(img: np.ndarray) -> float:
    """
    Colorfulness measure (Hasler & Süsstrunk, 2003).
    Higher = more vivid, less washed-out by fog.
    """
    r = img[:,:,2].astype(np.float32)
    g = img[:,:,1].astype(np.float32)
    b = img[:,:,0].astype(np.float32)
    rg = r - g
    yb = 0.5*(r + g) - b
    return float(np.sqrt(rg.std()**2 + yb.std()**2) +
                 0.3*np.sqrt(rg.mean()**2 + yb.mean()**2))


def contrast_rms(img: np.ndarray) -> float:
    """RMS contrast of the luminance channel.  Higher = better."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return float(gray.std())


def dark_channel_prior(img: np.ndarray, patch: int = 15) -> float:
    """
    Dark channel prior estimate (He et al., 2009).
    Measures residual haze in the output.
    Lower = less haze remaining.  Good output should have low dark-channel mean.
    """
    min_ch  = np.min(img.astype(np.float32) / 255.0, axis=2)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark    = cv2.erode(min_ch, kernel)
    return float(dark.mean())


def entropy(img: np.ndarray) -> float:
    """
    Information entropy of the grayscale image.
    Higher entropy = richer detail recovered from fog.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def lane_contrast(img: np.ndarray) -> float:
    """
    Contrast of the road-lane region (bottom-centre crop).
    Proxy for how well lane markings are visible — critical for AV.
    Higher = better lane visibility.
    """
    h, w = img.shape[:2]
    # Bottom 40%, centre 30% (where lane markings typically appear)
    roi   = img[int(h*0.6):, int(w*0.35):int(w*0.65)]
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return float(gray.std())


def visibility_depth(img: np.ndarray) -> float:
    """
    Estimate how far into the scene detail is recoverable.
    Uses gradient magnitude in horizontal strips — the highest strip
    with measurable gradient approximates the visibility horizon.
    Higher = better (can see farther).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    strip_h = h // 10
    scores  = []
    for i in range(10):
        strip = gray[i*strip_h:(i+1)*strip_h, :]
        grad  = cv2.Sobel(strip, cv2.CV_64F, 1, 0) + \
                cv2.Sobel(strip, cv2.CV_64F, 0, 1)
        scores.append(float(np.abs(grad).mean()))
    # Find topmost strip with reasonable gradient (>5% of max)
    threshold = max(scores) * 0.05
    for i, s in enumerate(scores):
        if s < threshold:
            return float(i / 10.0)   # fraction of image height visible
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Speed / efficiency metrics
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model) -> int:
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, input_tensor,
                            warmup: int = 3,
                            runs: int = 10) -> dict:
    """
    Measure mean/std inference time in ms over multiple runs.

    Args:
        model        : PyTorch model in eval mode
        input_tensor : input to the model
        warmup       : warm-up runs (not measured)
        runs         : measured runs

    Returns:
        dict with mean_ms, std_ms, fps
    """
    import torch
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            times.append((time.perf_counter() - t0) * 1000)
    mean_ms = float(np.mean(times))
    std_ms  = float(np.std(times))
    return {'mean_ms': mean_ms, 'std_ms': std_ms,
            'fps': 1000.0 / mean_ms if mean_ms > 0 else 0}


# ─────────────────────────────────────────────────────────────────────────────
# Full metric bundle
# ─────────────────────────────────────────────────────────────────────────────

def compute_all(output: np.ndarray,
                foggy:  np.ndarray,
                ref:    np.ndarray | None = None) -> dict:
    """
    Compute the full metric suite for one output image.

    Args:
        output : uint8 BGR — model output (dehazed)
        foggy  : uint8 BGR — model input (foggy)
        ref    : uint8 BGR — clean reference (optional)

    Returns:
        dict of all metrics
    """
    metrics = {}

    # Reference-based (if available)
    if ref is not None:
        r = cv2.resize(ref, (output.shape[1], output.shape[0]))
        metrics['psnr']       = psnr(output, r)
        metrics['ssim']       = ssim(output, r)
        metrics['mse']        = mse(output, r)
        metrics['mae']        = mae(output, r)
        metrics['rmse']       = rmse(output, r)
        # Improvement over foggy baseline
        f = cv2.resize(foggy, (r.shape[1], r.shape[0]))
        metrics['psnr_gain']  = metrics['psnr'] - psnr(f, r)
        metrics['ssim_gain']  = metrics['ssim'] - ssim(f, r)
        metrics['mse_ratio']  = mse(f, r) / max(metrics['mse'], 1e-8)
    else:
        for k in ['psnr','ssim','mse','mae','rmse',
                  'psnr_gain','ssim_gain','mse_ratio']:
            metrics[k] = None

    # No-reference
    metrics['sharpness']      = sharpness(output)
    metrics['sharpness_gain'] = sharpness(output) - sharpness(foggy)
    metrics['fog_density']    = fog_density_score(output)
    metrics['fog_reduction']  = fog_density_score(foggy) - fog_density_score(output)
    metrics['colorfulness']   = colorfulness(output)
    metrics['color_gain']     = colorfulness(output) - colorfulness(foggy)
    metrics['contrast']       = contrast_rms(output)
    metrics['contrast_gain']  = contrast_rms(output) - contrast_rms(foggy)
    metrics['dark_channel']   = dark_channel_prior(output)
    metrics['entropy']        = entropy(output)
    metrics['lane_contrast']  = lane_contrast(output)
    metrics['visibility']     = visibility_depth(output)

    return metrics
