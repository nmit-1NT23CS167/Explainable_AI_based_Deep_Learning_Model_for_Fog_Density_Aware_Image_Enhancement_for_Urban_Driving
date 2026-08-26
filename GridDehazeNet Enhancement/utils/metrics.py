"""
Image Quality Metrics
=====================
PSNR  — Peak Signal-to-Noise Ratio (dB)   higher = better quality
SSIM  — Structural Similarity Index        closer to 1 = more similar
MSE   — Mean Squared Error (pixel space)   lower  = better
Sharpness — Laplacian variance             higher = sharper / less blurry
"""

import cv2
import numpy as np
from dataclasses import dataclass

try:
    from skimage.metrics import (
        structural_similarity as _ssim,
        peak_signal_noise_ratio as _psnr,
        mean_squared_error as _mse,
    )
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


@dataclass
class MetricsResult:
    psnr:      float | None     # dB   (None if no reference)
    ssim:      float | None     # 0–1  (None if no reference)
    mse:       float | None     # px²  (None if no reference)
    sharpness: float            # Laplacian variance (no reference needed)

    def __str__(self) -> str:
        lines = ['─' * 48, '  Image Quality Metrics', '─' * 48]
        if self.psnr is not None:
            lines.append(f'  PSNR       : {self.psnr:>8.3f} dB'
                         f'   {"✓ Good" if self.psnr > 28 else "△ Fair" if self.psnr > 20 else "✗ Poor"}')
        else:
            lines.append('  PSNR       : N/A  (no reference image)')
        if self.ssim is not None:
            lines.append(f'  SSIM       : {self.ssim:>8.4f}'
                         f'   {"✓ Good" if self.ssim > 0.8 else "△ Fair" if self.ssim > 0.6 else "✗ Poor"}')
        else:
            lines.append('  SSIM       : N/A  (no reference image)')
        if self.mse is not None:
            lines.append(f'  MSE        : {self.mse:>8.3f} px²')
        else:
            lines.append('  MSE        : N/A  (no reference image)')
        lines.append(f'  Sharpness  : {self.sharpness:>8.2f}'
                     f'   (Laplacian variance — higher = sharper)')
        lines.append('─' * 48)
        return '\n'.join(lines)

    def as_dict(self) -> dict:
        return {'psnr': self.psnr, 'ssim': self.ssim,
                'mse':  self.mse,  'sharpness': self.sharpness}


def compute_metrics(image: np.ndarray,
                    reference: np.ndarray | None = None) -> MetricsResult:
    """
    Compute quality metrics for a single image.

    Args:
        image     : uint8 BGR image to evaluate
        reference : uint8 BGR ground-truth (optional)
                    If None, PSNR/SSIM/MSE are skipped.

    Returns:
        MetricsResult dataclass
    """
    # Sharpness is reference-free
    sharpness = _laplacian_var(image)

    if reference is None:
        return MetricsResult(psnr=None, ssim=None, mse=None,
                             sharpness=sharpness)

    # Match sizes
    if reference.shape != image.shape:
        reference = cv2.resize(reference,
                               (image.shape[1], image.shape[0]))

    if _HAS_SKIMAGE:
        psnr = float(_psnr(reference, image, data_range=255))
        ssim = float(_ssim(reference, image,
                           channel_axis=2, data_range=255))
        mse  = float(_mse(reference.astype(np.float32),
                          image.astype(np.float32)))
    else:
        # Pure-numpy fallbacks
        psnr = _psnr_numpy(reference, image)
        ssim = _ssim_numpy(reference, image)
        mse  = _mse_numpy(reference, image)

    return MetricsResult(psnr=psnr, ssim=ssim, mse=mse,
                         sharpness=sharpness)


def compare_metrics(foggy: np.ndarray,
                    enhanced: np.ndarray,
                    reference: np.ndarray | None = None) -> None:
    """
    Print a side-by-side comparison of foggy vs enhanced metrics to stdout.

    Args:
        foggy     : original foggy image (uint8 BGR)
        enhanced  : dehazed image (uint8 BGR)
        reference : clean ground-truth (uint8 BGR, optional)
    """
    m_fog = compute_metrics(foggy,    reference)
    m_enh = compute_metrics(enhanced, reference)

    print()
    print('═' * 60)
    print('  GridDehazeNet — Image Quality Report')
    print('═' * 60)
    _row('Metric', 'Foggy Input', 'Enhanced', 'Δ', header=True)
    print('─' * 60)

    def _fmt(v):
        return f'{v:.3f}' if v is not None else '  N/A '

    def _delta(a, b, higher_better=True):
        if a is None or b is None:
            return '  —  '
        d = b - a
        sign = '+' if d >= 0 else ''
        arrow = '↑' if (d > 0) == higher_better else '↓'
        return f'{arrow} {sign}{d:.3f}'

    _row('PSNR  (dB)',
         _fmt(m_fog.psnr), _fmt(m_enh.psnr),
         _delta(m_fog.psnr, m_enh.psnr, higher_better=True))
    _row('SSIM',
         _fmt(m_fog.ssim), _fmt(m_enh.ssim),
         _delta(m_fog.ssim, m_enh.ssim, higher_better=True))
    _row('MSE  (px²)',
         _fmt(m_fog.mse), _fmt(m_enh.mse),
         _delta(m_fog.mse, m_enh.mse, higher_better=False))
    _row('Sharpness',
         f'{m_fog.sharpness:.2f}', f'{m_enh.sharpness:.2f}',
         _delta(m_fog.sharpness, m_enh.sharpness, higher_better=True))
    print('─' * 60)
    if reference is None:
        print('  * PSNR/SSIM/MSE require a clean reference image.')
        print('    Use --reference <path> to enable full comparison.')
    print('═' * 60)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _laplacian_var(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _psnr_numpy(ref: np.ndarray, img: np.ndarray) -> float:
    mse = float(np.mean((ref.astype(np.float64) -
                          img.astype(np.float64)) ** 2))
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


def _ssim_numpy(ref: np.ndarray, img: np.ndarray) -> float:
    """Simplified SSIM (luminance only)."""
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float64)
    i = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = r.mean(), i.mean()
    s1,  s2  = r.std(),  i.std()
    s12 = ((r - mu1) * (i - mu2)).mean()
    return float(((2*mu1*mu2 + C1) * (2*s12 + C2)) /
                 ((mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)))


def _mse_numpy(ref: np.ndarray, img: np.ndarray) -> float:
    return float(np.mean((ref.astype(np.float64) -
                           img.astype(np.float64)) ** 2))


def _row(col1, col2, col3, col4, header=False):
    line = f'  {col1:<14} {col2:>12} {col3:>12} {col4:>12}'
    print(line)
    if header:
        print('─' * 60)
