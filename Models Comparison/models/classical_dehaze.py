"""Small dark-channel dehazing refinement for conservative neural outputs."""

import cv2
import numpy as np


def _dark_channel(image, patch):
    minimum = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(minimum, kernel)


def _guided_filter(guide, source, radius, epsilon):
    mean_guide = cv2.boxFilter(guide, cv2.CV_64F, (radius, radius))
    mean_source = cv2.boxFilter(source, cv2.CV_64F, (radius, radius))
    mean_product = cv2.boxFilter(guide * source, cv2.CV_64F,
                                 (radius, radius))
    covariance = mean_product - mean_guide * mean_source
    variance = (cv2.boxFilter(guide * guide, cv2.CV_64F, (radius, radius))
                - mean_guide * mean_guide)
    coefficient = covariance / (variance + epsilon)
    intercept = mean_source - coefficient * mean_guide
    mean_coefficient = cv2.boxFilter(coefficient, cv2.CV_64F,
                                     (radius, radius))
    mean_intercept = cv2.boxFilter(intercept, cv2.CV_64F,
                                   (radius, radius))
    return mean_coefficient * guide + mean_intercept


def dehaze_dcp(bgr, omega=0.88, minimum_transmission=0.12,
               patch=15, refine_radius=40):
    image = bgr.astype(np.float64) / 255.0
    dark = _dark_channel(image, patch)
    flat_dark = dark.reshape(-1)
    flat_image = image.reshape(-1, 3)
    count = max(int(flat_dark.size * 0.001), 1)
    indices = np.argpartition(flat_dark, -count)[-count:]
    atmospheric = flat_image[indices]
    atmospheric = atmospheric[np.argmax(atmospheric.sum(axis=1))]

    normalized = image / np.clip(atmospheric, 1e-6, None)
    transmission = 1.0 - omega * _dark_channel(normalized, patch)
    guide = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    transmission = _guided_filter(guide, transmission, refine_radius, 1e-3)
    transmission = np.clip(transmission, minimum_transmission, 1.0)

    restored = (image - atmospheric) / transmission[..., None] + atmospheric
    return np.clip(restored * 255.0, 0, 255).astype(np.uint8)
