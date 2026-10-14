import cv2
import numpy as np


def augment_contrast(image):
    pixels = image.astype(np.float32)

    # Use average grayscale intensity as the contrast centre.
    mean = float(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).mean())
    factor = np.random.uniform(0.90, 1.10)

    pixels = mean + (pixels - mean) * factor
    return np.rint(np.clip(pixels, 0, 255)).astype(np.uint8)


def augment_saturation(image):
    pixels = image.astype(np.float32)
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)[..., None]
    factor = np.random.uniform(0.90, 1.10)

    pixels = gray + (pixels - gray) * factor
    return np.rint(np.clip(pixels, 0, 255)).astype(np.uint8)


def augment_gamma(image):
    gamma = np.random.uniform(0.90, 1.10)

    # A lookup table applies the same brightness curve to every channel.
    values = np.arange(256, dtype=np.float32) / 255.0
    table = np.rint(255.0 * values ** gamma).astype(np.uint8)

    return cv2.LUT(image, table)