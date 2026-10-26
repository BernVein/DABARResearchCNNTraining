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

def augment_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    lightness = clahe.apply(lightness)

    return cv2.cvtColor(
        cv2.merge((lightness, a, b)),
        cv2.COLOR_LAB2RGB,
    )
    
def augment_brightness(image):
    offset = np.random.uniform(-15.0, 15.0)
    pixels = image.astype(np.float32) + offset
    return np.rint(np.clip(pixels, 0, 255)).astype(np.uint8)


def augment_noise(image):
    sigma = np.random.uniform(1.0, 5.0)
    noise = np.random.normal(0.0, sigma, image.shape)
    pixels = image.astype(np.float32) + noise
    return np.rint(np.clip(pixels, 0, 255)).astype(np.uint8)