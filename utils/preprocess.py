import cv2
import numpy as np
import torch

RADIUS = 900
GRID_SIZE = 4
INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def circular_crop(image, radius=RADIUS):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    square = image[cy - radius:cy + radius, cx - radius:cx + radius].copy()

    mask = np.zeros(square.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (radius, radius), radius, 255, -1)

    #  circle gray
    square[mask == 0] = (124, 116, 104)

    return square, mask

def extract_patches(image, mask):
    h, w = image.shape[:2]
    
    if mask.shape != (h, w) or h % GRID_SIZE or w % GRID_SIZE:
        raise ValueError(f"Mask dimensions must match image dimensions or be divisible by {GRID_SIZE}.")
    
    patch_h, patch_w = h // GRID_SIZE, w // GRID_SIZE
    patches, coverages = [], []
    
    for row in range(4):
        for col in range(4):
            y, x = row * patch_h, col * patch_w
            patch = image[y:y + patch_h, x:x + patch_w]
            patch_mask = mask[y:y + patch_h, x:x + patch_w]

            patches.append(patch)
            coverages.append(float((patch_mask > 0).mean()))
            
    return patches, coverages

def resize_image(image, input_size=INPUT_SIZE):
    return cv2.resize(
        image, 
        (input_size, input_size), 
        interpolation=cv2.INTER_AREA
        )

def image_to_tensor(image):
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
    mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
    return (tensor - mean) / std

def load_prepared_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    if image.shape != (INPUT_SIZE, INPUT_SIZE, 3):
        raise ValueError(f"Unexpected image dimensions: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()

def normalize_images(images):
    images = images.float() / 255.0
    mean = images.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = images.new_tensor(IMAGENET_STD)[:, None, None]
    return (images - mean) / std