import cv2
import matplotlib.pyplot as plt
from preprocess import (
    GRID_SIZE,
    RADIUS,
    circular_crop,
    extract_patches,
    resize_image,
)


def _preview_all(image, radius=RADIUS):
    cropped, mask = circular_crop(image, radius)
    patches, coverages = extract_patches(cropped, mask)

    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    axes = axes.ravel()

    axes[0].imshow(cropped)
    axes[0].set_title("Whole crop")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Valid pixels")

    for i, (patch, coverage) in enumerate(zip(patches, coverages)):
        axes[i + 2].imshow(patch)
        axes[i + 2].set_title(f"Patch {i + 1}: {coverage:.0%}")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()

    # Additional preview: resize after extracting patches
    resized_crop = resize_image(cropped)
    resized_patches = [resize_image(p) for p in patches]

    fig_resized, axes = plt.subplots(3, 6, figsize=(15, 8))
    axes = axes.ravel()

    axes[0].imshow(resized_crop, interpolation="nearest")
    axes[0].set_title("Whole crop: 224 × 224")

    axes[1].text(
        0.5, 0.5, "All images resized\nusing INTER_AREA",
        ha="center", va="center", transform=axes[1].transAxes,
    )

    for i, patch in enumerate(resized_patches):
        axes[i + 2].imshow(patch, interpolation="nearest")
        axes[i + 2].set_title(f"Patch {i + 1}: 224 × 224")

    for ax in axes:
        ax.axis("off")

    fig_resized.tight_layout()
    plt.show()

    return cropped, mask, patches, coverages


if __name__ == "__main__":
    image = cv2.imread(r"C:\Users\realb\Documents\PROJECT FILES\Dataset\for_training\regression\testing_tapped\new\img_20260925_153435_0006.jpg")
    if image is None:
        raise ValueError("Cannot read image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cropped, mask, patches, coverages = _preview_all(image)

    assert len(patches) == len(coverages) == GRID_SIZE ** 2
