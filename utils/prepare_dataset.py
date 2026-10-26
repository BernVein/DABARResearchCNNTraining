from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils.augment import (
    augment_contrast,
    augment_gamma,
    augment_saturation,
    augment_clahe,
    augment_brightness,
    augment_noise
)
from utils.preprocess import (
    GRID_SIZE,
    circular_crop,
    extract_patches,
    resize_image,
)


def save_rgb(path, image):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Cannot save image: {path}")


def prepare_dataset(manifest_path, output_root):
    records = pd.read_csv(
        manifest_path,
        dtype={"label": str, "batch": str},
        keep_default_na=False,
    )

    # CLAHE is preprocessing, so it is not a separate augmentation.
    augments = [
        augment_contrast,
        augment_gamma,
        augment_saturation,
        augment_brightness,
        augment_noise,
    ]

    for index, record in enumerate(records.to_dict("records"), start=1):
        source = Path(record["image_path"])
        base_name = f"{source.stem}"

        # Every split gets the CLAHE baseline.
        versions = [(base_name, None)]

        # Only training gets augmented copies of that baseline.
        if record["split"] == "training":
            versions += [
                (f"{base_name}_{augment.__name__}", augment)
                for augment in augments
            ]

        clahe_image = None

        for name, augment in versions:
            folder = (
                Path(output_root)
                / record["label"]
                / record["batch"]
                / name
            )

            required_files = [
                folder / f"{name}_full_resized.png",
                folder / "coverages.npy",
                *[
                    folder / f"{name}_p{i}.png"
                    for i in range(1, GRID_SIZE ** 2 + 1)
                ],
            ]

            if all(path.is_file() for path in required_files):
                print(f"{index}/{len(records)}: SKIPPED {name}")
                continue

            if clahe_image is None:
                image = cv2.imread(str(source))
                if image is None:
                    raise ValueError(f"Cannot read image: {source}")

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                clahe_image = augment_clahe(rgb)

            image = clahe_image.copy()
            if augment is not None:
                image = augment(image)

            folder.mkdir(parents=True, exist_ok=True)

            cropped, mask = circular_crop(image)
            patches, coverages = extract_patches(cropped, mask)

            save_rgb(
                folder / f"{name}_full_resized.png",
                resize_image(cropped),
            )

            for number, patch in enumerate(patches, start=1):
                save_rgb(
                    folder / f"{name}_p{number}.png",
                    resize_image(patch),
                )

            np.save(
                folder / "coverages.npy",
                np.asarray(coverages, dtype=np.float32),
            )

            print(f"{index}/{len(records)}: PROCESSED {name}")