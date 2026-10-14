from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils.preprocess import (
    GRID_SIZE,
    circular_crop,
    extract_patches,
    resize_image,
)


def save_rgb(path, image):
    # OpenCV writes BGR; our preprocessing uses RGB.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Cannot save image: {path}")


def prepare_dataset(manifest_path, output_root):
    records = pd.read_csv(
        manifest_path,
        dtype={"label": str, "batch": str},
        keep_default_na=False,
    )

    for index, record in enumerate(records.to_dict("records"), start=1):
        source = Path(record["image_path"])

        folder = (
            Path(output_root)
            / record["label"]
            / record["batch"]
            / source.stem
        )

        # Skip only when every expected prepared file exists.
        required_files = [
            folder / f"{source.stem}_full_resized.png",
            folder / "coverages.npy",
            *[
                folder / f"{source.stem}_p{i}.png"
                for i in range(1, GRID_SIZE ** 2 + 1)
            ],
        ]

        if all(path.is_file() for path in required_files):
            print(f"{index}/{len(records)}: SKIPPED {source.name}")
            continue

        # Existing incomplete folders can be processed again.
        folder.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Cannot read image: {source}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cropped, mask = circular_crop(image)
        patches, coverages = extract_patches(cropped, mask)

        save_rgb(
            folder / f"{source.stem}_full_resized.png",
            resize_image(cropped),
        )

        for number, patch in enumerate(patches, start=1):
            save_rgb(
                folder / f"{source.stem}_p{number}.png",
                resize_image(patch),
            )

        np.save(
            folder / "coverages.npy",
            np.asarray(coverages, dtype=np.float32),
        )

        print(f"{index}/{len(records)}: PROCESSED {source.name}")