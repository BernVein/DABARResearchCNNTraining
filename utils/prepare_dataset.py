from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils.preprocess import circular_crop, extract_patches, resize_image

OUTPUT_ROOT = Path(r"C:\Users\realb\Documents\PROJECT_FILES\Dataset\for_training\classification\raw_material_batched")


def save_rgb(path, image):
    # OpenCV writes BGR; our preprocessing uses RGB.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Cannot save image: {path}")


def prepare_dataset(manifest_path, output_root=OUTPUT_ROOT):
    records = pd.read_csv(manifest_path)

    for index, record in enumerate(records.to_dict("records"), start=1):
        source = Path(record["image_path"])
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Cannot read image: {source}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cropped, mask = circular_crop(image)
        patches, coverages = extract_patches(cropped, mask)

        folder = (
            Path(output_root)
            / record["label"]
            / record["batch"]
            / source.stem
        )
        # Refuse to overwrite an existing prepared photograph.
        folder.mkdir(parents=True, exist_ok=False)

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

        print(f"{index}/{len(records)}: {source.name}")


if __name__ == "__main__":
    prepare_dataset(Path("raw_material_manifest_classification.csv"))