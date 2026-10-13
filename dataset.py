from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.labels import infer_label_keys, parse_label
from utils.preprocess import GRID_SIZE, load_prepared_image


class RawMaterialDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: Literal["training", "validation", "testing"],
        task: Literal["classification", "regression"] = "regression",
        prepared_root=None,
    ):
        if task not in {"classification", "regression"}:
            raise ValueError(f"Unknown task: {task}")

        self.task = task

        # Use a supplied directory, or select one based on the task.
        if prepared_root is None:
            project_root = Path(__file__).resolve().parent
            prepared_root = (
                project_root.parent
                / "Dataset"
                / "for_training"
                / task
                / "90mm_preprocessed"
            )

        self.prepared_root = Path(prepared_root)

        # Preserve folder names as text, including numeric class names.
        df = pd.read_csv(
            manifest_path,
            dtype={"label": str},
            keep_default_na=False,
        )

        if task == "classification":
            # Build the same class order for every split.
            self.label_keys = tuple(sorted(df["label"].unique()))

            # Map each complete class name to an integer.
            self.class_to_idx = {
                name: index
                for index, name in enumerate(self.label_keys)
            }
        else:
            # Regression parses labels such as R5 S0.
            self.label_keys = infer_label_keys(df["label"])

        # Select the requested split after establishing the output order.
        self.images = df.loc[df["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        record = self.images.iloc[idx]
        stem = Path(record["image_path"]).stem

        folder = (
            self.prepared_root
            / record["label"]
            / record["batch"]
            / stem
        )

        whole = load_prepared_image(
            folder / f"{stem}_full_resized.png"
        )

        patches = torch.stack([
            load_prepared_image(folder / f"{stem}_p{i}.png")
            for i in range(1, GRID_SIZE ** 2 + 1)
        ])

        coverages = np.load(
            folder / "coverages.npy",
            allow_pickle=False,
        )

        if (
            coverages.shape != (GRID_SIZE ** 2,)
            or not np.isfinite(coverages).all()
            or ((coverages < 0) | (coverages > 1)).any()
        ):
            raise ValueError(f"Invalid coverage: {folder}")

        if self.task == "classification":
            # One integer identifies this image's class.
            target = torch.tensor(
                self.class_to_idx[record["label"]],
                dtype=torch.long,
            )
        else:
            # One floating-point percentage per regression output.
            label = parse_label(record["label"])
            target = torch.tensor(
                [label[key] for key in self.label_keys],
                dtype=torch.float32,
            )

        return (
            whole,
            patches,
            torch.tensor(coverages, dtype=torch.float32),
            target,
        )