from pathlib import Path
from typing import Literal
import torch
import cv2
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

from utils.prepare_dataset import OUTPUT_ROOT
from utils.preprocess import GRID_SIZE, load_prepared_image
from utils.labels import infer_label_keys, parse_label


class RawMaterialDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: Literal["training", "validation", "testing"],
    ):
        df = pd.read_csv(manifest_path)
        self.label_keys = infer_label_keys(df["label"])
        self.images = df.loc[df["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        record = self.images.iloc[idx]
        stem = Path(record["image_path"]).stem
        folder = OUTPUT_ROOT / record["label"] / record["batch"] / stem

        whole = load_prepared_image(folder / f"{stem}_full_resized.png")
        patches = torch.stack([
            load_prepared_image(folder / f"{stem}_p{i}.png")
            for i in range(1, GRID_SIZE ** 2 + 1)
        ])

        coverages = np.load(folder / "coverages.npy", allow_pickle=False)
        if (
            coverages.shape != (GRID_SIZE ** 2,)
            or not np.isfinite(coverages).all()
            or ((coverages < 0) | (coverages > 1)).any()
        ):
            raise ValueError(f"Invalid coverage: {folder}")

        label = parse_label(record["label"])

        return (
            whole,
            patches,
            torch.tensor(coverages, dtype=torch.float32),
            torch.tensor([label[key] for key in self.label_keys], dtype=torch.float32),
        )
