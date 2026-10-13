import argparse
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import RawMaterialDataset
from model import RawMatCNN
from utils.preprocess import (
    GRID_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    RADIUS,
    normalize_images,
)

PREPARED_ROOT = Path(r"C:\Users\realb\Documents\PROJECT_FILES\Dataset\for_training\classification\raw_material_batched")
ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "raw_material_MANIFEST_classification.csv"
OUTPUT = ROOT / "models" / "auto_detect.pth"
EPOCHS = 300
BATCH_SIZE = 32


@torch.no_grad()
def cache_features(model, dataset, device):
    if len(dataset) == 0:
        raise ValueError("Cannot cache an empty dataset")

    # Decode prepared PNGs once, without Windows worker startup overhead.
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model.eval()
    features, targets = [], []

    for whole, patches, _, target in loader:
        whole = normalize_images(whole.to(device, non_blocking=True))
        patches = normalize_images(patches.to(device, non_blocking=True))

        with torch.amp.autocast("cuda"):
            combined = model.extract_features(whole, patches)

        features.append(combined.float())
        targets.append(target.to(device))

    features = torch.cat(features)
    targets = torch.cat(targets)

    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("Nonfinite cached features or targets")

    return features, targets


def train(task: Literal["classification", "regression"] = "regression"):
    if task not in {"classification", "regression"}:
        raise ValueError(f"Unknown task: {task}")
    classification = task == "classification"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if OUTPUT.exists():
        raise FileExistsError(f"Choose a new OUTPUT path: {OUTPUT}")

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    train_data = RawMaterialDataset(MANIFEST, "training", task=task, prepared_root=PREPARED_ROOT)
    val_data = RawMaterialDataset(MANIFEST, "validation", task=task, prepared_root=PREPARED_ROOT)
    label_keys = train_data.label_keys

    if classification:
        if len(label_keys) < 2:
            raise ValueError("Classification needs at least two classes")
        missing = set(label_keys) - set(train_data.images["label"])
        if missing:
            raise ValueError(f"Classes missing from training: {sorted(missing)}")

    model = RawMatCNN(num_outputs=len(label_keys)).to(device)

    print("Task:", task, flush=True)
    print("OUTPUT order:", label_keys, flush=True)

    print("GPU:", torch.cuda.get_device_name(device), flush=True)
    print("Caching training features...", flush=True)
    train_x, train_y = cache_features(model, train_data, device)

    print("Caching validation features...", flush=True)
    val_x, val_y = cache_features(model, val_data, device)
    print("Feature shapes:", train_x.shape, val_x.shape, flush=True)

    optimizer = torch.optim.Adam(
        model.head.parameters(), lr=1e-3, weight_decay=1e-4,
    )
    # Classification uses integer targets and raw class scores.
    loss_function = (
        nn.CrossEntropyLoss() if classification else nn.SmoothL1Loss(beta=5.0)
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(train_x), device=device)
        loss_sum = torch.zeros((), device=device)

        for indices in order.split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            prediction = model.head(train_x[indices])
            loss = loss_function(prediction, train_y[indices])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")

            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(indices)

        model.eval()
        with torch.no_grad():
            prediction = model.head(val_x)
            if not torch.isfinite(prediction).all():
                raise RuntimeError("Nonfinite validation predictions")

            if classification:
                val_loss = loss_function(prediction, val_y).item()
                predicted_classes = prediction.argmax(dim=1)
                accuracy = (predicted_classes == val_y).float().mean().item()
                # Select classification checkpoints by lowest validation loss.
                score = val_loss
                metrics = {
                    "validation_loss": val_loss,
                    "validation_accuracy": accuracy,
                }
                validation_text = (
                    f"Val loss: {val_loss:.4f} | Val accuracy: {accuracy:.2%}"
                )
            else:
                errors = prediction - val_y
                if not torch.isfinite(errors).all():
                    raise RuntimeError("Nonfinite validation errors")
                rmse = errors.square().mean().sqrt().item()
                mae = errors.abs().mean().item()
                score = rmse
                metrics = {
                    "validation_rmse": rmse,
                    "validation_mae": mae,
                }
                validation_text = f"Val RMSE: {rmse:.4f} | Val MAE: {mae:.4f}"

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {(loss_sum / len(train_x)).item():.4f} | "
            f"{validation_text}",
            flush=True,
        )

        if score < best_score:
            best_score = score
            torch.save({
                "task": task,
                "state_dict": model.state_dict(),
                "num_outputs": len(label_keys),
                "label_keys": list(label_keys),
                "epoch": epoch + 1,
                **metrics,
                "pooling": "equal_mean_population_std",
                "input_size": INPUT_SIZE,
                "grid_size": GRID_SIZE,
                "crop_radius": RADIUS,
                "normalization_mean": IMAGENET_MEAN,
                "normalization_std": IMAGENET_STD,
                "train_records": train_data.images.to_dict("records"),
                "validation_records": val_data.images.to_dict("records"),
                "seed": 42,
            }, OUTPUT)

    print(f"Best checkpoint: {OUTPUT}")
    if classification:
        print(f"Best validation loss: {best_score:.4f}")
    else:
        print(f"Best validation RMSE: {best_score:.4f} percentage points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=["regression", "classification"],
        nargs="?",
        default="regression",
        help="Training task (default: regression)",
    )
    args = parser.parse_args()
    train(args.task)
