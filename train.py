from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import RawMaterialDataset
from model import RawMatCNN
from utils.preprocess import (
    GRID_SIZE,
    INPUT_SIZE,
    RADIUS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    normalize_images,
)


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.csv"
OUTPUT = ROOT / "outputs" / "rice_bran_90mm_run01.pth"
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


def train():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if OUTPUT.exists():
        raise FileExistsError(f"Choose a new output path: {OUTPUT}")

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    train_data = RawMaterialDataset(MANIFEST, "training")
    val_data = RawMaterialDataset(MANIFEST, "validation")
    label_keys = train_data.label_keys
    model = RawMatCNN(num_outputs=len(label_keys)).to(device)

    print("Output order:", label_keys, flush=True)

    print("GPU:", torch.cuda.get_device_name(device), flush=True)
    print("Caching training features...", flush=True)
    train_x, train_y = cache_features(model, train_data, device)

    print("Caching validation features...", flush=True)
    val_x, val_y = cache_features(model, val_data, device)
    print("Feature shapes:", train_x.shape, val_x.shape, flush=True)

    optimizer = torch.optim.Adam(
        model.head.parameters(), lr=1e-3, weight_decay=1e-4,
    )
    loss_function = nn.SmoothL1Loss(beta=5.0)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    best_rmse = float("inf")

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
            errors = model.head(val_x) - val_y
            if not torch.isfinite(errors).all():
                raise RuntimeError("Nonfinite validation predictions")
            rmse = errors.square().mean().sqrt().item()
            mae = errors.abs().mean().item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {(loss_sum / len(train_x)).item():.4f} | "
            f"Val RMSE: {rmse:.4f} | Val MAE: {mae:.4f}",
            flush=True,
        )

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save({
                "state_dict": model.state_dict(),
                "num_outputs": len(label_keys),
                "label_keys": list(label_keys),
                "epoch": epoch + 1,
                "validation_rmse": rmse,
                "validation_mae": mae,
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
    print(f"Best validation RMSE: {best_rmse:.4f} percentage points")


if __name__ == "__main__":
    train()
