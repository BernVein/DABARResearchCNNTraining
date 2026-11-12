import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import check_config  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DATASET_DIR = Path(r"C:\Users\realb\Documents\PROJECT_FILES\Dataset\for_training\regression\soybean_meal_augmented")

def shlop_all(source_root=DATASET_DIR, output_dir=None):
    source_root = Path(source_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Preprocessed dataset not found: {source_root}")

    output_dir = Path(output_dir) if output_dir else source_root.with_name(f"{source_root.name}_all")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    copied = skipped = 0
    for index, source in enumerate(images, start=1):
        target = output_dir / source.name

        # Names already include the image/augment name; if two labels or
        # batches still clash, prefix with label_batch to keep both.
        if target.exists() and target.stat().st_size != source.stat().st_size:
            relative = source.relative_to(source_root).parts
            target = output_dir / "_".join([*relative[:-2], source.name])

        if target.exists():
            skipped += 1
        else:
            shutil.copy2(source, target)
            copied += 1

        if index % 1000 == 0 or index == len(images):
            print(f"{index}/{len(images)} images handled")

    print(f"Done. Copied {copied}, skipped {skipped} (already there) -> {output_dir}")
    return output_dir


if __name__ == "__main__":
    check_config(is_notebook=False)
    shlop_all()
