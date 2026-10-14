from pathlib import Path

ROOT = Path(__file__).resolve().parent
#################### DATASET PREPARATION ####################
RAW_DATASET_DIR = Path(r"C:\Users\realb\Documents\PROJECT_FILES\Dataset\90mm")
TASK = "regression"
FILENAME = "rice_bran_90mm_augmented"
MANIFEST_FILENAME = ROOT / f"{FILENAME}_{TASK}.csv"

if TASK not in ["classification", "regression"]:
    raise ValueError(f"Invalid task: {TASK}. Must be one of 'classification', and 'regression'")

PREPROCESSED_DATASET_DIR = Path(r"C:\Users\realb\Documents\PROJECT_FILES\Dataset\for_training\regression\rice_bran_90mm_augmented")

#################### TRAINING ####################
PTH_MODEL_OUTPUT_LOCATION = ROOT / "models" / "pth" / "rice_bran_90mm_augmented.pth"
ONNX_MODEL_OUTPUT_LOCATION = ROOT / "models" / "onnx" / "rice_bran_90mm_augmented.onnx"
EPOCHS = 300
BATCH_SIZE = 32


def check_config(is_notebook: bool = False):
    print("CHECK CONFIGURATION DETAILS:")
    print(f"RAW DATASET DIR: {RAW_DATASET_DIR}")
    print(f"TASK: {TASK}")
    print(f"MANIFEST FILENAME (same dir as this file): {MANIFEST_FILENAME}")
    print(f"PREPROCESSED DATASET DIR: {PREPROCESSED_DATASET_DIR}")
    print(f"PTH MODEL OUTPUT: {PTH_MODEL_OUTPUT_LOCATION}")
    print(f"ONNX MODEL OUTPUT: {ONNX_MODEL_OUTPUT_LOCATION}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"BATCH SIZE: {BATCH_SIZE}")
    
    if is_notebook:
        print("Review the configuration above before running the next cell.")
        return

    can_proceed = (
        input("Is the configuration correct? [y/n]: ").strip().lower() == "y"
    )

    if not can_proceed:
        raise RuntimeError(
            "Did not continue because the configuration is not correct yet."
        )

    print("Configuration confirmed. Continuing...")