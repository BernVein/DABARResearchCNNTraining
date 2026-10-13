from pathlib import Path

import torch

from model import RawMatCNN

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "models" / "auto_detect.pth"
OUTPUT = ROOT / "models" / "auto.onnx"


def export():
    # Load only a checkpoint you created or trust.
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model = RawMatCNN(num_outputs=checkpoint["num_outputs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Example shapes: one whole image and its 16 patches.
    whole = torch.randn(1, 3, 224, 224)
    patches = torch.randn(1, 16, 3, 224, 224)

    torch.onnx.export(
        model,
        (whole, patches),
        str(OUTPUT),
        input_names=["whole", "patches"],
        output_names=["prediction"],
        dynamo=True,
        external_data=False,
        verify=True,
    )

    print(f"Exported: {OUTPUT}")


if __name__ == "__main__":
    export()