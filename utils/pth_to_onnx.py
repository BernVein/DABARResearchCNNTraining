import argparse
from pathlib import Path

import torch

from model import RawMatCNN


def export(checkpoint: str | Path, output: str | Path):
    # Load only a checkpoint you created or trust.
    loaded_checkpoint = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model = RawMatCNN(num_outputs=loaded_checkpoint["num_outputs"])
    model.load_state_dict(loaded_checkpoint["state_dict"])
    model.eval()

    # Example shapes: one whole image and its 16 patches.
    whole = torch.randn(1, 3, 224, 224)
    patches = torch.randn(1, 16, 3, 224, 224)

    torch.onnx.export(
        model,
        (whole, patches),
        str(output),
        input_names=["whole", "patches"],
        output_names=["prediction"],
        dynamo=True,
        external_data=False,
        verify=True,
    )

    print(f"Exported: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.checkpoint, args.output)
