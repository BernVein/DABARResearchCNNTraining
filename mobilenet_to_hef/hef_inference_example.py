"""
How to use rawmat_encoder.hef + rawmat_encoder_head.npz on the Raspberry Pi.

The .hef only contains the MobileNet encoder (one image -> 576 features). This file shows
the full prediction: 17 views through the chip, then pooling + head on the CPU.

    python hef_inference_example.py build/rawmat_encoder.hef build/rawmat_encoder_head_hailo.npz photo.jpg

Use the *_head_hailo.npz (made by the headfit step) when it exists: that head was trained
on the chip's quantized features and is more accurate with the .hef than *_head.npz.

NOTE: written from the HailoRT 4.24 user guide; it has not been run on a device yet.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

# Reuse the exact preprocessing used for training, and the CPU half of the model.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.preprocess import circular_crop, extract_patches, resize_image  # noqa: E402

from convert import load_head, predict  # noqa: E402


def make_views(image_path):
    """Photo -> uint8 RGB [17, 224, 224, 3]: whole image first, then the 16 patches.

    Same steps as utils/prepare_dataset.py. No /255 and no mean/std here: the
    normalization layer is inside the .hef, so the chip takes raw 0..255 pixels.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cropped, mask = circular_crop(image)
    patches, _ = extract_patches(cropped, mask)
    return np.stack([resize_image(cropped)] + [resize_image(patch) for patch in patches])


def main(hef_path, head_path, image_path):
    head = load_head(head_path)
    label_keys = [str(key) for key in head["label_keys"]]
    hef = HEF(hef_path)

    with VDevice() as device:
        # Load the network onto the chip (PCIe is how the Pi AI Kit / HAT is connected).
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = device.configure(hef, configure_params)[0]

        # Input: we send uint8 images as they are.
        # Output: FLOAT32 makes HailoRT convert the chip's integers back to real feature
        # values (de-quantization) for us.
        input_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        input_name = hef.get_input_vstream_infos()[0].name
        output_name = hef.get_output_vstream_infos()[0].name

        with InferVStreams(network_group, input_params, output_params) as pipeline:
            with network_group.activate(network_group.create_params()):
                views = make_views(image_path)                      # [17, 224, 224, 3]
                result = pipeline.infer({input_name: views})        # all 17 views in one call
                features = result[output_name].reshape(1, len(views), -1)   # [1, 17, 576]

    # CPU part: mean/std pooling over the views + the small head (float32).
    output = predict(head, features)[0]
    if str(head["task"]) == "classification":
        scores = np.exp(output - output.max())
        scores /= scores.sum()                                      # softmax -> probabilities
        best = int(output.argmax())
        print(f"{label_keys[best]} ({scores[best]:.1%})")
    else:
        print({key: round(float(value), 2) for key, value in zip(label_keys, output)})


if __name__ == "__main__":
    main(*sys.argv[1:4])
