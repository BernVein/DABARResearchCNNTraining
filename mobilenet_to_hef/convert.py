"""
RawMatCNN (.pth)  ->  ONNX  ->  Hailo HAR  ->  quantized HAR  ->  .hef

WHY THE OLD ONNX (models/onnx/auto.onnx) CANNOT BE USED FOR HAILO
-----------------------------------------------------------------
utils/pth_to_onnx.py exports the *whole* RawMatCNN.forward(). That graph has:
  * a 5-D input  "patches" [1, 16, 3, 224, 224]   (Hailo inputs are images: [H, W, C])
  * a Reshape that folds the 16 patches into the batch axis (the chip has no batch axis)
  * Gather / Slice / Sub / Sqrt for the mean+std pooling over the 17 views
  * opset 20 from the dynamo exporter            (the DFC parser supports opset 8, 11-17)
None of that maps onto the Hailo chip.

WHAT THIS SCRIPT BUILDS INSTEAD
-------------------------------
The expensive part of RawMatCNN is the MobileNetV3-Small encoder, and it processes every
view (1 whole image + 16 patches) independently. So we split the model in two:

  ON THE HAILO CHIP (.hef)          one 224x224 RGB uint8 image -> 576 features
  ON THE CPU (numpy, microseconds)  mean/std pooling over the 17 feature vectors + the
                                    small Linear-ReLU-Linear head (kept in float32)

At runtime you send the 17 views through the .hef as a batch of 17, then finish on the CPU.
This is mathematically the same model, and it is also the best layout for accuracy: the
std pooling and the head never get quantized at all.

STEPS (run all, or pick with --steps)
-------------------------------------
  export    .pth -> encoder ONNX (+ head weights .npz), verified against PyTorch
  calib     build the calibration set and a held-out test set from the prepared dataset
  parse     ONNX -> Hailo HAR, verified against ONNX Runtime        (needs Hailo DFC)
  optimize  model script + quantization -> quantized HAR            (needs Hailo DFC)
  headfit   re-train the CPU head on the quantized encoder's output (needs Hailo DFC
            and --prepared-root) -> <name>_head_hailo.npz, the head to use on the Pi
  test      SNR / accuracy of the quantized model vs. the float one (needs Hailo DFC)
  compile   quantized HAR -> .hef                                   (needs Hailo DFC)

"export" and "calib" only need torch/opencv, so they can run on Windows where the dataset
is. The other steps must run inside the Hailo AI SW Suite docker (or any env with the DFC).

EXAMPLES
--------
  # Windows, where the prepared dataset lives:
  python convert.py --steps export,calib --prepared-root "C:\\...\\for_training\\classification\\raw_material_batched"

  # Hailo docker (copy this folder, including build/, into shared_with_docker first):
  python convert.py --steps parse,optimize,test,compile

  # Everything in one go, if the DFC env can also see the dataset:
  python convert.py --prepared-root /local/shared_with_docker/raw_material_batched
"""

import argparse
import json
import sys
from itertools import zip_longest
from pathlib import Path, PureWindowsPath

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent  # DABARResearchCNNTraining/

INPUT_SIZE = 224   # every view is 224x224 (utils/preprocess.py INPUT_SIZE)
GRID_SIZE = 4      # 4x4 = 16 patches   (utils/preprocess.py GRID_SIZE)
VIEWS = 1 + GRID_SIZE ** 2  # 17 images go through the encoder per sample

# ImageNet normalization used in training (utils/preprocess.py), in 0..1 units...
MEAN_01 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_01 = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# ...and the same numbers in 0..255 units. The Hailo model gets a normalization layer
# with these values, so on the chip the input is the raw uint8 RGB image.
MEAN_255 = (MEAN_01 * 255.0).tolist()   # [123.675, 116.28, 103.53]
STD_255 = (STD_01 * 255.0).tolist()     # [58.395, 57.12, 57.375]


# =============================================================================
# Small helpers shared by several steps
# =============================================================================

class VerificationError(Exception):
    """A float-vs-float check failed: the conversion itself is wrong, stop here."""


def snr_db(reference, test):
    """Signal-to-noise ratio in dB: how big the signal is compared to the error.

    Every +6 dB halves the error. 20 dB = 10% error, 40 dB = 1% error.
    """
    reference = np.asarray(reference, dtype=np.float64)
    noise = np.asarray(test, dtype=np.float64) - reference
    noise_power = np.sum(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return float(10.0 * np.log10(np.sum(reference ** 2) / noise_power))


def read_rgb(path):
    """Read a prepared 224x224 PNG as RGB uint8.

    np.fromfile + imdecode is used instead of cv2.imread because cv2.imread cannot open
    paths with non-ASCII characters on Windows.
    """
    import cv2

    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    if image.shape != (INPUT_SIZE, INPUT_SIZE, 3):
        raise ValueError(f"Expected {INPUT_SIZE}x{INPUT_SIZE}x3, got {image.shape}: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR, the model wants RGB


def normalize(images_uint8):
    """uint8 RGB [..., H, W, 3] -> float32 normalized exactly like training."""
    return (images_uint8.astype(np.float32) / 255.0 - MEAN_01) / STD_01


def pool_views(features):
    """[samples, 17, 576] encoder features -> [samples, 1728] head input.

    Same as RawMatCNN.extract_features(): whole-image features, then the mean and the
    population std (ddof=0 == torch's unbiased=False) of the 16 patch features.
    """
    whole = features[:, 0]
    patches = features[:, 1:]
    return np.concatenate([whole, patches.mean(axis=1), patches.std(axis=1)], axis=1)


def run_head(head, pooled):
    """The Linear(1728,128) -> ReLU -> Linear(128,outputs) head, in numpy."""
    hidden = np.maximum(pooled @ head["w0"].T + head["b0"], 0.0)
    return hidden @ head["w1"].T + head["b1"]


def predict(head, features):
    """Encoder features [samples, 17, 576] -> final model output [samples, outputs]."""
    return run_head(head, pool_views(features))


def load_head(path):
    """Load the head weights + metadata written by the export step."""
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def onnx_features(onnx_path, images_uint8):
    """Run the float ONNX encoder with ONNX Runtime. This is our 'ground truth'.

    images_uint8: [N, 224, 224, 3] RGB. Returns [N, 576].
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    outputs = []
    for image in images_uint8:
        # ONNX wants NCHW (channels first) normalized float; batch is fixed to 1.
        tensor = normalize(image).transpose(2, 0, 1)[None]
        outputs.append(session.run(None, {name: tensor})[0].reshape(-1))
    return np.stack(outputs)


class Paths:
    """All files this script reads/writes, in one place."""

    def __init__(self, args):
        out, name = Path(args.out_dir), args.name
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        self.onnx = out / f"{name}.onnx"                    # encoder only, Hailo-friendly
        self.head = out / f"{name}_head.npz"                # CPU head weights + labels
        self.hailo_head = out / f"{name}_head_hailo.npz"    # head re-trained on chip features
        self.calib = out / f"{name}_calib.npy"              # uint8 [N,224,224,3]
        self.test = out / f"{name}_test.npz"                # held-out samples for testing
        self.parsed_har = out / f"{name}_parsed.har"
        self.alls = out / f"{name}.alls"                    # the model script we generate
        self.quantized_har = out / f"{name}_quantized.har"
        self.compiled_har = out / f"{name}_compiled.har"
        self.report = out / f"{name}_report.json"
        self.hef = out / f"{name}.hef"


# =============================================================================
# STEP 1: export  (.pth -> ONNX encoder + head weights)
# =============================================================================

def step_export(args, paths):
    import torch
    from torch import nn
    from torchvision.models import mobilenet_v3_small

    print("\n=== export: .pth -> encoder ONNX + head weights ===")

    # The checkpoint is a plain dict of tensors/lists, so weights_only=True can load it
    # without executing any pickled code.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]

    # In model.py:  self.encoder = Sequential(backbone.features, backbone.avgpool, Flatten)
    # so the MobileNet weights are stored under the prefix "encoder.0.".
    # weights=None: do not download ImageNet weights, ours overwrite them anyway.
    features = mobilenet_v3_small(weights=None).features
    features.load_state_dict({
        key[len("encoder.0."):]: value
        for key, value in state.items() if key.startswith("encoder.0.")
    })  # strict by default: errors if a single weight is missing or unexpected

    # What goes on the chip: conv features + global average pool. Output [1, 576, 1, 1].
    # Flatten is left out on purpose; a 4-D output is the natural Hailo output and we
    # simply reshape it on the CPU.
    encoder = nn.Sequential(features, nn.AdaptiveAvgPool2d(1))
    encoder.eval()  # eval mode: BatchNorm uses its stored statistics (required by Hailo)

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    export_kwargs = dict(
        input_names=["input"],
        output_names=["features"],
        opset_version=args.opset,   # DFC supports 8 and 11-17; its guide uses 13
        export_params=True,         # store the weights inside the .onnx
        do_constant_folding=True,   # pre-compute constant sub-graphs
    )
    # Newer torch defaults to the "dynamo" exporter, which produced the opset-20 file.
    # The legacy TorchScript exporter gives the plain graph the Hailo parser expects.
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(encoder, dummy, str(paths.onnx), **export_kwargs)

    import onnx
    onnx.checker.check_model(onnx.load(str(paths.onnx)))  # structural sanity check

    # The DFC guide recommends onnx-simplifier when parsing fails. Use it if installed.
    try:
        import onnxsim
        simplified, ok = onnxsim.simplify(onnx.load(str(paths.onnx)))
        if ok:
            onnx.save(simplified, str(paths.onnx))
            print("onnx-simplifier applied")
    except ImportError:
        pass

    # ---- head weights + metadata for the CPU part --------------------------------
    task = checkpoint.get("config.TASK", checkpoint.get("task", "classification"))
    np.savez(
        paths.head,
        w0=state["head.0.weight"].numpy(), b0=state["head.0.bias"].numpy(),
        w1=state["head.2.weight"].numpy(), b1=state["head.2.bias"].numpy(),
        label_keys=np.array(checkpoint["label_keys"]),  # output order of the model
        task=np.array(task),
    )
    head = load_head(paths.head)

    # ---- verification: (ONNX encoder + numpy head) must equal the PyTorch model ----
    # Reference = RawMatCNN.forward() re-done in torch with the checkpoint weights.
    views = torch.rand(2, VIEWS, 3, INPUT_SIZE, INPUT_SIZE)             # 2 fake samples, 0..1
    views_uint8 = (views * 255).round().byte()                           # what the camera gives
    normalized = torch.from_numpy(
        normalize(views_uint8.permute(0, 1, 3, 4, 2).numpy())           # NHWC, normalized
    ).permute(0, 1, 4, 2, 3)                                             # back to NCHW
    with torch.no_grad():
        feats = encoder(normalized.flatten(0, 1)).reshape(2, VIEWS, -1)
        combined = torch.cat([
            feats[:, 0], feats[:, 1:].mean(dim=1), feats[:, 1:].std(dim=1, unbiased=False),
        ], dim=1)
        torch_head = nn.Sequential(nn.Linear(combined.shape[1], 128), nn.ReLU(),
                                   nn.Linear(128, checkpoint["num_outputs"]))
        torch_head.load_state_dict({
            key[len("head."):]: value for key, value in state.items() if key.startswith("head.")
        })
        expected = torch_head(combined).numpy()

    onnx_feats = onnx_features(
        paths.onnx, views_uint8.permute(0, 1, 3, 4, 2).numpy().reshape(-1, INPUT_SIZE, INPUT_SIZE, 3)
    ).reshape(2, VIEWS, -1)
    got = predict(head, onnx_feats)

    feature_snr = snr_db(feats.numpy(), onnx_feats)
    output_snr = snr_db(expected, got)
    print(f"ONNX encoder vs PyTorch   : {feature_snr:.1f} dB")
    print(f"ONNX + numpy head vs torch: {output_snr:.1f} dB (max abs diff {np.abs(expected - got).max():.2e})")
    if min(feature_snr, output_snr) < 80:   # float32 round-off alone gives >100 dB
        raise VerificationError("Exported model does not match PyTorch. Do not continue.")

    print(f"task={task} outputs={[str(key) for key in head['label_keys']]}")
    print(f"saved {paths.onnx}\nsaved {paths.head}")


# =============================================================================
# STEP 2: calib  (calibration set + held-out test set from the prepared dataset)
# =============================================================================

def sample_folder(prepared_root, record):
    """Folder that utils/prepare_dataset.py wrote for one manifest row."""
    # The manifest stores Windows paths; PureWindowsPath splits them correctly even
    # when this runs on Linux inside the Hailo docker.
    stem = PureWindowsPath(record["image_path"]).stem
    return Path(prepared_root) / record["label"] / record["batch"] / stem, stem


def load_views(prepared_root, record):
    """One sample -> uint8 [17, 224, 224, 3]: index 0 = whole image, 1..16 = patches."""
    folder, stem = sample_folder(prepared_root, record)
    files = [folder / f"{stem}_full_resized.png"]
    files += [folder / f"{stem}_p{i}.png" for i in range(1, VIEWS)]
    return np.stack([read_rgb(path) for path in files])


def step_calib(args, paths):
    import pandas as pd

    print("\n=== calib: build calibration + test sets ===")
    if args.prepared_root is None:
        raise SystemExit("--prepared-root is required for the calib step "
                         "(the folder written by utils/prepare_dataset.py)")

    head = load_head(paths.head)
    label_keys = [str(key) for key in head["label_keys"]]
    task = str(head["task"])

    # Read labels as text so numeric folder names survive (same as dataset.py).
    manifest = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    rng = np.random.default_rng(args.seed)

    # ---- calibration set: TRAINING split only -------------------------------------
    # Quantization picks each layer's value range from these images, so they must look
    # like what the chip will see: both the whole image and the patches, every class.
    train = manifest[manifest["split"] == "training"]

    # Shuffle inside each class, then interleave the classes (a, b, c, d, a, b, ...) so
    # that any prefix of the list is class-balanced.
    per_class = [group.sample(frac=1.0, random_state=args.seed).to_dict("records")
                 for _, group in train.groupby("label")]
    ordered = [record for row in zip_longest(*per_class) for record in row if record is not None]

    # The whole-image view is 1 of 17 encoder inputs but drives 1/3 of the head input,
    # so give it more weight than 1/17 in calibration.
    want_wholes = min(len(ordered), int(args.calib_size * args.calib_whole_fraction))
    want_patches = args.calib_size - want_wholes
    patches_per_record = int(np.ceil(want_patches / len(ordered)))
    if patches_per_record > VIEWS - 1:
        patches_per_record = VIEWS - 1
        print(f"Dataset is small: using every patch ({len(ordered) * (VIEWS - 1)} available).")

    images = []
    for index, record in enumerate(ordered):
        views = load_views(args.prepared_root, record)
        if index < want_wholes:
            images.append(views[0])
        # A different random subset of the 16 patches for every record.
        picks = rng.choice(np.arange(1, VIEWS), size=patches_per_record, replace=False)
        images.extend(views[picks])
    calib = np.stack(images)
    calib = calib[rng.permutation(len(calib))][:args.calib_size]  # shuffle once, here
    np.save(paths.calib, calib)   # uint8 on disk = 4x smaller than float32
    print(f"calibration set: {calib.shape} uint8, range [{calib.min()}, {calib.max()}] -> {paths.calib}")
    if len(calib) < 1024:
        print("WARNING: Hailo recommends >= 1024 calibration images (finetune needs 1024).")

    # ---- test set: splits the model never trained on -------------------------------
    test = manifest[manifest["split"].isin(args.test_splits.split(","))].to_dict("records")
    if not test:
        raise SystemExit(f"No manifest rows in splits {args.test_splits!r}")
    targets = targets_of(test, task, label_keys)
    np.savez(
        paths.test,
        images=np.stack([load_views(args.prepared_root, record) for record in test]),  # [M,17,224,224,3]
        targets=targets,
        names=np.array([PureWindowsPath(record["image_path"]).name for record in test]),
    )
    print(f"test set: {len(test)} samples x {VIEWS} views -> {paths.test}")


# =============================================================================
# Hailo steps. TensorFlow is imported BEFORE the Hailo SDK on purpose: the DFC guide
# notes that inside the Hailo docker the GPU is only usable in that import order.
# =============================================================================

def hailo_imports():
    import os

    # On import, the Hailo SDK only picks a GPU whose memory is < 5% used, and otherwise
    # HIDES all GPUs (CUDA_VISIBLE_DEVICES=99). On a laptop Windows already uses ~350 MB of
    # the GPU for the screen (8% of 4 GB), so the GPU was always rejected. Choosing GPU 0
    # ourselves skips that check. Set CUDA_VISIBLE_DEVICES yourself to override this.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    try:
        import tensorflow as tf
        from hailo_sdk_client import ClientRunner, InferenceContext
    except ImportError as error:
        raise SystemExit(
            f"{error}\nThe parse/optimize/test/compile steps need the Hailo Dataflow "
            "Compiler. Run them inside the Hailo AI SW Suite docker."
        ) from error

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow sees {len(gpus)} GPU(s)")
    return ClientRunner, InferenceContext, len(gpus) > 0


def emulate(runner, context, images):
    """Run the Hailo emulator on [N,224,224,3] images and return [N, 576] features."""
    with runner.infer_context(context) as ctx:
        result = runner.infer(ctx, images)
    return np.asarray(result).reshape(len(images), -1)


def layers_of_type(runner, layer_type):
    """Layer names in the parsed Hailo network (HN) that have the given type."""
    layers = runner.get_hn_dict()["layers"]
    return [name for name, layer in layers.items() if layer["type"] == layer_type]


# =============================================================================
# STEP 3: parse  (ONNX -> HAR)
# =============================================================================

def step_parse(args, paths):
    ClientRunner, InferenceContext, _ = hailo_imports()
    print("\n=== parse: ONNX -> Hailo HAR ===")

    # hw_arch must match the chip on the Pi: "hailo8l" for the Raspberry Pi AI Kit /
    # AI HAT+ 13 TOPS, "hailo8" for the 26 TOPS HAT.
    runner = ClientRunner(hw_arch=args.hw_arch)

    # Translate the ONNX graph into Hailo's internal network. No start/end nodes are
    # given: the whole file is the encoder, which is what the DFC guide suggests trying
    # first. The input shape is static, so net_input_shapes is not needed either.
    runner.translate_onnx_model(str(paths.onnx), args.name)
    runner.save_har(str(paths.parsed_har))

    print("input layers :", layers_of_type(runner, "input_layer"))
    print("output layers:", layers_of_type(runner, "output_layer"))

    # ---- verification: Hailo's float model must equal ONNX Runtime -----------------
    # SDK_NATIVE runs the parsed model as-is: float32, no normalization layer yet, so
    # it gets *normalized* images, in NHWC order (Hailo is channels-last).
    test = np.load(paths.test, allow_pickle=False)
    images = test["images"][:2].reshape(-1, INPUT_SIZE, INPUT_SIZE, 3)   # 2 samples = 34 views
    native = emulate(runner, InferenceContext.SDK_NATIVE, normalize(images))
    reference = onnx_features(paths.onnx, images)
    value = snr_db(reference, native)
    print(f"Hailo native vs ONNX Runtime: {value:.1f} dB (should be > 60 dB)")
    if value < 40:
        raise VerificationError("Parsed model differs from the ONNX model. Do not continue.")
    print(f"saved {paths.parsed_har}")


# =============================================================================
# STEP 4: optimize  (model script + quantization)
# =============================================================================

def build_model_script(args, runner, calib_count, has_gpu, output_16bit):
    """Write the .alls model script: the commands that steer optimization/compilation."""
    output_layer = layers_of_type(runner, "output_layer")[0]
    batch = args.batch_size
    lines = [
        # --- model modification ---------------------------------------------------
        # Put the ImageNet normalization ON THE CHIP. From here on the model input is
        # the raw 0..255 RGB image, so the Pi does no float math on the pixels and the
        # calibration set must NOT be normalized.
        f"normalization1 = normalization({MEAN_255}, {STD_255})",

        # --- how hard to optimize -------------------------------------------------
        # optimization_level: 0 equalization only | 1 + bias correction | 2 + finetune
        #                     3 + AdaRound (256 images) | 4 + AdaRound (1024 images)
        # compression_level=0: never use 4-bit weights. This model is tiny; 4-bit would
        # only cost accuracy.
        f"model_optimization_flavor(optimization_level={args.opt_level}, compression_level=0, batch_size={batch})",

        # --- calibration ----------------------------------------------------------
        # Number of images used to measure each layer's value range (default is only 64).
        f"model_optimization_config(calibration, batch_size={batch}, calibset_size={min(args.calibset_size, calib_count)})",

        # Print the output SNR at the end of optimize(), measured on 64 images
        # (the default 16 is a noisy estimate).
        "model_optimization_config(checker_cfg, policy=enabled, dataset_size=64)",
    ]

    if output_16bit:
        # 16-bit output instead of 8-bit. The output is a feature vector that the CPU
        # head keeps computing with, so its precision matters a lot. It is also the
        # result of averaging 7x7=49 values: at 8 bits that extra precision is thrown
        # away, at 16 bits it is kept. Costs nothing but 576 extra bytes per inference.
        lines.append(f"quantization_param({output_layer}, precision_mode=a16_w16)")

    layers_16bit = [f"{args.name}/{layer.strip()}" for layer in args.layers_16bit.split(",") if layer.strip()]
    if layers_16bit:
        # 16-bit activations + weights for the layers the noise analysis blames most.
        # Costs some speed on the chip. Not every layer type supports it (the DFC guide
        # lists conv, dw, avgpool, dense, add, activations...; NOT ew_mult).
        lines.append(f"quantization_param([{', '.join(layers_16bit)}], precision_mode=a16_w16)")

    if args.avgpool_reduction:
        # The squeeze-excite blocks average a whole feature map (up to 56x56) into one
        # value. This adds a first average-pool stage so the reduction happens in two
        # smaller steps (used in your earlier optimize_rice_bran.py).
        for layer in layers_of_type(runner, "avgpool"):
            lines.append(f"pre_quantization_optimization(global_avgpool_reduction, layers={layer}, "
                         "division_factors=[7, 7])")

    if args.activation_clipping:
        # Ignore the most extreme 0.01% of activations when choosing ranges. Helps when
        # rare outliers stretch a layer's range; try it if the SNR is low.
        lines.append("pre_quantization_optimization(activation_clipping, layers={*}, "
                     "mode=percentile, clipping_values=[0.01, 99.99])")

    # The flavor above is silently lowered to 0 by the DFC when it finds no GPU. Naming
    # the algorithm explicitly makes it run anyway (slow on CPU, but same result).
    if args.opt_level == 1:
        # Iterative bias correction: fixes the small constant offset quantization adds
        # to each layer. Recommended by Hailo for depthwise layers (MobileNet is full of them).
        lines.append("post_quantization_optimization(bias_correction, policy=enabled)")
    elif args.opt_level == 2:
        # Finetune (QFT): briefly re-trains the quantized weights so the quantized
        # output matches the float output. The loss is measured on the output layer by
        # default, i.e. it directly maximizes the SNR of our feature vector.
        dataset_size = min(args.finetune_images, calib_count)
        lines.append(
            "post_quantization_optimization(finetune, policy=enabled, "
            f"dataset_size={dataset_size}, epochs={args.finetune_epochs}, "
            f"learning_rate={args.finetune_lr}, batch_size={batch})"
        )
    elif args.opt_level >= 3:
        # AdaRound: learns, layer by layer, whether each weight rounds up or down.
        # mode=train_all: train every layer (the default only trains 4-bit layers, and
        # we have none). Slowest option; needs a GPU to be practical.
        dataset_size = min(1024 if args.opt_level == 4 else 256, calib_count)
        lines.append(
            "post_quantization_optimization(adaround, policy=enabled, mode=train_all, "
            f"dataset_size={dataset_size}, batch_size={batch})"
        )

    if args.compile_max:
        # Compilation-stage command (ignored during optimize): search harder for the
        # fastest on-chip layout. Does not change accuracy, makes compile much slower.
        lines.append("performance_param(compiler_optimization_level=max)")

    if args.extra_alls:
        # Your own extra commands, e.g. fixes for layers the noise analysis points at.
        lines.append(Path(args.extra_alls).read_text().strip())

    if args.opt_level >= 2 and not has_gpu:
        print("WARNING: no GPU visible. Level >= 2 still runs, but can take hours on CPU.")
    return "\n".join(lines) + "\n"


def step_optimize(args, paths):
    ClientRunner, InferenceContext, has_gpu = hailo_imports()
    print("\n=== optimize: model script + quantization ===")

    # Calibration images: float32 array [N, H, W, C] with values 0..255 (NOT normalized,
    # because the normalization layer is now part of the model).
    calib = np.load(paths.calib).astype(np.float32)
    assert calib.ndim == 4 and calib.shape[1:] == (INPUT_SIZE, INPUT_SIZE, 3), calib.shape
    assert calib.max() > 1.5, "calibration images must be 0..255, not 0..1"
    print(f"calibration set: {calib.shape}, range [{calib.min():.0f}, {calib.max():.0f}]")

    test = np.load(paths.test, allow_pickle=False)
    check_images = test["images"][:2].reshape(-1, INPUT_SIZE, INPUT_SIZE, 3)
    reference = onnx_features(paths.onnx, check_images)

    def optimize_with(output_16bit):
        runner = ClientRunner(har=str(paths.parsed_har))   # always start from the parsed model
        script = build_model_script(args, runner, len(calib), has_gpu, output_16bit)
        paths.alls.write_text(script)
        print(f"--- model script ({paths.alls}) ---\n{script}-----------------------")
        runner.load_model_script(script)

        # Apply only the float-domain changes (normalization layer, layer fusing,
        # equalization) so we can test them before any quantization happens.
        runner.optimize_full_precision(calib)

        # SDK_FP_OPTIMIZED = the modified float model. It now takes raw 0..255 images.
        # If this does not match ONNX Runtime, the normalization values or the channel
        # order are wrong, and quantizing would only hide that bug.
        fp = emulate(runner, InferenceContext.SDK_FP_OPTIMIZED, check_images.astype(np.float32))
        value = snr_db(reference, fp)
        print(f"Hailo FP-optimized (0..255 input) vs ONNX Runtime: {value:.1f} dB (should be > 40 dB)")
        if value < 30:
            raise VerificationError("Model with on-chip normalization differs from the original.")

        # The real work: collect statistics on the calibration set, convert weights and
        # activations to integers, then run the accuracy-recovery algorithm chosen above.
        runner.optimize(calib)
        return runner

    try:
        runner = optimize_with(output_16bit=args.output_16bit)
    except VerificationError:
        raise                    # a real conversion bug; 8-bit output would not fix it
    except Exception as error:   # noqa: BLE001 - the DFC raises many different error types
        if not args.output_16bit:
            raise
        print(f"Optimization with a 16-bit output failed ({error}).\nRetrying with an 8-bit output...")
        runner = optimize_with(output_16bit=False)

    runner.save_har(str(paths.quantized_har))
    print(f"saved {paths.quantized_har}")

    # A new quantized model makes the old headfit results wrong: delete them so the
    # headfit step re-computes them from this model.
    for stale in [paths.hailo_head, *paths.out.glob(f"{args.name}_quantized_*_features.npy")]:
        if stale.exists():
            stale.unlink()
            print(f"removed outdated {stale.name}")


# =============================================================================
# STEP 5: headfit  (re-train the small CPU head on what the CHIP outputs)
# =============================================================================
#
# WHY THIS HELPS
# The encoder is frozen (train.py only trains the head), so the .hef encoder is fixed and
# only the head decides the answer. The original head was trained on exact float
# features; the chip gives slightly different (quantized) features, and the head reads
# that difference as signal. Here we re-run train.py's head training, with the same
# recipe, but on the features of the quantized Hailo emulator. The head then learns what
# the chip really produces. The .hef does not change, only the small .npz on the CPU.
#
# Only the training split is used to train and the validation split to pick the best
# epoch (exactly like train.py). The testing split stays untouched for the test step.

def hailo_features(runner, context, prepared_root, records, cache_path):
    """[records, 17, 576] features of the Hailo emulator, cached on disk (slow on CPU)."""
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if len(cached) == len(records):
            print(f"using cached features {cache_path}")
            return cached
    features = []
    for start in range(0, len(records), 16):   # 16 samples = 272 images per emulator call
        chunk = records[start:start + 16]
        views = np.stack([load_views(prepared_root, record) for record in chunk])
        flat = views.reshape(-1, INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32)
        features.append(emulate(runner, context, flat).reshape(len(chunk), VIEWS, -1))
        print(f"  emulated {start + len(chunk)}/{len(records)} samples", flush=True)
    features = np.concatenate(features)
    np.save(cache_path, features)
    return features


def targets_of(records, task, label_keys):
    """Manifest rows -> training targets, the same encoding as dataset.py."""
    if task == "classification":
        return np.array([label_keys.index(record["label"]) for record in records], dtype=np.int64)
    # Regression labels look like "R5 S0": first character = key, rest = percentage.
    parsed = [{part[0]: float(part[1:]) for part in record["label"].split()} for record in records]
    return np.array([[label[key] for key in label_keys] for label in parsed], dtype=np.float32)


def train_head(head, train_x, train_y, val_x, val_y, task, args):
    """train.py's head training (Adam 1e-3, wd 1e-4, batch 32, best epoch on validation).

    Starts from the original head weights, so it adapts the head instead of learning
    from zero, and keeps the epoch with the best validation score.
    """
    import torch
    from torch import nn

    torch.manual_seed(args.seed)
    model = nn.Sequential(nn.Linear(train_x.shape[1], 128), nn.ReLU(), nn.Linear(128, head["w1"].shape[0]))
    with torch.no_grad():
        for layer, (w, b) in zip((model[0], model[2]), (("w0", "b0"), ("w1", "b1"))):
            layer.weight.copy_(torch.from_numpy(head[w]))
            layer.bias.copy_(torch.from_numpy(head[b]))

    classification = task == "classification"
    loss_function = nn.CrossEntropyLoss() if classification else nn.SmoothL1Loss(beta=5.0)  # same as train.py
    optimizer = torch.optim.Adam(model.parameters(), lr=args.headfit_lr, weight_decay=1e-4)
    train_x, val_x = torch.from_numpy(train_x).float(), torch.from_numpy(val_x).float()
    train_y, val_y = torch.from_numpy(train_y), torch.from_numpy(val_y)

    def score():
        """Lower is better: validation loss (classification) or RMSE (regression), like train.py."""
        model.eval()
        with torch.no_grad():
            prediction = model(val_x)
            if classification:
                return loss_function(prediction, val_y).item()
            return (prediction - val_y).square().mean().sqrt().item()

    best_score, best_epoch = score(), 0          # epoch 0 = the original head
    best_state = {key: value.clone() for key, value in model.state_dict().items()}
    print(f"original head on Hailo features: validation score {best_score:.4f}")
    for epoch in range(1, args.headfit_epochs + 1):
        model.train()
        for indices in torch.randperm(len(train_x)).split(32):
            optimizer.zero_grad(set_to_none=True)
            loss_function(model(train_x[indices]), train_y[indices]).backward()
            optimizer.step()
        value = score()
        if value < best_score:
            best_score, best_epoch = value, epoch
            best_state = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    print(f"best validation score {best_score:.4f} at epoch {best_epoch}/{args.headfit_epochs}")
    model.load_state_dict(best_state)
    return {
        "w0": model[0].weight.detach().numpy(), "b0": model[0].bias.detach().numpy(),
        "w1": model[2].weight.detach().numpy(), "b1": model[2].bias.detach().numpy(),
    }, best_score, best_epoch


def step_headfit(args, paths):
    import pandas as pd

    ClientRunner, InferenceContext, _ = hailo_imports()
    print("\n=== headfit: re-train the CPU head on the quantized encoder's features ===")
    if args.prepared_root is None:
        raise SystemExit("--prepared-root is required for the headfit step")

    runner = ClientRunner(har=str(paths.quantized_har))
    head = load_head(paths.head)
    task = str(head["task"])
    label_keys = [str(key) for key in head["label_keys"]]
    manifest = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)

    data = {}
    for split in ("training", "validation"):
        records = manifest[manifest["split"] == split].to_dict("records")
        if not records:
            raise SystemExit(f"No {split} rows in {args.manifest}")
        print(f"{split}: {len(records)} samples through the quantized emulator")
        features = hailo_features(runner, InferenceContext.SDK_QUANTIZED, args.prepared_root,
                                  records, paths.out / f"{args.name}_quantized_{split}_features.npy")
        data[split] = (pool_views(features).astype(np.float32), targets_of(records, task, label_keys))

    weights, best_score, best_epoch = train_head(head, *data["training"], *data["validation"], task, args)
    np.savez(paths.hailo_head, **weights, label_keys=head["label_keys"], task=head["task"])
    print(f"saved {paths.hailo_head}  (use this one with the .hef)")


# =============================================================================
# STEP 6: test  (float vs quantized, on held-out samples)
# =============================================================================

def step_test(args, paths):
    ClientRunner, InferenceContext, _ = hailo_imports()
    print("\n=== test: quantized model vs float model ===")

    runner = ClientRunner(har=str(paths.quantized_har))
    head = load_head(paths.head)
    task = str(head["task"])
    label_keys = [str(key) for key in head["label_keys"]]

    test = np.load(paths.test, allow_pickle=False)
    samples = test["images"]                                   # [M, 17, 224, 224, 3] uint8
    targets = test["targets"]
    count = len(samples)
    flat = samples.reshape(-1, INPUT_SIZE, INPUT_SIZE, 3)      # [M*17, 224, 224, 3]
    raw = flat.astype(np.float32)                              # 0..255, what the chip gets

    # Three versions of the same encoder on the same images:
    features = {
        # 1. the original float model (ONNX Runtime) = ground truth
        "onnx": onnx_features(paths.onnx, flat),
        # 2. Hailo float model after the model script (normalization, equalization)
        "fp_optimized": emulate(runner, InferenceContext.SDK_FP_OPTIMIZED, raw),
        # 3. Hailo quantized model: integer emulation of what the chip will compute
        #    (a close approximation of the hardware, not bit-exact)
        "quantized": emulate(runner, InferenceContext.SDK_QUANTIZED, raw),
    }
    features = {key: value.reshape(count, VIEWS, -1) for key, value in features.items()}
    outputs = {key: predict(head, value) for key, value in features.items()}

    truth, quant = features["onnx"], features["quantized"]
    report = {
        "samples": count,
        # Was anything lost before quantization? (should be very high)
        "fp_optimized_vs_onnx_db": snr_db(truth, features["fp_optimized"]),
        # The numbers that matter: quantized vs the original float model.
        "encoder_snr_db": snr_db(truth, quant),
        "encoder_snr_whole_view_db": snr_db(truth[:, 0], quant[:, 0]),
        "encoder_snr_patch_views_db": snr_db(truth[:, 1:], quant[:, 1:]),
        "pooled_1728_snr_db": snr_db(pool_views(truth), pool_views(quant)),
        "model_output_snr_db": snr_db(outputs["onnx"], outputs["quantized"]),
        "model_output_max_abs_diff": float(np.abs(outputs["onnx"] - outputs["quantized"]).max()),
    }

    # The head re-trained on the chip's features (headfit step), if it exists.
    # This is the one to judge the .hef by: it is what will run on the Pi.
    if paths.hailo_head.exists():
        outputs["quantized_hailo_head"] = predict(load_head(paths.hailo_head), quant)

    if task == "classification":
        float_class = outputs["onnx"].argmax(axis=1)
        report["float_accuracy"] = float((float_class == targets).mean())
        # Margin = top score minus runner-up. If quantization noise (max abs diff) is
        # far below the smallest margin, the decision is safe.
        ordered = np.sort(outputs["onnx"], axis=1)
        report["float_smallest_margin"] = float((ordered[:, -1] - ordered[:, -2]).min())
        for key in ("quantized", "quantized_hailo_head"):
            if key not in outputs:
                continue
            quant_class = outputs[key].argmax(axis=1)
            report[f"{key}_accuracy"] = float((quant_class == targets).mean())
            # Agreement: how often the quantized model picks the same class as the float one.
            report[f"{key}_class_agreement"] = float((float_class == quant_class).mean())
            for index in np.flatnonzero(float_class != quant_class):
                print(f"  DISAGREE ({key}) {test['names'][index]}: float={label_keys[float_class[index]]} "
                      f"quantized={label_keys[quant_class[index]]}")
    else:
        # Regression: errors in percentage points (MAE = mean absolute error).
        report["float_mae"] = float(np.abs(outputs["onnx"] - targets).mean())
        report["float_rmse"] = float(np.sqrt(np.mean((outputs["onnx"] - targets) ** 2)))
        for key in ("quantized", "quantized_hailo_head"):
            if key not in outputs:
                continue
            report[f"{key}_mae"] = float(np.abs(outputs[key] - targets).mean())
            report[f"{key}_rmse"] = float(np.sqrt(np.mean((outputs[key] - targets) ** 2)))
        # Only meaningful for the original head: how far the chip moves its predictions.
        report["quantized_vs_float_mae"] = float(np.abs(outputs["onnx"] - outputs["quantized"]).mean())
        # Per-label mean prediction, e.g. does R10 still come out near 10?
        for value in np.unique(targets[:, 0]):
            rows = targets[:, 0] == value
            report[f"mean_prediction_at_{value:g}"] = {
                key: round(float(output[rows, 0].mean()), 3) for key, output in outputs.items()}

    if args.analyze_noise:
        # Layer noise analysis: quantize ONE layer at a time (rest stays float) and
        # measure the SNR at the output. Low values point at the layers that hurt.
        # The result is also stored in the HAR for `hailo profiler <har>`.
        try:
            runner.analyze_noise(raw, batch_size=2, data_count=64)
            runner.save_har(str(paths.quantized_har))
            statistics = runner.get_params_statistics()
            output_layer = layers_of_type(runner, "output_layer")[0]
            per_layer = {}
            for layer in runner.get_hn_model():
                value = statistics.get(f"{layer.name}/layer_noise_analysis/noise_results/{output_layer}")
                if value is not None:
                    per_layer[layer.name] = float(np.asarray(value).reshape(-1)[0])
            worst = sorted(per_layer.items(), key=lambda item: item[1])[:8]
            report["worst_layers_snr_db"] = dict(worst)
            print("Layers that add the most noise (lowest SNR first):")
            for name, value in worst:
                print(f"  {value:6.1f} dB  {name}")
        except Exception as error:   # noqa: BLE001 - diagnostics only, never fatal
            print(f"analyze_noise skipped: {error}")

    paths.report.write_text(json.dumps(report, indent=2))
    print("\n--- report ---")
    for key, value in report.items():
        print(f"{key:32s} {value if isinstance(value, dict) else round(value, 4)}")
    print(f"saved {paths.report}")
    if task == "classification" and report["quantized_class_agreement"] < 1.0:
        print("WARNING: quantized and float models disagree on some samples.")


# =============================================================================
# STEP 6: compile  (quantized HAR -> HEF)
# =============================================================================

def step_compile(args, paths):
    ClientRunner, _, _ = hailo_imports()
    print("\n=== compile: quantized HAR -> HEF ===")

    # The quantized HAR already carries the chip type and the model script.
    runner = ClientRunner(har=str(paths.quantized_har))

    # Place the layers onto the chip's resources and produce the binary. Returns bytes.
    hef = runner.compile()
    paths.hef.write_bytes(hef)

    # Keeps the allocation results; open with `hailo profiler <file>` for an HTML report.
    runner.save_har(str(paths.compiled_har))
    print(f"saved {paths.hef} ({len(hef) / 1e6:.1f} MB)")
    head = paths.hailo_head if paths.hailo_head.exists() else paths.head
    print(f"Copy {paths.hef.name} and {head.name} to the Pi. "
          "See hef_inference_example.py for how to use them together.")


# =============================================================================

STEPS = {
    "export": step_export, "calib": step_calib, "parse": step_parse,
    "optimize": step_optimize, "headfit": step_headfit, "test": step_test,
    "compile": step_compile,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", default=",".join(STEPS),
                        help=f"comma-separated subset of: {','.join(STEPS)}")
    parser.add_argument("--name", default="rawmat_encoder", help="network / file name")
    parser.add_argument("--out-dir", default=str(HERE / "build"))

    group = parser.add_argument_group("inputs")
    group.add_argument("--checkpoint", default=str(PROJECT / "models" / "pth" / "auto.pth"))
    group.add_argument("--manifest", default=str(PROJECT / "raw_material_manifest_classification.csv"))
    group.add_argument("--prepared-root", default=None,
                       help="prepared dataset folder (config.PREPROCESSED_DATASET_DIR)")

    group = parser.add_argument_group("export")
    group.add_argument("--opset", type=int, default=13, help="ONNX opset (DFC supports 8, 11-17)")

    group = parser.add_argument_group("calibration / test data")
    group.add_argument("--calib-size", type=int, default=2048, help="images in the calibration set")
    group.add_argument("--calib-whole-fraction", type=float, default=0.25,
                       help="target share of whole-image views (capped by what exists)")
    group.add_argument("--test-splits", default="testing", help="e.g. testing or validation,testing")
    group.add_argument("--seed", type=int, default=42)

    group = parser.add_argument_group("optimization")
    group.add_argument("--hw-arch", default="hailo8l", help="hailo8l (Pi AI Kit) or hailo8")
    group.add_argument("--opt-level", type=int, default=2, choices=range(5),
                       help="0 fastest ... 4 best/slowest (2 = finetune)")
    group.add_argument("--batch-size", type=int, default=8, help="lower it if the GPU runs out of memory")
    group.add_argument("--calibset-size", type=int, default=1024, help="images used for range statistics")
    group.add_argument("--finetune-images", type=int, default=2048)
    group.add_argument("--finetune-epochs", type=int, default=8)
    group.add_argument("--finetune-lr", type=float, default=0.0001)
    group.add_argument("--no-output-16bit", dest="output_16bit", action="store_false",
                       help="keep the feature output at 8 bits")
    group.add_argument("--activation-clipping", action="store_true")
    group.add_argument("--layers-16bit", default="",
                       help="comma-separated layer names to run in 16-bit, e.g. conv52,avgpool10 "
                            "(pick them from the --analyze-noise list)")
    group.add_argument("--avgpool-reduction", action="store_true",
                       help="split the big squeeze-excite average pools in two (Hailo tip for MobileNetV3)")

    group = parser.add_argument_group("headfit")
    group.add_argument("--headfit-epochs", type=int, default=300, help="same as config.EPOCHS")
    group.add_argument("--headfit-lr", type=float, default=1e-3, help="same as train.py")
    group.add_argument("--extra-alls", default=None, help="file with extra model script commands")
    group.add_argument("--analyze-noise", action="store_true", help="per-layer SNR in the test step (slow)")
    group.add_argument("--compile-max", action="store_true", help="slow compile, highest FPS")
    args = parser.parse_args()

    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    unknown = [step for step in steps if step not in STEPS]
    if unknown:
        parser.error(f"unknown steps: {unknown}")

    paths = Paths(args)
    for name in STEPS:              # always run in pipeline order
        if name in steps:
            STEPS[name](args, paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
