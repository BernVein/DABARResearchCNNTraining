import json
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from utils.labels import infer_label_keys, parse_label


class MultiOutputTests(unittest.TestCase):
    def test_single_and_multiple_keys(self):
        self.assertEqual(infer_label_keys(["R0", "R5"]), ("R",))
        self.assertEqual(infer_label_keys(["R5 S10", "S0 R2.5"]), ("R", "S"))
        self.assertEqual(parse_label("R5 S10 X0"), {"R": 5, "S": 10, "X": 0})

    def test_missing_and_invalid_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same output keys"):
            infer_label_keys(["R5 S10", "R0"])
        for label in ("", "R5 R10", "Rnan", "Rinf", "R-1", "R101", "rice5"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                parse_label(label)

    def test_dataset_and_cache_preserve_output_columns(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is unavailable in this interpreter")
        from dataset import RawMaterialDataset
        from train import cache_features

        class FeatureEncoder(torch.nn.Module):
            def extract_features(self, whole, patches):
                return whole.new_zeros((len(whole), 1728))

        for labels, expected in (
            (["R5", "R0"], [[5.0], [0.0]]),
            (["R5 S10", "S0 R2.5"], [[5.0, 10.0], [2.5, 0.0]]),
        ):
            with self.subTest(labels=labels):
                frame = pd.DataFrame({
                    "image_path": ["one.png", "two.png"],
                    "label": labels,
                    "batch": ["batch_1", "batch_2"],
                    "split": ["training", "training"],
                })
                with patch("dataset.pd.read_csv", return_value=frame):
                    dataset = RawMaterialDataset(Path("manifest.csv"), "training")
                with (
                    patch("dataset.load_prepared_image", return_value=torch.zeros(
                        (3, 224, 224), dtype=torch.uint8
                    )),
                    patch("dataset.np.load", return_value=np.ones(16)),
                    patch("train.torch.amp.autocast", return_value=nullcontext()),
                ):
                    features, targets = cache_features(
                        FeatureEncoder(), dataset, torch.device("cpu")
                    )
                self.assertEqual(features.shape, (2, 1728))
                self.assertEqual(targets.tolist(), expected)
                head = torch.nn.Linear(1728, len(dataset.label_keys))
                loss = torch.nn.SmoothL1Loss()(head(features), targets)
                loss.backward()
                self.assertIsNotNone(head.weight.grad)

    def test_manifest_notebook_checks_multi_output_labels(self):
        notebook_path = Path(__file__).resolve().parent / "generate_manifest.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = {
            cell["id"]: "".join(cell["source"])
            for cell in notebook["cells"] if cell["cell_type"] == "code"
        }
        images = [
            Path("dataset") / label / f"batch_{batch}" / "sample.png"
            for label in ("R5 S10", "R0 S0") for batch in range(3)
        ]
        root = Mock()
        root.rglob.return_value = images
        namespace = {"DATASET_DIR": root, "infer_label_keys": infer_label_keys,
                     "pd": pd}
        with patch.object(Path, "is_file", return_value=True):
            exec(code_cells["c110eb6d"], namespace)
            exec(code_cells["cf985abc"], namespace)
        self.assertEqual(namespace["label_keys"], ("R", "S"))
        self.assertEqual(len(namespace["df"]), 6)
        self.assertEqual(set(namespace["df"]["split"]),
                         {"training", "validation", "testing"})


if __name__ == "__main__":
    unittest.main()
