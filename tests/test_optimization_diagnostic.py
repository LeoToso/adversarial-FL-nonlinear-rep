import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from core.models import FedModel, LinearHead
from core.objectives import TaskObjective
from scripts.diagnose_femnist_optimization import head_probe, run_case, score_report


class OptimizationDiagnosticTests(unittest.TestCase):
    def loader(self):
        return DataLoader(TensorDataset(torch.randn(8, 4), torch.arange(8) % 2), batch_size=4)

    def model(self, *args):
        return FedModel(nn.Linear(4, 3), LinearHead(3, 62))

    def test_collapse_counts(self):
        out = score_report(torch.tensor([[4., 0.], [4., 0.]]), torch.tensor([0, 1]),
                           TaskObjective("classification", "cross_entropy", 2))
        self.assertEqual(out["prediction_histogram"], [2, 0])
        self.assertEqual(out["dominant_prediction_fraction"], 1.)

    def test_probe_preserves_parameters_and_logs_both_optimizers(self):
        model = self.model()
        before = copy.deepcopy(model.state_dict())
        out = head_probe(model, self.loader(), TaskObjective("classification", "multiclass_ls", 62), "cpu", 2)
        self.assertEqual(set(out["fits"]), {"SGD", "Adam"})
        for key in before:
            torch.testing.assert_close(before[key], model.state_dict()[key], rtol=0, atol=0)

    def test_instrumented_rounds_save_finite_gradients_and_checkpoint(self):
        for loss in ("cross_entropy", "multiclass_ls"):
            with tempfile.TemporaryDirectory() as directory, patch(
                    "core.fed_fedrep.build_model", side_effect=self.model):
                path = Path(directory)
                run_case([self.loader()], loss, 2, 2, 2, "cpu", 42, path)
                report = json.loads((path / "optimization.json").read_text())
                self.assertEqual(len(report["history"]), 2)
                self.assertGreater(report["history"][0]["backbone_update_norm"], 0)
                self.assertGreater(report["history"][0]["backbone_raw_gradient_norms"][0], 0)
                self.assertTrue((path / "final_model.pt").exists())
