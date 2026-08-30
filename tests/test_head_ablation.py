import copy
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from core.fed_fedrep import FedRep
from core.models import FedModel, LinearHead
from scripts.diagnose_femnist_head_ablation import diagnostic_round


class HeadAblationTests(unittest.TestCase):
    def test_phase_modes_and_matched_minibatches(self):
        def build(*args):
            return FedModel(nn.Sequential(nn.Linear(4, 3), nn.Dropout(.3)), LinearHead(3, 62))
        torch.manual_seed(42)
        with patch("core.fed_fedrep.build_model", side_effect=build):
            base = FedRep("femnist", 1, 0, None, None, repr_dim=3, head_steps=2,
                          loss_type="multiclass_ls")
        loader = DataLoader(TensorDataset(torch.arange(32).reshape(8,4).float()/32,
                                         torch.arange(8)%2), batch_size=4)
        cases = []
        for optimizer, dropout in (("SGD", True), ("SGD", False), ("Adam", False)):
            trainer = copy.deepcopy(base)
            seen = []
            model = trainer.client_models[0]
            handle = model.register_forward_pre_hook(lambda m, args: seen.append(
                (args[0].clone(), m.backbone[1].training)))
            row = diagnostic_round(trainer, [loader], optimizer, dropout, 42)
            handle.remove()
            self.assertEqual([mode for x, mode in seen], [dropout, dropout, True])
            self.assertGreater(row["backbone_update_norm"], 0)
            self.assertTrue(all(p.grad is None and p.requires_grad for p in model.parameters()))
            cases.append(seen)
        for seen in cases[1:]:
            for (x, _), (y, _) in zip(cases[0], seen):
                torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_real_cnn_round_both_optimizers(self):
        loader = DataLoader(TensorDataset(torch.rand(2,1,28,28), torch.tensor([0,1])), batch_size=2)
        for optimizer in ("SGD", "Adam"):
            trainer = FedRep("femnist", 1, 0, None, None, repr_dim=8, head_steps=1,
                             loss_type="multiclass_ls")
            row = diagnostic_round(trainer, [loader], optimizer, False, 42)
            self.assertGreater(row["backbone_update_norm"], 0)
            self.assertTrue(torch.isfinite(torch.tensor(row["mean_post_head_loss"])))
