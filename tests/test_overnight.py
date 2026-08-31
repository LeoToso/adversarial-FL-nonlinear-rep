import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from core.fed_fedrep import FedRep
from core.models import FedModel, LinearHead
from scripts.run_femnist_overnight import configurations
from experiments.run_experiment import run_experiment


def tiny(*args, **kwargs):
    return FedModel(nn.Sequential(nn.Flatten(), nn.Linear(4, 8)), LinearHead(8, 62))


class OvernightTests(unittest.TestCase):
    def test_grid_is_80_unique_configs_with_unchanged_baseline(self):
        configs = configurations()
        self.assertEqual(len(configs), 80)
        self.assertEqual(len({(c['loss_type'], c['algorithm'], c['aggregator'],
                              c['attack'], c['seed']) for c in configs}), 80)
        for c in configs:
            self.assertEqual(c['rounds'], 200)
            self.assertEqual(c['head_optimizer'], 'sgd' if c['algorithm'] == 'baseline' else 'adam_reset')

    def test_production_adam_is_recreated_each_round(self):
        with patch('core.fed_fedrep.build_model', side_effect=tiny):
            trainer = FedRep('femnist', 1, 0, lambda v: v[0], lambda v: [],
                             head_steps=2, head_optimizer='adam_reset', lr_head=.001)
        loader = DataLoader(TensorDataset(torch.rand(4,4), torch.tensor([0,1,0,1])), batch_size=4)
        original, created = torch.optim.Adam, []
        def factory(*args, **kwargs):
            opt = original(*args, **kwargs)
            created.append(opt)
            return opt
        with patch('torch.optim.Adam', side_effect=factory):
            trainer.train_round([loader])
            trainer.train_round([loader])
        self.assertEqual(len(created), 2)
        self.assertTrue(all(int(s['step']) == 2 for o in created for s in o.state.values()))

    def test_every_smoke_case_runs_and_saves_optimizer_metadata(self):
        loader = DataLoader(TensorDataset(torch.rand(4,4), torch.tensor([0,1,2,3])), batch_size=4)
        for cfg in configurations(smoke=True):
            with self.subTest(cfg=cfg), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.pt'
                with patch('core.fed_fedrep.build_model', side_effect=tiny), \
                     patch('core.fed_baseline.build_model', side_effect=tiny), \
                     patch('experiments.run_experiment.get_loaders', return_value=([loader]*10,[loader]*10,loader)):
                    result = run_experiment(**cfg, device='cpu', verbose=False, checkpoint_path=str(path))
                self.assertEqual(len(result.history), 2)
                checkpoint = torch.load(path, weights_only=True)
                self.assertEqual(checkpoint['config']['head_optimizer'],
                                 None if cfg['algorithm'] == 'baseline' else 'adam_reset')
