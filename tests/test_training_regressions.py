"""Offline regressions: no real datasets, GPU, or ByzFL installation required."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from core import datasets as data
from core.fed_fedrep import FedRep
from core.models import FedModel, LinearHead
from core.objectives import TaskObjective


class FakeCIFAR:
    def __init__(self, root, train, download, transform):
        self.transform = transform
        rng = np.random.default_rng(5)
        self.data = rng.integers(0, 256, (100, 32, 32, 3), dtype=np.uint8)
        self.targets = [i % 10 for i in range(100)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.data[i])), self.targets[i]


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.meta = copy.deepcopy(data.DATASET_META)
        data._FEMNIST_LEAF_CACHE.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        data.DATASET_META.clear()
        data.DATASET_META.update(self.meta)
        data._FEMNIST_LEAF_CACHE.clear()
        self.tmp.cleanup()

    def write_leaf(self, train_labels=(0, 1), test_labels=(61,), pixel=0.5):
        for split, labels in (("train", train_labels), ("test", test_labels)):
            directory = self.root / split
            directory.mkdir(exist_ok=True)
            payload = {"users": ["writer"], "num_samples": [len(labels)],
                       "user_data": {"writer": {
                           "x": [[pixel] * 784 for _ in labels],
                           "y": list(labels)}}}
            (directory / "sample.json").write_text(json.dumps(payload))

    def test_leaf_keeps_62_outputs_even_for_low_training_labels(self):
        self.write_leaf()
        train, test, _ = data.load_femnist_leaf(str(self.root), n_clients=1)
        self.assertEqual(data.DATASET_META["femnist"]["n_classes"], 62)
        x, y = next(iter(test[0]))
        self.assertEqual(int(y[0]), 61)
        self.assertEqual(tuple(x.shape), (1, 1, 28, 28))
        self.assertTrue(torch.allclose(x, torch.full_like(x, (0.5 - 0.1307) / 0.3081)))
        loss = TaskObjective("classification", "cross_entropy", 62)(
            torch.zeros(1, 62), y)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(train[0].dataset), 2)

    def test_leaf_rejects_out_of_range_labels(self):
        for label in (-1, 62, 1.5):
            with self.subTest(label=label):
                data._FEMNIST_LEAF_CACHE.clear()
                self.write_leaf(test_labels=(label,))
                with self.assertRaisesRegex(ValueError, "label"):
                    data.load_femnist_leaf(str(self.root), n_clients=1)

    def test_leaf_and_letters_proxy_reset_metadata(self):
        self.write_leaf()
        data.load_femnist_leaf(str(self.root), n_clients=1)

        class FakeEMNIST:
            def __init__(self, *args, **kwargs):
                self.targets = torch.arange(1, 27).repeat(2)
            def __len__(self):
                return len(self.targets)

        with patch.object(data.datasets, "EMNIST", FakeEMNIST):
            data.load_femnist(1, 1.0, data_dir=str(self.root), use_leaf=False)
        self.assertEqual(data.DATASET_META["femnist"]["n_classes"], 26)
        data.load_femnist_leaf(str(self.root), n_clients=1)
        self.assertEqual(data.DATASET_META["femnist"]["n_classes"], 62)

    def test_leaf_rejects_invalid_pixels(self):
        for pixel in (255.0, float("nan")):
            with self.subTest(pixel=pixel):
                data._FEMNIST_LEAF_CACHE.clear()
                self.write_leaf(pixel=pixel)
                with self.assertRaisesRegex(ValueError, "pixel"):
                    data.load_femnist_leaf(str(self.root), n_clients=1)

    def test_leaf_rejects_mismatched_samples(self):
        self.write_leaf()
        path = self.root / "train" / "sample.json"
        payload = json.loads(path.read_text())
        payload["user_data"]["writer"]["x"].pop()
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "shape|sample"):
            data.load_femnist_leaf(str(self.root), n_clients=1)

    def test_cifar_heldout_transform_is_deterministic_and_disjoint(self):
        for dataset_name, loader in (("CIFAR10", data.load_cifar10),
                                     ("CIFAR100", data.load_cifar100)):
            with self.subTest(dataset=dataset_name), patch.object(
                    data.datasets, dataset_name, FakeCIFAR):
                train, test, _ = loader(2, 1.0, data_dir=self.tmp.name)
                for tr, te in zip(train, test):
                    self.assertFalse(set(tr.dataset.indices) & set(te.dataset.indices))
                    self.assertIsNot(tr.dataset.dataset, te.dataset.dataset)
                    train_tf = tr.dataset.dataset.transform.transforms
                    eval_tf = te.dataset.dataset.transform.transforms
                    self.assertTrue(any(isinstance(t, transforms.RandomCrop) for t in train_tf))
                    self.assertFalse(any(isinstance(t, (transforms.RandomCrop,
                                                       transforms.RandomHorizontalFlip))
                                         for t in eval_tf))
                    first = te.dataset[0][0]
                    for _ in range(5):
                        self.assertTrue(torch.equal(first, te.dataset[0][0]))


def tiny_model(dataset, repr_dim, n_classes, linear):
    return FedModel(nn.Sequential(nn.Flatten(), nn.Linear(4, repr_dim)),
                    LinearHead(repr_dim, n_classes))


class FedRepGradientTests(unittest.TestCase):
    def trainer(self, loss_type="cross_entropy"):
        with patch("core.fed_fedrep.build_model", side_effect=tiny_model):
            return FedRep("cifar10", 1, 0,
                          aggregator=lambda vectors: torch.stack(vectors).mean(0),
                          attack=lambda vectors: [], repr_dim=3, head_steps=2,
                          lr_head=0.1, loss_type=loss_type)

    def loaders(self):
        return [DataLoader(TensorDataset(torch.arange(16).reshape(4, 4).float() / 16,
                                        torch.tensor([0, 1, 0, 1])), batch_size=4)]

    def test_stale_backbone_gradients_do_not_change_head_updates(self):
        for loss_type in ("cross_entropy", "multiclass_ls"):
            with self.subTest(loss=loss_type):
                torch.manual_seed(7)
                clean = self.trainer(loss_type)
                stale = copy.deepcopy(clean)
                for p in stale.client_models[0].backbone.parameters():
                    p.grad = torch.full_like(p, 10000.0)
                clean.train_round(self.loaders())
                stale.train_round(self.loaders())
                for p, q in zip(clean.client_models[0].parameters(),
                                stale.client_models[0].parameters()):
                    torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_clipping_is_phase_specific_and_gradients_are_cleared(self):
        trainer = self.trainer()
        cm = trainer.client_models[0]
        head_ids = {id(p) for p in cm.head.parameters()}
        backbone_ids = {id(p) for p in cm.backbone.parameters()}
        calls = []
        original = torch.nn.utils.clip_grad_norm_

        def record(parameters, *args, **kwargs):
            parameters = list(parameters)
            calls.append({id(p) for p in parameters})
            return original(parameters, *args, **kwargs)

        with patch("torch.nn.utils.clip_grad_norm_", side_effect=record):
            for _ in range(2):
                trainer.train_round(self.loaders())
        self.assertEqual(calls, [head_ids, head_ids, backbone_ids] * 2)
        self.assertTrue(all(p.grad is None for p in cm.parameters()))
        self.assertTrue(all(p.requires_grad for p in cm.parameters()))


class RealCNNIntegrationTests(unittest.TestCase):
    def test_training_evaluation_and_checkpoint_for_both_losses(self):
        from experiments.run_experiment import run_experiment

        torch.manual_seed(19)
        train_loader = DataLoader(TensorDataset(torch.rand(4, 1, 28, 28),
                                               torch.tensor([0, 1, 2, 61])), batch_size=4)
        test_loader = DataLoader(TensorDataset(torch.rand(2, 1, 28, 28),
                                              torch.tensor([0, 61])), batch_size=2)
        for loss_type in ("cross_entropy", "multiclass_ls"):
            with self.subTest(loss=loss_type), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.pt"
                with patch("experiments.run_experiment.get_loaders",
                           return_value=([train_loader], [test_loader], test_loader)):
                    result = run_experiment(
                        dataset="femnist", alpha=1.0, n_clients=1, n_byzantine=0,
                        algorithm="fedrep_nonlinear", aggregator="Average",
                        attack="SignFlipping", loss_type=loss_type, repr_dim=8,
                        head_steps=1, rounds=2, eval_every=1, device="cpu",
                        lr_schedule="cosine", lr_min=0.001, verbose=False,
                        checkpoint_path=str(path))
                self.assertEqual(len(result.history), 2)
                self.assertEqual(result.train_history[0]["learning_rate"], 0.1)
                self.assertEqual(result.train_history[-1]["learning_rate"], 0.001)
                json.dumps(result.to_dict(), allow_nan=False)
                saved = torch.load(path, map_location="cpu", weights_only=True)
                self.assertEqual(saved["round"], 2)
                self.assertEqual(saved["client_head_state_dicts"][0]["fc.weight"].shape[0], 62)

    def test_repeated_evaluation_does_not_change_model_or_predictions(self):
        torch.manual_seed(12)
        trainer = FedRep("femnist", 1, 0,
                         aggregator=lambda vectors: torch.stack(vectors).mean(0),
                         attack=lambda vectors: [], repr_dim=8, head_steps=1)
        loader = DataLoader(TensorDataset(torch.rand(2, 1, 28, 28),
                                         torch.tensor([0, 61])), batch_size=2)
        trainer.train_round([loader])
        trainer.client_test_loaders = [loader]
        before = copy.deepcopy(trainer.client_models[0].state_dict())
        first = trainer.evaluate(loader)
        second = trainer.evaluate(loader)
        self.assertEqual(first, second)
        for key, tensor in trainer.client_models[0].state_dict().items():
            torch.testing.assert_close(before[key], tensor, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
