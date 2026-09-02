import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.collins_femnist import ArrayDataset, CollinsConfig, CollinsFEMNISTTrainer, CollinsMLP
from core.aggregators import ByzantineAttack


def test_capacity_fitting_preserves_minimum_and_class_balance():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts" / "prepare_collins_femnist.py"
    spec = importlib.util.spec_from_file_location("prepare_collins_femnist", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    requested = np.full(10, 60)
    per_class, assignments = module.fit_disjoint_class_capacity(
        requested, n_clients=10, classes_per_client=3,
        pool_sizes={label: 59 for label in range(10)}, min_train=50,
    )
    for label in range(10):
        used = sum(per_class[client]
                   for client, labels in enumerate(assignments)
                   if label in labels)
        assert used <= 59
    assert min(per_class) * 3 * 0.9 >= 50


def test_compatibility_circular_indices_have_requested_length():
    pool_length = 7
    start, stop = 5, 12
    indices = np.arange(start, stop) % pool_length
    assert indices.tolist() == [5, 6, 0, 1, 2, 3, 4]


def test_alie_matches_coordinatewise_mean_plus_population_std():
    honest = [
        torch.tensor([1.0, 2.0, 4.0]),
        torch.tensor([3.0, 4.0, 8.0]),
        torch.tensor([5.0, 8.0, 12.0]),
    ]
    tau = 1.5
    malicious = ByzantineAttack("ALIE", f=2, tau=tau)(honest)
    stacked = torch.stack(honest)
    expected = (
        stacked.mean(dim=0)
        + tau * stacked.std(dim=0, unbiased=False)
    )
    assert len(malicious) == 2
    assert all(torch.allclose(vector, expected) for vector in malicious)


def tiny_partition(n=4):
    train, test = [], []
    for client in range(n):
        labels = np.array([client % 3, (client + 1) % 3] * 3, dtype=np.int64)
        x = np.random.default_rng(client).random((len(labels), 784), dtype=np.float32)
        train.append(ArrayDataset(x, labels))
        test.append(ArrayDataset(x[:2], labels[:2]))
    return train, test


def test_official_model_split():
    model = CollinsMLP()
    linears = [module for module in model.modules()
               if isinstance(module, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [
        (784, 512), (512, 256), (256, 64), (64, 10)
    ]


def test_clean_protocol_rejects_robust_aggregator():
    train, test = tiny_partition()
    cfg = CollinsConfig(population_clients=4, honest_per_round=2,
                        aggregator="NNM+TrMean", byzantine_per_round=0)
    with pytest.raises(ValueError, match="must use Average"):
        CollinsFEMNISTTrainer(cfg, train, test)


def test_short_clean_run_has_final_round_records():
    train, test = tiny_partition()
    cfg = CollinsConfig(algorithm="fedrep", rounds=2,
                        population_clients=4, honest_per_round=2,
                        head_epochs=1, representation_epochs=1,
                        batch_size=2, eval_every=1, official_softmax_ce=False)
    result = CollinsFEMNISTTrainer(cfg, train, test).run()
    assert [record["round"] for record in result["history"]] == [1, 2]
    assert result["reporting_rule"].startswith("unweighted client mean")
