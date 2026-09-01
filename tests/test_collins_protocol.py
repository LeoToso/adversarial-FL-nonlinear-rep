import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.collins_femnist import ArrayDataset, CollinsConfig, CollinsFEMNISTTrainer, CollinsMLP


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
