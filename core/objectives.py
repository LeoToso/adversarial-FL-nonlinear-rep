"""Task-aware objectives and metrics used by both FL algorithms."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskObjective(nn.Module):
    """Cross entropy, multiclass least squares, or scalar least squares.

    The multiclass least-squares objective is exactly the paper's surrogate:
        mean ||e_y - score(x)||_2^2.
    No softmax is applied before the squared loss.
    """

    def __init__(self, task: str, loss_type: str, n_outputs: int):
        super().__init__()
        self.task = task
        self.loss_type = loss_type
        self.n_outputs = n_outputs
        valid = {"classification": {"cross_entropy", "multiclass_ls"},
                 "regression": {"least_squares"}}
        if task not in valid or loss_type not in valid[task]:
            raise ValueError(f"Invalid task/loss pair: {task}/{loss_type}")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.task == "regression":
            return F.mse_loss(prediction.reshape(-1), target.float().reshape(-1))
        if self.loss_type == "cross_entropy":
            return F.cross_entropy(prediction, target.long())
        one_hot = F.one_hot(target.long(), num_classes=self.n_outputs).to(prediction.dtype)
        return ((prediction - one_hot) ** 2).sum(dim=1).mean()


def batch_statistics(task: str, prediction: torch.Tensor, target: torch.Tensor):
    """Return (numerator, count) for accuracy or squared prediction error."""
    if task == "classification":
        correct = (prediction.argmax(dim=1) == target.long()).sum().item()
        return float(correct), int(target.numel())
    squared_error = ((prediction.reshape(-1) - target.float().reshape(-1)) ** 2).sum().item()
    return float(squared_error), int(target.numel())


@torch.no_grad()
def evaluate_model(model, loader, objective, device):
    model.eval()
    loss_sum = metric_sum = 0.0
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        prediction = model(x)
        batch_n = int(y.numel())
        loss_sum += float(objective(prediction, y).item()) * batch_n
        value, n = batch_statistics(objective.task, prediction, y)
        metric_sum += value
        count += n
    if count == 0:
        raise ValueError("Cannot evaluate on an empty loader.")
    out = {"test_loss": loss_sum / count}
    if objective.task == "classification":
        out.update(test_acc=metric_sum / count, metric_name="accuracy",
                   metric_value=metric_sum / count)
    else:
        out.update(test_mse=metric_sum / count, metric_name="mse",
                   metric_value=metric_sum / count)
    return out
