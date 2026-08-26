import torch
import torch.nn as nn
from copy import deepcopy

class LocalOnly:
    """Each client trains independently — no federation."""

    def __init__(self, global_model, client_loaders, client_test_loaders,
                 lr, momentum, device):
        self.device = device
        self.client_loaders      = client_loaders
        self.client_test_loaders = client_test_loaders
        self.criterion = nn.CrossEntropyLoss()

        # each client gets its own independent copy of the model
        self.client_models = [deepcopy(global_model).to(device)
                              for _ in range(len(client_loaders))]
        self.optimizers = [
            torch.optim.SGD(m.parameters(), lr=lr, momentum=momentum)
            for m in self.client_models
        ]

    def train_round(self, client_loaders=None):
        model = self.client_models[0]
        opt   = self.optimizers[0]
        losses = []
        model.train()
        for x, y in self.client_loaders[0]:
            x, y = x.to(self.device), y.to(self.device)
            opt.zero_grad()
            loss = self.criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item())
        return {"loss": sum(losses) / len(losses) if losses else 0.0}
    
    def evaluate_global(self, test_loader):
        # for local training, global eval = average of local evals
        return self.evaluate(test_loader)

    def evaluate(self, test_loader=None, **kwargs):
        model = self.client_models[0]
        loader = self.client_test_loaders[0]
        model.eval()
        correct = total = 0
        total_loss = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = model(x)
                total_loss += self.criterion(logits, y).item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total   += y.size(0)
        return {"test_acc": correct/total, "test_loss": total_loss/total}