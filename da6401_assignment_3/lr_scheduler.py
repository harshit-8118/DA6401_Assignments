import math
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def _get_lr_scale(self) -> float:
        step = self.last_epoch + 1
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

    def get_lr(self) -> list[float]:
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


def get_lr_history(d_model: int, warmup_steps: int, total_steps: int) -> list[float]:
    dummy_model = __import__('torch').nn.Linear(1, 1)
    optimizer = optim.Adam(dummy_model.parameters(), lr=1.0)
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    history = []
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    return history
