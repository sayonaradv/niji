import numpy as np
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
    _LRScheduler,
)


class CosineWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epochs: int,
        warmup_epochs: int,
        warmup_start_factor: float = 1.0 / 3,
        eta_min: float = 0.0,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        linear_schedule = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            total_iters=warmup_epochs,
        )

        cosine_schedule = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=eta_min,
        )

        self.scheduler = SequentialLR(
            optimizer,
            schedulers=[linear_schedule, cosine_schedule],
            milestones=[warmup_epochs],
        )

        super().__init__(optimizer)

    def _initial_step(self):
        """Override to prevent warning during initialization."""

    def step(self):
        """Step the underlying scheduler."""
        self.scheduler.step()

    def state_dict(self):
        """Return state dict for checkpointing."""
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.scheduler.load_state_dict(state_dict)

    def get_last_lr(self):
        """Get the last computed learning rate."""
        return self.scheduler.get_last_lr()
