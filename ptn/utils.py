import copy
import torch
import torch.nn as nn
import torch.optim as optim


class RollbackOnIncrease:
    """
    If loss increases (by more than eps), restore previous params and reduce LR.
    Works with any optimizer and multiple param groups.
    """

    def __init__(
        self,
        model,
        optimizer,
        factor=0.5,
        min_lr=1e-6,
        eps=0.0,
        on_cpu=True,
        verbose=True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.factor = factor
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose
        # cache "last good" state
        self.prev_loss = float("inf")
        # keep param snapshot; optionally move tensors to CPU to save GPU mem
        self._model_state = {
            k: v.detach().cpu() if on_cpu else v.detach().clone()
            for k, v in model.state_dict().items()
        }
        self._opt_state = copy.deepcopy(optimizer.state_dict())

    def _reduce_lr(self):
        for pg in self.optimizer.param_groups:
            old = pg["lr"]
            new = max(old * self.factor, self.min_lr)
            pg["lr"] = new
            if self.verbose:
                print(f"[RollbackOnIncrease] LR: {old:.3e} -> {new:.3e}")

    def _restore(self):
        # restore model + optimizer
        device = next(self.model.parameters()).device
        to_load = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self._model_state.items()
        }
        self.model.load_state_dict(to_load, strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(self._opt_state))

    def step(self, loss_value: float):
        """
        Call after your optimizer.step().
        If loss worsened: restore previous state and reduce LR.
        Otherwise: accept new state as the 'good' checkpoint.
        Returns: True if rolled back, else False.
        """
        if loss_value > self.prev_loss + self.eps:
            # loss increased -> rollback + cut LR
            if self.verbose:
                print(
                    f"[RollbackOnIncrease] Loss ↑ ({loss_value:.6f} > {self.prev_loss:.6f}). Rolling back."
                )
            self._restore()
            self._reduce_lr()
            return True
        else:
            # accept: update checkpoint
            self.prev_loss = loss_value
            self._model_state = {
                k: v.detach().cpu() if v.is_floating_point() else v
                for k, v in self.model.state_dict().items()
            }
            self._opt_state = copy.deepcopy(self.optimizer.state_dict())
            return False
