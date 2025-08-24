from mtp.mheads._utils import window_input_ids


def get_mt_head_loss(
    y, z, use_memory_efficient_loss: bool = True, window_shift: int = 1
):
    """Compute mhead loss.

    Args:
        y (torch.Tensor): Target tensor of shape (B, T). Note: this should be the unshifted target.
        z (torch.Tensor): Hidden state tensor. Should be of shape (B, T-H, D).
        use_memory_efficient_loss (bool, optional): Whether to use memory efficient loss. Defaults to True.
        window_shift (int, optional): The number of steps to shift the window. Defaults to 1. See example in `window_input_ids` docstring.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - loss (torch.Tensor): Loss tensor of shape (1,).
            - logits (torch.Tensor): Logits tensor of shape (B, T, V). Note: this is the logits from the first head only, we discard the rest.
    """
    if self.mt_head is None:
        raise ValueError("Multi-token prediction is not enabled for this model.")

    B, T = y.shape
    H_, D = min(self.config.mt_horizon, z.size(1)), z.size(-1)

    # Create targets
    # Shape: (B, T) -> (B, T, H)
    yw = window_input_ids(
        y,
        horizon=H_,
        shift=window_shift,
        ignore_index=-100,  # used to mask positions beyond seq length
    )

    # Sub-sample for memory efficiency
    if use_memory_efficient_loss and self.config.mt_horizon > 1:
        offset = torch.randint(0, H_, (1,)).item()
        z = z[:, offset::H_]
        yw = yw[:, offset::H_]

    # Merge batch and sequence dims
    z = z.reshape(-1, D)  # (BT, D)
    yw = yw.reshape(-1, H_)  # (BT, H)
    output = self.mt_head(z, yw, ignore_index=-100)
    loss = output.loss.mean()
    # logits = output.logits[:, 0].reshape(B, T, -1)  # (BT, H, V) -> (B, T, V)
    return loss, None
