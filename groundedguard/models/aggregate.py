"""Window aggregation, shared by training (Module 4) and inference (Module 6).

The model has ONE output per token (num_labels=1): a hallucination logit. For each
response token the logits of all windows of the response are combined into one score
("supported if any window supports it"), and the loss is computed on that:

  min      hard MIN over windows. Gradients flow only to the window that best supports
           each token, so windows that do not show a token's evidence are not forced
           to call it supported.
  softmin  -tau * logsumexp(-x / tau) over windows. Smooth version of min (-> min as
           tau -> 0); every window gets some gradient. Always <= min, by at most
           tau * ln(n_windows).
"""

import torch
import torch.nn.functional as F

from groundedguard.models.dataset import IGNORE

AGGREGATIONS = ("min", "softmin")


def window_grid(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    """logits [W, L] (one score per window token) -> [R, max windows, T] response-token
    logits per window; +inf where a response has no such window or t >= response_len."""
    group, slot = batch["group"], batch["slot"]
    resp_len = batch["response_len"]
    R, T = resp_len.size(0), int(resp_len.max())
    W, L = logits.shape

    t = torch.arange(T, device=logits.device)
    pos = (batch["response_start"][:, None] + t).clamp(max=L - 1)            # [W, T]
    valid = t[None, :] < resp_len[group][:, None]                             # [W, T]
    per_window = logits.gather(1, pos).masked_fill(~valid, float("inf"))      # [W, T]

    grid = logits.new_full((R, int(slot.max()) + 1, T), float("inf"))         # [R, max windows, T]
    grid[group, slot] = per_window
    return grid


def aggregate_windows(logits: torch.Tensor, batch: dict, aggregation: str = "min", tau: float = 1.0) -> torch.Tensor:
    """[W, L] -> [R, T] response-token logits. Padded positions (t >= response_len) are
    +inf; mask them with labels != IGNORE."""
    grid = window_grid(logits, batch)
    if aggregation == "min":
        return grid.amin(dim=1)
    if aggregation == "softmin":
        # Padded positions have no finite window; give logsumexp zeros there instead of
        # all -inf (whose backward is NaN), then put +inf back.
        empty = torch.isinf(grid).all(dim=1)                                  # [R, T]
        z = (-grid / tau).masked_fill(empty[:, None, :], 0.0)
        return (-tau * torch.logsumexp(z, dim=1)).masked_fill(empty, float("inf"))
    raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")


def min_over_windows(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    return aggregate_windows(logits, batch, "min")


def grouped_loss(logits: torch.Tensor, batch: dict, pos_weight: float | None = None,
                 aggregation: str = "min", tau: float = 1.0) -> torch.Tensor:
    """BCE on the window-aggregated logits, averaged over real response tokens."""
    agg = aggregate_windows(logits, batch, aggregation, tau)
    labels = batch["labels"]
    mask = labels != IGNORE
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=agg.device)
    return F.binary_cross_entropy_with_logits(agg[mask], labels[mask].float(), pos_weight=pw)
