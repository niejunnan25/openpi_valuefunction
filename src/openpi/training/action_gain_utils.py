"""Distributional gain-label utilities for action-conditioned critics."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def make_atoms(min_value: float, max_value: float, num_bins: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Create evenly spaced categorical atoms."""
    if num_bins <= 1:
        raise ValueError(f"num_bins must be > 1, got {num_bins}")
    if max_value <= min_value:
        raise ValueError(f"max_value ({max_value}) must be > min_value ({min_value})")
    return torch.linspace(float(min_value), float(max_value), int(num_bins), device=device, dtype=dtype)


def _validate_atoms(atoms: torch.Tensor) -> torch.Tensor:
    atoms = atoms.flatten()
    if atoms.ndim != 1 or atoms.numel() <= 1:
        raise ValueError("atoms must be a 1D tensor with at least two entries")
    if not torch.all(atoms[1:] > atoms[:-1]):
        raise ValueError("atoms must be strictly increasing")
    return atoms


def two_hot_scalar_to_bins(values: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """Project scalar values to adjacent categorical atoms with linear interpolation."""
    atoms = _validate_atoms(atoms).to(device=values.device, dtype=values.dtype)
    values = values.clamp(min=atoms[0], max=atoms[-1])

    upper = torch.searchsorted(atoms, values.contiguous(), right=True).clamp(1, atoms.numel() - 1)
    lower = upper - 1

    lower_atoms = atoms[lower]
    upper_atoms = atoms[upper]
    denom = (upper_atoms - lower_atoms).clamp_min(torch.finfo(values.dtype).eps)
    upper_weight = (values - lower_atoms) / denom
    lower_weight = 1.0 - upper_weight

    out = torch.zeros(*values.shape, atoms.numel(), device=values.device, dtype=values.dtype)
    out.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    out.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return out


def project_distribution_to_bins(values: torch.Tensor, probs: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """Project weighted scalar support points to a categorical distribution over atoms.

    Args:
        values: Tensor with shape [..., N].
        probs: Probability mass tensor broadcastable to values, shape [..., N].
        atoms: Target 1D atom tensor with shape [G].

    Returns:
        Tensor with shape [..., G].
    """
    values, probs = torch.broadcast_tensors(values, probs)
    atoms = _validate_atoms(atoms).to(device=values.device, dtype=values.dtype)
    values = values.clamp(min=atoms[0], max=atoms[-1])
    probs = probs.to(dtype=values.dtype)

    flat_values = values.reshape(-1, values.shape[-1])
    flat_probs = probs.reshape(-1, probs.shape[-1])

    upper = torch.searchsorted(atoms, flat_values.contiguous(), right=True).clamp(1, atoms.numel() - 1)
    lower = upper - 1

    lower_atoms = atoms[lower]
    upper_atoms = atoms[upper]
    denom = (upper_atoms - lower_atoms).clamp_min(torch.finfo(values.dtype).eps)
    upper_weight = (flat_values - lower_atoms) / denom
    lower_weight = 1.0 - upper_weight

    out = torch.zeros(flat_values.shape[0], atoms.numel(), device=values.device, dtype=values.dtype)
    out.scatter_add_(1, lower, flat_probs * lower_weight)
    out.scatter_add_(1, upper, flat_probs * upper_weight)
    return out.reshape(*values.shape[:-1], atoms.numel())


def distribution_difference(
    p_curr: torch.Tensor,
    p_next: torch.Tensor,
    value_atoms: torch.Tensor,
    gain_atoms: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute p(V_next - V_curr) and project it to gain atoms.

    Args:
        p_curr: Current value distribution, shape [B, M].
        p_next: Future value distribution, shape [B, M].
        value_atoms: Value atoms, shape [M].
        gain_atoms: Gain atoms, shape [G].

    Returns:
        Gain target probabilities, shape [B, G].
    """
    if p_curr.ndim != 2 or p_next.ndim != 2:
        raise ValueError(f"p_curr and p_next must have shape [B, M], got {p_curr.shape} and {p_next.shape}")
    if p_curr.shape != p_next.shape:
        raise ValueError(f"p_curr and p_next shapes must match, got {p_curr.shape} and {p_next.shape}")

    dtype = p_curr.dtype
    device = p_curr.device
    value_atoms = _validate_atoms(value_atoms).to(device=device, dtype=dtype)
    gain_atoms = _validate_atoms(gain_atoms).to(device=device, dtype=dtype)
    if p_curr.shape[-1] != value_atoms.numel():
        raise ValueError(f"Expected {value_atoms.numel()} value bins, got {p_curr.shape[-1]}")

    p_curr = p_curr / p_curr.sum(dim=-1, keepdim=True).clamp_min(eps)
    p_next = p_next / p_next.sum(dim=-1, keepdim=True).clamp_min(eps)

    delta = value_atoms[:, None] - value_atoms[None, :]
    pair_probs = p_next[:, :, None] * p_curr[:, None, :]
    delta_flat = delta.reshape(1, -1).expand(p_curr.shape[0], -1)
    prob_flat = pair_probs.reshape(p_curr.shape[0], -1)

    gain_target = project_distribution_to_bins(delta_flat, prob_flat, gain_atoms)
    return gain_target / gain_target.sum(dim=-1, keepdim=True).clamp_min(eps)


def soft_cross_entropy(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross entropy against soft categorical targets."""
    if logits.shape != target_probs.shape:
        raise ValueError(f"logits and target_probs shapes must match, got {logits.shape} and {target_probs.shape}")
    loss = -(target_probs.to(dtype=logits.dtype) * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def expected_value(probs: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
    """Return the expected scalar value of a categorical distribution."""
    atoms = atoms.to(device=probs.device, dtype=probs.dtype)
    return (probs * atoms).sum(dim=-1)


def aggregate_gain_probs(probs: torch.Tensor, gain_atoms: torch.Tensor, eta: float) -> dict[str, torch.Tensor]:
    """Aggregate a gain distribution into up/flat/down probabilities."""
    gain_atoms = gain_atoms.to(device=probs.device, dtype=probs.dtype)
    up_mask = gain_atoms > eta
    flat_mask = gain_atoms.abs() <= eta
    down_mask = gain_atoms < -eta
    return {
        "p_up": probs[..., up_mask].sum(dim=-1),
        "p_flat": probs[..., flat_mask].sum(dim=-1),
        "p_down": probs[..., down_mask].sum(dim=-1),
    }
