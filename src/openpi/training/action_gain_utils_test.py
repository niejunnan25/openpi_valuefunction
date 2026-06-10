import torch

from openpi.training.action_gain_utils import distribution_difference
from openpi.training.action_gain_utils import make_atoms


def test_distribution_difference_one_hot_delta_lands_on_expected_gain_atom():
    value_atoms = make_atoms(-1.0, 0.0, 51)
    gain_atoms = make_atoms(-1.0, 1.0, 101)
    p_curr = torch.zeros(1, 51)
    p_next = torch.zeros(1, 51)
    p_curr[0, 20] = 1.0  # -0.6
    p_next[0, 30] = 1.0  # -0.4

    target = distribution_difference(p_curr, p_next, value_atoms, gain_atoms)

    assert target.shape == (1, 101)
    assert torch.allclose(target.sum(dim=-1), torch.ones(1))
    assert target.argmax(dim=-1).item() == 60  # +0.2 on [-1, 1] with 0.02 spacing.
    assert torch.allclose(target[0, 60], torch.tensor(1.0))


def test_distribution_difference_same_one_hot_distribution_is_zero_gain():
    value_atoms = make_atoms(-1.0, 0.0, 51)
    gain_atoms = make_atoms(-1.0, 1.0, 101)
    p_curr = torch.zeros(1, 51)
    p_next = torch.zeros(1, 51)
    p_curr[0, 20] = 1.0
    p_next[0, 20] = 1.0

    target = distribution_difference(p_curr, p_next, value_atoms, gain_atoms)

    assert target.argmax(dim=-1).item() == 50
    assert torch.allclose(target[0, 50], torch.tensor(1.0))


def test_distribution_difference_preserves_probability_mass_for_dense_inputs():
    value_atoms = make_atoms(-1.0, 0.0, 51)
    gain_atoms = make_atoms(-1.0, 1.0, 101)
    p_curr = torch.softmax(torch.randn(3, 51), dim=-1)
    p_next = torch.softmax(torch.randn(3, 51), dim=-1)

    target = distribution_difference(p_curr, p_next, value_atoms, gain_atoms)

    assert target.shape == (3, 101)
    assert torch.allclose(target.sum(dim=-1), torch.ones(3), atol=1e-6)
