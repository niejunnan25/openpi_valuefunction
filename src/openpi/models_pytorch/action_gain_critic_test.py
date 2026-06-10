import pytest
import torch

from openpi.models_pytorch.action_gain_critic import ActionGainCritic


def test_action_gain_critic_accepts_full_action_horizon():
    critic = ActionGainCritic(input_dim=1024, hidden_dim=256, num_gain_bins=101, horizon=5)
    logits = critic(torch.randn(2, 50, 1024))
    assert logits.shape == (2, 101)


def test_action_gain_critic_accepts_horizon_only_input():
    critic = ActionGainCritic(input_dim=1024, hidden_dim=256, num_gain_bins=101, horizon=5)
    logits = critic(torch.randn(2, 5, 1024))
    assert logits.shape == (2, 101)


def test_action_gain_critic_rejects_short_action_hidden():
    critic = ActionGainCritic(input_dim=1024, hidden_dim=256, num_gain_bins=101, horizon=5)
    with pytest.raises(ValueError, match="Need at least"):
        critic(torch.randn(2, 4, 1024))
