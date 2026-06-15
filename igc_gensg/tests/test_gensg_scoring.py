from __future__ import annotations

import numpy as np
import torch

from igc_gensg.scripts.stage1_gensg_sanity import (
    average_executed_action_attention,
    content_token_mask,
    gensg_score,
)


def test_content_token_mask_filters_command_and_function_words() -> None:
    token_ids = np.asarray([10, 11, 12, 13, 14, 15, 16])
    token_text = ["put", "▁the", "▁black", "▁bowl", "▁on", "▁plate", "<pad>"]
    raw = np.ones(len(token_text), dtype=bool)

    mask = content_token_mask(token_ids, token_text, raw)

    assert mask.tolist() == [False, False, True, True, False, True, False]


def test_gensg_score_uses_q_once_for_token_weights() -> None:
    action = np.asarray([[[1.0, 0.0], [0.0, 0.0]]])
    language = np.asarray([[0.5, 0.5]])
    token_maps = np.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )
    q = np.asarray([2.0, 1.0])
    mask = np.asarray([True, True])

    scores, contrib = gensg_score(action, language, token_maps, q, mask)

    # The first token gets normalized weight 2 / (2 + 1). If q were applied
    # again after normalization, this contribution would be 4 / 3 instead.
    assert np.isclose(contrib[0, 0], 2.0 / 3.0)
    assert np.isclose(contrib[0, 1], 0.0)
    assert np.isclose(scores[0], 2.0 / 3.0)


def test_gensg_score_preserves_image_and_language_mass() -> None:
    token_maps = np.asarray([[[1.0, 0.0], [0.0, 0.0]]])
    q = np.asarray([1.0])
    mask = np.asarray([True])
    actions = np.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.5, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ]
    )
    language = np.asarray([[1.0], [1.0], [0.25]])

    scores, _ = gensg_score(actions, language, token_maps, q, mask)

    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    assert np.isclose(scores[1], 0.5 * scores[0])
    assert np.isclose(scores[2], 0.25 * scores[0])


def test_average_executed_action_attention_uses_action_prefix() -> None:
    attn = torch.zeros((1, 1, 6, 3), dtype=torch.float32)
    # Last four query rows are action tokens with values 1, 2, 3, 4.
    attn[0, 0, 2, :] = 1.0
    attn[0, 0, 3, :] = 2.0
    attn[0, 0, 4, :] = 3.0
    attn[0, 0, 5, :] = 4.0

    first_two = average_executed_action_attention(attn, horizon=4, prefix_len=3, action_token_count=2)
    all_four = average_executed_action_attention(attn, horizon=4, prefix_len=3, action_token_count=None)

    assert torch.allclose(first_two, torch.full((1, 1, 3), 1.5))
    assert torch.allclose(all_four, torch.full((1, 1, 3), 2.5))
