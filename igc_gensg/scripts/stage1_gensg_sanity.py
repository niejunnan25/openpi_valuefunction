from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import math
import os
import pathlib
import random
import socket
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch


AGGREGATIONS = ("early1", "middle1", "last1", "last3_mean", "last5_mean", "all_steps_mean")
PURE_SCORE_NAMES = ("score_topq", "score_entropy", "score_imc")


def register_transformers_replace(root: pathlib.Path) -> None:
    candidates = [
        root / "src/openpi/models_pytorch/transformers_replace",
        pathlib.Path("/vla/users/niejunnan/codebase/openpi-modified/src/openpi/models_pytorch/transformers_replace"),
        pathlib.Path("/vla/users/niejunnan/codebase/openpi/src/openpi/models_pytorch/transformers_replace"),
    ]
    for replace_root in candidates:
        if not replace_root.exists():
            continue
        import transformers
        import transformers.models

        root_s = str(replace_root)
        models_s = str(replace_root / "models")
        if root_s not in transformers.__path__:
            transformers.__path__.insert(0, root_s)
        if models_s not in transformers.models.__path__:
            transformers.models.__path__.insert(0, models_s)
        for name in ("gemma", "paligemma", "siglip"):
            try:
                pkg = __import__(f"transformers.models.{name}", fromlist=["dummy"])
                pkg_path = str(replace_root / "models" / name)
                if hasattr(pkg, "__path__") and pkg_path not in pkg.__path__:
                    pkg.__path__.insert(0, pkg_path)
            except Exception:
                pass
        return


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_observations(path: pathlib.Path, indices: list[int] | None, count: int) -> list[dict[str, Any]]:
    z = np.load(path, allow_pickle=True)
    metadata = json.loads(str(z["metadata_json"].item()))
    if indices is None:
        selected = list(range(min(count, len(metadata))))
    else:
        selected = [idx for idx in indices if 0 <= idx < len(metadata)]
    observations: list[dict[str, Any]] = []
    for idx in selected:
        meta = dict(metadata[idx])
        meta["source_obs_index"] = idx
        observations.append(
            {
                "observation/image": np.asarray(z[f"obs{idx}_image"], dtype=np.uint8),
                "observation/wrist_image": np.asarray(z[f"obs{idx}_wrist_image"], dtype=np.uint8),
                "observation/state": np.asarray(z[f"obs{idx}_state"], dtype=np.float32),
                "prompt": str(meta["task_description"]),
                "__meta__": meta,
            }
        )
    return observations


def prepare_observation(policy, obs: dict[str, Any], device: str):
    import jax

    from openpi.models import model as _model

    clean = {k: v for k, v in obs.items() if not k.startswith("__")}
    transformed = policy._input_transform(jax.tree.map(lambda x: x, clean))
    batched = jax.tree.map(lambda x: np.asarray(x)[None, ...], transformed)
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(device), batched)
    return _model.Observation.from_dict(inputs), inputs


def repeat_tensor_batch(x: torch.Tensor, k: int) -> torch.Tensor:
    if x.shape[0] == k:
        return x
    if x.shape[0] != 1:
        raise ValueError(f"expected batch 1 or {k}, got {tuple(x.shape)}")
    return x.expand(k, *x.shape[1:]).contiguous()


def repeat_cache(cache: Any, k: int) -> Any:
    if cache is None or k == 1:
        return cache
    from transformers.cache_utils import DynamicCache

    legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else tuple(cache)
    repeated = tuple((key.repeat_interleave(k, dim=0), value.repeat_interleave(k, dim=0)) for key, value in legacy)
    return DynamicCache.from_legacy_cache(repeated)


def infer_prefix_spans(observation, prefix_len: int) -> dict[str, dict[str, Any]]:
    from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS

    image_keys = [key for key in IMAGE_KEYS if key in observation.images]
    image_keys.extend(key for key in observation.images if key not in image_keys)
    lang_len = int(observation.tokenized_prompt.shape[1])
    if not image_keys:
        raise ValueError("No image keys found in observation.")
    image_token_total = prefix_len - lang_len
    if image_token_total <= 0 or image_token_total % len(image_keys) != 0:
        raise ValueError(
            f"Cannot infer image spans: prefix_len={prefix_len}, lang_len={lang_len}, images={len(image_keys)}"
        )
    tokens_per_image = image_token_total // len(image_keys)
    grid_size = int(round(math.sqrt(tokens_per_image)))
    if grid_size * grid_size != tokens_per_image:
        raise RuntimeError(f"Image tokens per camera are not square: {tokens_per_image}")
    spans: dict[str, dict[str, Any]] = {}
    offset = 0
    for key in image_keys:
        spans[key] = {"start": offset, "end": offset + tokens_per_image, "tokens": tokens_per_image}
        offset += tokens_per_image
    spans["language"] = {"start": offset, "end": offset + lang_len, "tokens": lang_len}
    spans["__meta__"] = {
        "prefix_len": prefix_len,
        "lang_len": lang_len,
        "grid_size": grid_size,
        "tokens_per_image": tokens_per_image,
        "image_keys": image_keys,
    }
    return spans


def make_noise(seed: int, k: int, horizon: int, dim: int, device: str) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return torch.randn((k, horizon, dim), generator=gen, device=device, dtype=torch.float32)


def average_executed_action_attention(
    attn_weights: torch.Tensor,
    *,
    horizon: int,
    prefix_len: int,
    action_token_count: int | None,
) -> torch.Tensor:
    capture_action_tokens = horizon if action_token_count is None else max(1, min(int(action_token_count), int(horizon)))
    action_weights = attn_weights.detach()[:, :, -horizon:, :prefix_len].float()
    return action_weights[:, :, :capture_action_tokens, :].mean(dim=2)


def aggregate_steps(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "early1":
        return arr[0]
    if name == "middle1":
        return arr[arr.shape[0] // 2]
    if name == "last1":
        return arr[-1]
    if name == "last3_mean":
        return arr[-min(3, arr.shape[0]) :].mean(axis=0)
    if name == "last5_mean":
        return arr[-min(5, arr.shape[0]) :].mean(axis=0)
    if name == "all_steps_mean":
        return arr.mean(axis=0)
    raise KeyError(name)


def safe_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(x.sum(axis=-1, keepdims=True), eps)


def entropy_concentration(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    flat = p.reshape(*p.shape[:-2], -1) if p.ndim >= 2 else p
    flat = safe_normalize_rows(flat)
    out = np.zeros(flat.shape[:-1], dtype=np.float64)
    for idx in np.ndindex(out.shape):
        q = flat[idx]
        q = q[q > 0]
        entropy = 0.0 if q.size == 0 else float(-(q * np.log(q)).sum() / math.log(flat.shape[-1]))
        out[idx] = 1.0 - entropy
    return out


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), eps)
    q = q / np.maximum(q.sum(axis=-1, keepdims=True), eps)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        vals = np.where(a > eps, a * (np.log(a + eps) - np.log(b + eps)), 0.0)
        return vals.sum(axis=-1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def make_center_prior(grid_size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    center = (grid_size - 1) / 2.0
    sigma = max(grid_size / 4.0, 1.0)
    prior = np.exp(-((yy - center) ** 2 + (xx - center) ** 2) / (2.0 * sigma**2))
    return (prior / prior.sum()).astype(np.float64)


def compute_pure_scores(grids: np.ndarray) -> dict[str, np.ndarray]:
    flat = np.asarray(grids, dtype=np.float64).reshape(grids.shape[0], -1)
    image_mass = flat.sum(axis=-1)
    concentration = entropy_concentration(grids)
    topn = max(1, int(math.ceil(flat.shape[-1] * 0.05)))
    topq = np.sort(flat, axis=-1)[:, -topn:].mean(axis=-1)
    return {
        "score_topq": topq.astype(np.float64),
        "score_entropy": concentration.astype(np.float64),
        "score_imc": (image_mass * concentration).astype(np.float64),
        "image_mass": image_mass.astype(np.float64),
        "concentration": concentration.astype(np.float64),
    }


def selected_entropy(indices: list[int], k: int) -> float:
    if not indices or k <= 1:
        return 0.0
    counts = Counter(indices)
    probs = np.array([counts.get(i, 0) / len(indices) for i in range(k)], dtype=np.float64)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum() / math.log(k)) if probs.size else 0.0


def mean(xs: list[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def stdev(xs: list[float]) -> float:
    return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def decode_prompt_tokens(token_ids: np.ndarray) -> list[str]:
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer

        tok = PaligemmaTokenizer(max_len=int(len(token_ids)))
        sp = tok._tokenizer
        pieces = []
        for token in token_ids:
            token_int = int(token)
            if token_int <= 0:
                pieces.append("<pad>")
            else:
                try:
                    pieces.append(sp.id_to_piece(token_int))
                except Exception:
                    pieces.append(str(token_int))
        return pieces
    except Exception:
        return [str(int(x)) for x in token_ids]


_UNINFORMATIVE_PROMPT_TOKENS = {
    "",
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "both",
    "by",
    "do",
    "does",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "is",
    "it",
    "near",
    "of",
    "on",
    "onto",
    "or",
    "outside",
    "that",
    "the",
    "then",
    "there",
    "these",
    "this",
    "to",
    "with",
    "within",
    # Common LIBERO command verbs. These are task syntax, not visual entities.
    "bring",
    "close",
    "get",
    "grab",
    "grasp",
    "lift",
    "move",
    "open",
    "pick",
    "place",
    "pull",
    "push",
    "put",
    "remove",
    "slide",
    "take",
    "turn",
}


def _normalize_prompt_piece(text: str) -> str:
    text = str(text).replace("▁", " ").strip().lower()
    return "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-"}).strip("_-")


def content_token_mask(token_ids: np.ndarray, token_text: list[str], token_mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(token_mask, dtype=bool).copy()
    for idx, (token_id, text) in enumerate(zip(token_ids, token_text)):
        text_s = str(text)
        normalized = _normalize_prompt_piece(text_s)
        if (
            int(token_id) <= 0
            or text_s.startswith("<")
            or text_s in {"\n", "\\n"}
            or normalized in _UNINFORMATIVE_PROMPT_TOKENS
            or not any(ch.isalnum() for ch in normalized)
        ):
            valid[idx] = False
    return valid


@torch.no_grad()
def encode_prefix_context_with_grounding(
    model,
    observation,
    *,
    camera: str,
    prefix_layers: list[int] | None,
) -> dict[str, Any]:
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_len = int(prefix_embs.shape[1])
    spans = infer_prefix_spans(observation, prefix_len)
    if camera not in spans:
        raise KeyError(f"Camera {camera!r} not in prefix spans {list(spans)}")
    lang_span = spans["language"]
    image_span = spans[camera]
    grid_size = int(spans["__meta__"]["grid_size"])
    layers_all = model.paligemma_with_expert.paligemma.language_model.layers
    layer_indices = list(range(len(layers_all))) if prefix_layers is None else list(prefix_layers)

    capture: dict[int, torch.Tensor] = {}
    originals: list[tuple[Any, Any]] = []
    for layer_idx in layer_indices:
        attn_module = layers_all[layer_idx].self_attn
        original_forward = attn_module.forward

        def make_wrapped(orig, layer_number: int):
            def wrapped_forward(*args, **kwargs):
                result = orig(*args, **kwargs)
                attn_output, attn_weights = result
                if attn_weights is not None:
                    weights = attn_weights.detach().float()
                    values = weights[
                        :,
                        :,
                        lang_span["start"] : lang_span["end"],
                        image_span["start"] : image_span["end"],
                    ]
                    capture[layer_number] = values.reshape(
                        values.shape[0],
                        values.shape[1],
                        values.shape[2],
                        grid_size,
                        grid_size,
                    ).cpu()
                return attn_output, attn_weights

            return wrapped_forward

        attn_module.forward = make_wrapped(original_forward, layer_idx)
        originals.append((attn_module, original_forward))

    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    try:
        prefix_out, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
    finally:
        for module, original_forward in originals:
            module.forward = original_forward

    missing = [idx for idx in layer_indices if idx not in capture]
    if missing:
        raise RuntimeError(f"No prefix grounding attention captured for layers {missing[:8]}")
    token_ids = observation.tokenized_prompt.detach().cpu().numpy()[0]
    token_mask = observation.tokenized_prompt_mask.detach().cpu().numpy()[0].astype(bool)
    grounding = torch.stack([capture[idx][0] for idx in layer_indices], dim=0).numpy().astype(np.float32)
    return {
        "state": state,
        "prefix_pad_masks": prefix_pad_masks,
        "past_key_values": past_key_values,
        "prefix_out": prefix_out[0],
        "prefix_spans": spans,
        "prefix_len": prefix_len,
        "image_keys": list(observation.images.keys()),
        "prefix_grounding": grounding,  # [PL, PH, J, G, G]
        "prefix_layers": layer_indices,
        "token_ids": token_ids,
        "token_mask": token_mask,
        "token_text": decode_prompt_tokens(token_ids),
    }


@torch.no_grad()
def sample_k_with_generation_image_language_attention(
    model,
    prefix_context: dict[str, Any],
    *,
    k: int,
    device: str,
    num_steps: int,
    seed: int,
    camera: str,
    action_layers: list[int] | None,
    action_token_count: int | None = None,
) -> dict[str, Any]:
    all_layers = model.paligemma_with_expert.gemma_expert.model.layers
    layer_indices = list(range(len(all_layers))) if action_layers is None else list(action_layers)
    horizon = int(model.config.action_horizon)
    capture_action_tokens = horizon if action_token_count is None else max(1, min(int(action_token_count), horizon))
    dim = int(model.config.action_dim)
    repeated_state = repeat_tensor_batch(prefix_context["state"], k)
    repeated_prefix_pad_masks = repeat_tensor_batch(prefix_context["prefix_pad_masks"], k)
    repeated_past_key_values = repeat_cache(prefix_context["past_key_values"], k)
    spans = prefix_context["prefix_spans"]
    image_span = spans[camera]
    lang_span = spans["language"]
    prefix_len = int(prefix_context["prefix_len"])
    grid_size = int(spans["__meta__"]["grid_size"])

    capture_state: dict[str, dict[int, torch.Tensor]] = {"image": {}, "language": {}}
    originals: list[tuple[Any, Any]] = []
    for layer_idx in layer_indices:
        attn_module = all_layers[layer_idx].self_attn
        original_forward = attn_module.forward

        def make_wrapped(orig, layer_number: int):
            def wrapped_forward(*args, **kwargs):
                result = orig(*args, **kwargs)
                attn_output, attn_weights = result
                if attn_weights is not None:
                    weights = average_executed_action_attention(
                        attn_weights,
                        horizon=horizon,
                        prefix_len=prefix_len,
                        action_token_count=capture_action_tokens,
                    )
                    image_values = weights[:, :, image_span["start"] : image_span["end"]]
                    language_values = weights[:, :, lang_span["start"] : lang_span["end"]]
                    capture_state["image"][layer_number] = image_values.reshape(
                        image_values.shape[0],
                        image_values.shape[1],
                        grid_size,
                        grid_size,
                    ).cpu()
                    capture_state["language"][layer_number] = language_values.cpu()
                return attn_output, attn_weights

            return wrapped_forward

        attn_module.forward = make_wrapped(original_forward, layer_idx)
        originals.append((attn_module, original_forward))

    noise = make_noise(seed, k, horizon, dim, device)
    x_t = noise
    dt_step = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    current_time = torch.tensor(1.0, dtype=torch.float32, device=device)
    per_step_image: list[torch.Tensor] = []
    per_step_language: list[torch.Tensor] = []
    times: list[float] = []
    t0 = time.monotonic()
    try:
        while current_time >= -dt_step / 2:
            capture_state["image"] = {}
            capture_state["language"] = {}
            expanded_time = current_time.expand(k)
            v_t = model.denoise_step(
                repeated_state,
                repeated_prefix_pad_masks,
                repeated_past_key_values,
                x_t,
                expanded_time,
            )
            missing_image = [idx for idx in layer_indices if idx not in capture_state["image"]]
            missing_lang = [idx for idx in layer_indices if idx not in capture_state["language"]]
            if missing_image or missing_lang:
                raise RuntimeError(
                    f"Missing generation attention: image={missing_image[:8]} language={missing_lang[:8]}"
                )
            per_step_image.append(torch.stack([capture_state["image"][idx] for idx in layer_indices], dim=0))
            per_step_language.append(torch.stack([capture_state["language"][idx] for idx in layer_indices], dim=0))
            times.append(float(current_time.detach().cpu().item()))
            x_t = x_t + dt_step * v_t
            current_time = current_time + dt_step
    finally:
        for module, original_forward in originals:
            module.forward = original_forward
    sync()
    runtime_ms = (time.monotonic() - t0) * 1000.0
    return {
        "actions": x_t,
        "noise": noise,
        "action_image": torch.stack(per_step_image, dim=0).numpy().astype(np.float32),  # [T,L,K,H,G,G]
        "action_language": torch.stack(per_step_language, dim=0).numpy().astype(np.float32),  # [T,L,K,H,J]
        "times": times,
        "layers": layer_indices,
        "runtime_ms": runtime_ms,
        "action_token_count": int(capture_action_tokens),
        "attention_camera": camera,
        "camera_span": {
            "start": int(image_span["start"]),
            "end": int(image_span["end"]),
            "tokens": int(image_span["tokens"]),
        },
        "language_span": {
            "start": int(lang_span["start"]),
            "end": int(lang_span["end"]),
            "tokens": int(lang_span["tokens"]),
        },
    }


def compute_grounding_quality(
    prefix_grounding: np.ndarray,
    token_mask: np.ndarray,
    *,
    prefix_layers: list[int],
    token_ids: np.ndarray | None,
    top_heads: int,
    seed: int,
) -> dict[str, Any]:
    arr = np.asarray(prefix_grounding, dtype=np.float64)
    pl, ph, j, g, _ = arr.shape
    flat = arr.reshape(pl, ph, j, -1)
    mass = flat.sum(axis=-1)
    maps = safe_normalize_rows(flat)
    conc = entropy_concentration(maps.reshape(pl, ph, j, g, g))
    uniform = np.full((g * g,), 1.0 / (g * g), dtype=np.float64)
    center = make_center_prior(g).reshape(-1)
    global_distinct = js_divergence(maps, uniform)
    center_distinct = js_divergence(maps, center)
    denom = max(int(token_mask.sum()), 1)
    mean_token_map = safe_normalize_rows(np.sum(maps * token_mask[None, None, :, None], axis=2) / denom)
    token_distinct = js_divergence(maps, mean_token_map[:, :, None, :])
    occurrence_penalty = np.ones((j,), dtype=np.float64)
    if token_ids is not None:
        ids = [int(x) for x, valid in zip(token_ids, token_mask) if bool(valid)]
        counts = Counter(ids)
        occurrence_penalty = np.asarray(
            [1.0 / math.sqrt(max(1, counts.get(int(token_id), 1))) for token_id in token_ids],
            dtype=np.float64,
        )
    q_all = (
        np.sqrt(np.maximum(mass, 0.0))
        * conc
        * np.sqrt(np.maximum(global_distinct, 0.0))
        * np.maximum(token_distinct, 0.0)
        * (0.5 + np.maximum(center_distinct, 0.0))
        * occurrence_penalty[None, None, :]
    )
    q_all = np.where(token_mask[None, None, :], q_all, 0.0)
    head_score = np.zeros((pl, ph), dtype=np.float64)
    for li in range(pl):
        for hi in range(ph):
            vals = q_all[li, hi][token_mask]
            if vals.size:
                n = max(1, int(math.ceil(vals.size * 0.20)))
                head_score[li, hi] = float(np.sort(vals)[-n:].mean() + np.std(vals))
    order = np.argsort(head_score.reshape(-1))[::-1]
    selected_flat = order[: max(1, min(top_heads, order.size))]
    selected = [(int(x // ph), int(x % ph)) for x in selected_flat]
    rng = np.random.default_rng(seed)
    random_li = int(rng.integers(0, pl))
    random_hi = int(rng.integers(0, ph))

    def combine(heads: list[tuple[int, int]], *, use_q_weights: bool = True) -> tuple[np.ndarray, np.ndarray]:
        head_maps = []
        head_q = []
        for li, hi in heads:
            head_maps.append(maps[li, hi])
            head_q.append(q_all[li, hi])
        hm = np.stack(head_maps, axis=0)
        hq = np.stack(head_q, axis=0)
        if use_q_weights:
            weights = hq / np.maximum(hq.sum(axis=0, keepdims=True), 1e-12)
        else:
            weights = np.full_like(hq, 1.0 / max(hq.shape[0], 1), dtype=np.float64)
        combined = (hm * weights[:, :, None]).sum(axis=0)
        q = hq.max(axis=0) if use_q_weights else np.where(token_mask, 1.0, 0.0)
        q = np.where(token_mask, q, 0.0)
        return combined.reshape(j, g, g), q

    main_maps, q = combine(selected)
    no_q_maps, no_q_values = combine([(li, hi) for li in range(pl) for hi in range(ph)], use_q_weights=False)
    random_maps, random_q = combine([(random_li, random_hi)])
    uniform_maps = np.broadcast_to(uniform.reshape(g, g), (j, g, g)).copy()
    center_maps = np.broadcast_to(center.reshape(g, g), (j, g, g)).copy()
    mean_q = float(np.mean(q[token_mask])) if token_mask.any() and np.any(q[token_mask] > 0) else 1.0
    return {
        "maps": main_maps.astype(np.float32),
        "q": q.astype(np.float64),
        "maps_no_q": no_q_maps.astype(np.float32),
        "q_no_q": no_q_values.astype(np.float64),
        "selected_prefix_heads": [
            {
                "local_layer_index": li,
                "layer": int(prefix_layers[li]),
                "head": hi,
                "score": float(head_score[li, hi]),
            }
            for li, hi in selected
        ],
        "random_maps": random_maps.astype(np.float32),
        "random_q": random_q.astype(np.float64),
        "random_prefix_head": {
            "local_layer_index": random_li,
            "layer": int(prefix_layers[random_li]),
            "head": random_hi,
            "score": float(head_score[random_li, random_hi]),
        },
        "shuffled_maps": main_maps[np.random.default_rng(seed + 13).permutation(j)].astype(np.float32),
        "random_token_maps": np.random.default_rng(seed + 29).random((j, g, g)).astype(np.float32),
        "center_maps": center_maps.astype(np.float32),
        "center_q": np.where(token_mask, np.full((j,), mean_q), 0.0),
        "uniform_maps": uniform_maps.astype(np.float32),
        "head_score": head_score,
        "mass": mass,
        "concentration": conc,
    }


def gensg_score(
    action_image_grids: np.ndarray,
    action_language: np.ndarray,
    token_maps: np.ndarray,
    q: np.ndarray,
    token_mask: np.ndarray,
    *,
    return_components: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    k = int(action_image_grids.shape[0])
    action_flat = np.asarray(action_image_grids, dtype=np.float64).reshape(k, -1)
    image_mass = action_flat.sum(axis=-1)
    a = safe_normalize_rows(action_flat)
    visual_concentration = entropy_concentration(np.asarray(action_image_grids, dtype=np.float64))
    g = safe_normalize_rows(np.asarray(token_maps, dtype=np.float64).reshape(token_maps.shape[0], -1))
    w = np.asarray(action_language, dtype=np.float64)
    qv = np.asarray(q, dtype=np.float64)
    valid = np.asarray(token_mask, dtype=bool)
    qv = np.where(valid, qv, 0.0)
    w = np.where(valid[None, :], w, 0.0)
    language_mass = w.sum(axis=-1)
    overlap = a @ g.T
    weighted = w * qv[None, :]
    weighted = weighted / np.maximum(weighted.sum(axis=-1, keepdims=True), 1e-12)
    # q_j is used once to choose reliable token maps. The final score keeps the
    # candidate-level evidence that was previously normalized away.
    contrib = weighted * overlap
    alignment = contrib.sum(axis=-1)
    scores = image_mass * visual_concentration * language_mass * alignment
    if return_components:
        components = {
            "image_mass": image_mass.astype(np.float64),
            "visual_concentration": np.asarray(visual_concentration, dtype=np.float64),
            "language_mass": language_mass.astype(np.float64),
            "alignment": alignment.astype(np.float64),
        }
        return scores, contrib, components
    return scores, contrib


def pairwise_action_l2(actions: torch.Tensor) -> tuple[float, float, float]:
    flat = actions.detach().float().cpu().numpy().reshape(actions.shape[0], -1)
    vals = []
    for i in range(flat.shape[0]):
        for j in range(i + 1, flat.shape[0]):
            vals.append(float(np.linalg.norm(flat[i] - flat[j])))
    if not vals:
        return 0.0, 0.0, 0.0
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def render_obs_figure(
    out_path: pathlib.Path,
    *,
    rgb: np.ndarray,
    token_text: list[str],
    token_maps: np.ndarray,
    q: np.ndarray,
    action_grid: np.ndarray,
    scores: np.ndarray,
    selected: int,
    top_token_indices: list[int],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_tokens = min(len(top_token_indices), 5)
    fig_cols = max(4, n_tokens + 2)
    fig, axes = plt.subplots(2, fig_cols, figsize=(3.0 * fig_cols, 6.0))
    for ax in axes.reshape(-1):
        ax.axis("off")
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("rgb")
    for idx, tok_idx in enumerate(top_token_indices[:n_tokens], start=1):
        axes[0, idx].imshow(token_maps[tok_idx], cmap="viridis")
        axes[0, idx].set_title(f"{tok_idx}:{token_text[tok_idx]}\nq={q[tok_idx]:.2g}")
    vmax = float(np.max(action_grid)) if np.max(action_grid) > 0 else 1.0
    for cand in range(min(action_grid.shape[0], fig_cols)):
        axes[1, cand].imshow(action_grid[cand], cmap="magma", vmin=0.0, vmax=vmax)
        axes[1, cand].set_title(f"k={cand} s={scores[cand]:.2g}" + (" *" if cand == selected else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_int_list(spec: str) -> list[int] | None:
    spec = str(spec).strip()
    if not spec or spec == "none" or spec == "all":
        return None
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GenSG-IGC stage1 offline sanity check.")
    parser.add_argument("--root", default="/vla/users/niejunnan/knows/openpi")
    parser.add_argument("--checkpoint", default="/vla/users/niejunnan/assets/openpi-assets/checkpoints/pi0_libero_pytorch")
    parser.add_argument("--config", default="pi0_libero")
    parser.add_argument("--observations", default="igc_select_22/data/stage1_22_full40_observations.npz")
    parser.add_argument("--out-dir", default="igc_gensg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--execution-action-tokens", type=int, default=5)
    parser.add_argument("--obs-count", type=int, default=16)
    parser.add_argument(
        "--obs-indices",
        default="0,1,4,8,10,11,16,19,20,21,22,25,30,31,32,35",
        help="Comma-separated observation indices from the NPZ. Use 'none' for first obs-count.",
    )
    parser.add_argument("--seed", type=int, default=3100)
    parser.add_argument("--camera", default="base_0_rgb")
    parser.add_argument("--prefix-layers", default="all")
    parser.add_argument("--action-layers", default="all")
    parser.add_argument("--prefix-top-heads", type=int, default=4)
    parser.add_argument("--max-figures", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = pathlib.Path(args.root)
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    client_src = root / "packages/openpi-client/src"
    if client_src.exists() and str(client_src) not in sys.path:
        sys.path.insert(0, str(client_src))
    register_transformers_replace(root)

    out_dir = root / args.out_dir
    logs_dir = out_dir / "logs"
    results_dir = out_dir / "results"
    figures_dir = out_dir / "figures"
    reports_dir = out_dir / "reports"
    for d in (logs_dir, results_dir, figures_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    with (logs_dir / "commands_stage1_gensg.txt").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": dt.datetime.now().isoformat(),
                    "hostname": socket.gethostname(),
                    "argv": sys.argv,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    from openpi.policies import policy_config
    from openpi.training import config as training_config

    obs_indices = parse_int_list(args.obs_indices)
    observations = load_observations(root / args.observations, obs_indices, args.obs_count)
    cfg = training_config.get_config(args.config)
    if hasattr(cfg.model, "pytorch_compile_mode"):
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))

    print(json.dumps({"event": "load_start", "timestamp": dt.datetime.now().isoformat()}), flush=True)
    load_t0 = time.monotonic()
    policy = policy_config.create_trained_policy(
        cfg,
        pathlib.Path(args.checkpoint),
        sample_kwargs={"num_steps": args.num_steps},
        pytorch_device=args.device,
    )
    sync()
    print(json.dumps({"event": "load_done", "load_s": time.monotonic() - load_t0}), flush=True)
    if not getattr(policy, "_is_pytorch_model", False):
        raise RuntimeError("GenSG sanity check supports PyTorch pi0 only.")
    model = policy._model
    if getattr(model.config, "pi05", False):
        raise RuntimeError("GenSG sanity check must use pi0, not pi05.")

    prefix_all_layers = model.paligemma_with_expert.paligemma.language_model.layers
    action_all_layers = model.paligemma_with_expert.gemma_expert.model.layers
    prefix_layers = list(range(len(prefix_all_layers))) if args.prefix_layers == "all" else parse_int_list(args.prefix_layers)
    action_layers = list(range(len(action_all_layers))) if args.action_layers == "all" else parse_int_list(args.action_layers)
    if prefix_layers is None:
        prefix_layers = list(range(len(prefix_all_layers)))
    if action_layers is None:
        action_layers = list(range(len(action_all_layers)))

    obs_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    metric_store: dict[tuple[str, int, int, str], dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    control_store: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    rng = random.Random(args.seed)

    for obs_i, obs in enumerate(observations):
        meta = dict(obs.get("__meta__", {}))
        print(json.dumps({"event": "obs_start", "obs_index": obs_i, "meta": meta}, ensure_ascii=False), flush=True)
        observation, _inputs = prepare_observation(policy, obs, args.device)
        sync()
        prefix_t0 = time.monotonic()
        prefix_context = encode_prefix_context_with_grounding(
            model,
            observation,
            camera=args.camera,
            prefix_layers=prefix_layers,
        )
        sync()
        prefix_ms = (time.monotonic() - prefix_t0) * 1000.0
        run = sample_k_with_generation_image_language_attention(
            model,
            prefix_context,
            k=args.k,
            device=args.device,
            num_steps=args.num_steps,
            seed=args.seed + obs_i * 1009,
            camera=args.camera,
            action_layers=action_layers,
            action_token_count=args.execution_action_tokens,
        )
        action_pair_mean, action_pair_min, action_pair_max = pairwise_action_l2(run["actions"])
        raw_token_mask = np.asarray(prefix_context["token_mask"], dtype=bool)
        token_text = list(prefix_context["token_text"])
        token_ids = np.asarray(prefix_context["token_ids"])
        token_mask = content_token_mask(token_ids, token_text, raw_token_mask)
        grounding = compute_grounding_quality(
            prefix_context["prefix_grounding"],
            token_mask,
            prefix_layers=prefix_layers,
            token_ids=token_ids,
            top_heads=args.prefix_top_heads,
            seed=args.seed + obs_i * 17,
        )
        q = np.asarray(grounding["q"], dtype=np.float64)
        top_tokens = np.argsort(q)[::-1]
        top_tokens = [int(x) for x in top_tokens if token_mask[x]][:10]
        q_nonzero = bool(np.max(q[token_mask]) > 1e-12) if token_mask.any() else False
        token_q_spread = float(np.std(q[token_mask])) if token_mask.any() else 0.0
        token_map_flat = grounding["maps"].reshape(grounding["maps"].shape[0], -1)
        valid_maps = safe_normalize_rows(token_map_flat[token_mask])
        token_map_pairwise = 0.0
        if valid_maps.shape[0] > 1:
            vals = []
            for a in range(valid_maps.shape[0]):
                for b in range(a + 1, valid_maps.shape[0]):
                    vals.append(float(np.linalg.norm(valid_maps[a] - valid_maps[b])))
            token_map_pairwise = float(np.mean(vals)) if vals else 0.0

        for rank, tok_idx in enumerate(top_tokens, start=1):
            token_rows.append(
                {
                    "obs_index": obs_i,
                    "source_obs_index": meta.get("source_obs_index"),
                    "suite": meta.get("suite"),
                    "task_id": meta.get("task_id"),
                    "task_description": meta.get("task_description"),
                    "rank": rank,
                    "token_index": tok_idx,
                    "token_text": token_text[tok_idx],
                    "q": float(q[tok_idx]),
                    "token_valid": bool(token_mask[tok_idx]),
                    "selected_prefix_heads": json.dumps(grounding["selected_prefix_heads"]),
                    "random_prefix_head": json.dumps(grounding["random_prefix_head"]),
                }
            )

        action_image = run["action_image"]
        action_language = run["action_language"]
        random_action_layer = rng.choice(run["layers"])
        random_action_head = rng.randrange(action_image.shape[3])
        figure_written = False

        for aggregation in AGGREGATIONS:
            agg_img = aggregate_steps(action_image, aggregation)
            agg_lang = aggregate_steps(action_language, aggregation)
            for li, layer in enumerate(run["layers"]):
                for head in range(agg_img.shape[2]):
                    grids = np.asarray(agg_img[li, :, head], dtype=np.float32)
                    langs = np.asarray(agg_lang[li, :, head], dtype=np.float32)
                    pure = compute_pure_scores(grids)
                    gensg, contrib = gensg_score(grids, langs, grounding["maps"], q, token_mask)
                    no_q, _ = gensg_score(
                        grids,
                        langs,
                        grounding["maps_no_q"],
                        grounding["q_no_q"],
                        token_mask,
                    )
                    no_w, _ = gensg_score(grids, np.ones_like(langs), grounding["maps"], q, token_mask)
                    random_token, _ = gensg_score(grids, langs, grounding["random_token_maps"], q, token_mask)
                    shuffled, _ = gensg_score(grids, langs, grounding["shuffled_maps"], q, token_mask)
                    random_prefix, _ = gensg_score(grids, langs, grounding["random_maps"], grounding["random_q"], token_mask)
                    center_prior, _ = gensg_score(grids, langs, grounding["center_maps"], grounding["center_q"], token_mask)
                    score_dict = {
                        "gensg": gensg,
                        "no_q": no_q,
                        "no_W": no_w,
                        "random_token": random_token,
                        "shuffled_token": shuffled,
                        "random_prefix_head": random_prefix,
                        "center_prior": center_prior,
                    }
                    score_dict.update({name: pure[name] for name in PURE_SCORE_NAMES})
                    for score_name, score_values in score_dict.items():
                        key = (aggregation, int(layer), int(head), score_name)
                        metric_store[key]["score_variance"].append(float(np.var(score_values)))
                        metric_store[key]["selected_index"].append(int(np.argmax(score_values)))
                        metric_store[key]["selected_score"].append(float(np.max(score_values)))
                        metric_store[key]["score_std"].append(float(np.std(score_values)))
                    if int(layer) == random_action_layer and int(head) == random_action_head:
                        key = (aggregation, "random_action_head_gensg")
                        control_store[key]["score_variance"].append(float(np.var(gensg)))
                        control_store[key]["selected_index"].append(int(np.argmax(gensg)))
                    if (
                        not figure_written
                        and obs_i < args.max_figures
                        and aggregation == "last3_mean"
                        and int(layer) == 17
                        and int(head) == 0
                    ):
                        selected = int(np.argmax(gensg))
                        contrib_selected = contrib[selected]
                        top_contrib = [
                            int(x)
                            for x in np.argsort(contrib_selected)[::-1]
                            if token_mask[x] and contrib_selected[x] > 0
                        ][:5]
                        if not top_contrib:
                            top_contrib = top_tokens[:5]
                        render_obs_figure(
                            figures_dir / f"stage1_gensg_obs{obs_i:02d}_l17h0_last3.png",
                            rgb=np.asarray(obs["observation/image"]),
                            token_text=token_text,
                            token_maps=grounding["maps"],
                            q=q,
                            action_grid=grids,
                            scores=gensg,
                            selected=selected,
                            top_token_indices=top_contrib,
                        )
                        figure_written = True

        obs_row = {
            "obs_index": obs_i,
            "source_obs_index": meta.get("source_obs_index"),
            "suite": meta.get("suite"),
            "task_id": meta.get("task_id"),
            "task_description": meta.get("task_description"),
            "prefix_ms": prefix_ms,
            "generation_runtime_ms": run["runtime_ms"],
            "action_pairwise_l2_mean": action_pair_mean,
            "action_pairwise_l2_min": action_pair_min,
            "action_pairwise_l2_max": action_pair_max,
            "prefix_grounding_shape": list(prefix_context["prefix_grounding"].shape),
            "action_image_shape": list(action_image.shape),
            "action_language_shape": list(action_language.shape),
            "times": run["times"],
            "camera": args.camera,
            "action_token_count": int(run.get("action_token_count", args.execution_action_tokens)),
            "camera_span": run["camera_span"],
            "language_span": run["language_span"],
            "token_count_valid": int(token_mask.sum()),
            "token_q_nonzero": q_nonzero,
            "token_q_spread": token_q_spread,
            "token_map_pairwise_l2": token_map_pairwise,
            "top_tokens": json.dumps(
                [{"idx": idx, "text": token_text[idx], "q": float(q[idx])} for idx in top_tokens],
                ensure_ascii=False,
            ),
            "selected_prefix_heads": json.dumps(grounding["selected_prefix_heads"]),
            "random_prefix_head": json.dumps(grounding["random_prefix_head"]),
        }
        obs_rows.append(obs_row)
        with (logs_dir / "stage1_gensg_obs_summaries.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(obs_row, ensure_ascii=False, sort_keys=True) + "\n")
        np.savez_compressed(
            results_dir / f"stage1_gensg_obs{obs_i:02d}.npz",
            action_image=action_image,
            action_language=action_language,
            prefix_grounding=prefix_context["prefix_grounding"],
            gensg_maps=grounding["maps"],
            gensg_maps_no_q=grounding["maps_no_q"],
            q=q,
            q_no_q=grounding["q_no_q"],
            token_mask=token_mask,
            token_ids=np.asarray(prefix_context["token_ids"]),
            raw_token_mask=raw_token_mask,
            token_text=np.asarray(token_text, dtype=object),
            times=np.asarray(run["times"], dtype=np.float32),
            action_token_count=np.asarray([int(run.get("action_token_count", args.execution_action_tokens))], dtype=np.int32),
            layers=np.asarray(run["layers"], dtype=np.int32),
            prefix_layers=np.asarray(prefix_layers, dtype=np.int32),
            meta_json=json.dumps(meta, ensure_ascii=False),
        )
        print(
            json.dumps(
                {
                    "event": "obs_done",
                    "obs_index": obs_i,
                    "prefix_grounding_shape": obs_row["prefix_grounding_shape"],
                    "action_image_shape": obs_row["action_image_shape"],
                    "action_language_shape": obs_row["action_language_shape"],
                    "top_tokens": json.loads(obs_row["top_tokens"]),
                    "generation_runtime_ms": run["runtime_ms"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    metric_rows: list[dict[str, Any]] = []
    for (aggregation, layer, head, score_name), vals in metric_store.items():
        selected = [int(x) for x in vals["selected_index"]]
        metric_rows.append(
            {
                "aggregation": aggregation,
                "layer": layer,
                "head": head,
                "score_name": score_name,
                "n_observations": len(selected),
                "score_variance_mean": mean(vals["score_variance"]),
                "score_variance_std": stdev(vals["score_variance"]),
                "score_std_mean": mean(vals["score_std"]),
                "selected_score_mean": mean(vals["selected_score"]),
                "selected_index_entropy": selected_entropy(selected, args.k),
                "selected_index_counts": json.dumps(dict(Counter(selected)), sort_keys=True),
            }
        )
    metric_rows.sort(
        key=lambda r: (
            1 if r["score_name"] == "gensg" else 0,
            float(r["score_variance_mean"]),
            float(r["selected_index_entropy"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(metric_rows, start=1):
        row["rank"] = rank
    metric_rows = [{"rank": row.pop("rank"), **row} for row in metric_rows]

    control_rows: list[dict[str, Any]] = []
    for (aggregation, control_name), vals in control_store.items():
        selected = [int(x) for x in vals["selected_index"]]
        control_rows.append(
            {
                "aggregation": aggregation,
                "control_name": control_name,
                "n_observations": len(selected),
                "score_variance_mean": mean(vals["score_variance"]),
                "selected_index_entropy": selected_entropy(selected, args.k),
                "selected_index_counts": json.dumps(dict(Counter(selected)), sort_keys=True),
            }
        )

    write_csv(results_dir / "stage1_gensg_head_time_metrics.csv", metric_rows)
    write_csv(results_dir / "stage1_gensg_token_quality.csv", token_rows)
    write_csv(results_dir / "stage1_gensg_controls.csv", control_rows)
    with (results_dir / "stage1_gensg_obs_summaries.json").open("w", encoding="utf-8") as f:
        json.dump(obs_rows, f, ensure_ascii=False, indent=2)

    shape_ok = all(row["action_image_shape"][2] == args.k and row["action_language_shape"][2] == args.k for row in obs_rows)
    prefix_ok = all(row["prefix_grounding_shape"][2] >= row["token_count_valid"] for row in obs_rows)
    q_ok = all(bool(row["token_q_nonzero"]) and float(row["token_q_spread"]) > 1e-12 for row in obs_rows)
    token_maps_ok = all(float(row["token_map_pairwise_l2"]) > 1e-6 for row in obs_rows)
    gensg_rows = [r for r in metric_rows if r["score_name"] == "gensg"]
    gensg_var_ok = any(float(r["score_variance_mean"]) > 1e-12 for r in gensg_rows)
    selected_ok = any(float(r["selected_index_entropy"]) > 0.0 for r in gensg_rows)
    function_or_relation_like = {"▁the", "▁to", "▁and", "▁of", "▁it", "▁up", "▁in", "▁on", "▁both"}
    token_by_obs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        token_by_obs[int(row["obs_index"])].append(row)
    top1_nonentity = 0
    top3_nonentity = 0
    top3_total = 0
    for rows_for_obs in token_by_obs.values():
        rows_for_obs = sorted(rows_for_obs, key=lambda r: int(r["rank"]))
        if rows_for_obs and str(rows_for_obs[0]["token_text"]) in function_or_relation_like:
            top1_nonentity += 1
        for row in rows_for_obs[:3]:
            top3_total += 1
            if str(row["token_text"]) in function_or_relation_like:
                top3_nonentity += 1
    top1_nonentity_rate = top1_nonentity / max(len(token_by_obs), 1)
    top3_nonentity_rate = top3_nonentity / max(top3_total, 1)
    token_purity_ok = top1_nonentity_rate <= 0.25 and top3_nonentity_rate <= 0.25
    def find_metric(score_name: str, aggregation: str, layer: Any, head: Any) -> dict[str, Any] | None:
        for row in metric_rows:
            if (
                row["score_name"] == score_name
                and row["aggregation"] == aggregation
                and int(row["layer"]) == int(layer)
                and int(row["head"]) == int(head)
            ):
                return row
        return None

    control_score_names = ("random_token", "shuffled_token", "random_prefix_head", "center_prior")
    def control_max_for(row: dict[str, Any] | None) -> float:
        if row is None:
            return float("inf")
        vals = []
        for name in control_score_names:
            ctrl = find_metric(name, row["aggregation"], row["layer"], row["head"])
            if ctrl is not None:
                vals.append(float(ctrl["score_variance_mean"]))
        return max(vals) if vals else float("inf")

    top_gensg = gensg_rows[:10]
    last_rows = [r for r in gensg_rows if r["aggregation"] in {"last1", "last3_mean", "last5_mean"}]
    phase2_rows = [r for r in gensg_rows if r["aggregation"] in {"last1", "last3_mean"}]
    phase2_clean_rows = [r for r in phase2_rows if float(r["score_variance_mean"]) >= control_max_for(r)]
    phase2_recommendation = (
        phase2_clean_rows[0]
        if phase2_clean_rows
        else (phase2_rows[0] if phase2_rows else (gensg_rows[0] if gensg_rows else None))
    )
    recommended_control_max = control_max_for(phase2_recommendation)
    recommended_gensg_var = (
        float(phase2_recommendation["score_variance_mean"]) if phase2_recommendation is not None else float("nan")
    )
    controls_not_better = bool(
        phase2_recommendation is not None
        and math.isfinite(recommended_control_max)
        and recommended_gensg_var >= recommended_control_max
    )
    early_rows = [r for r in gensg_rows if r["aggregation"] in {"early1", "middle1"}]
    all_rows = [r for r in gensg_rows if r["aggregation"] == "all_steps_mean"]
    late_better_than_all = mean([float(r["score_variance_mean"]) for r in last_rows[:20]]) >= mean(
        [float(r["score_variance_mean"]) for r in all_rows[:20]]
    )
    late_better_than_early = mean([float(r["score_variance_mean"]) for r in last_rows[:20]]) >= mean(
        [float(r["score_variance_mean"]) for r in early_rows[:20]]
    )
    if (
        shape_ok
        and prefix_ok
        and q_ok
        and token_maps_ok
        and gensg_var_ok
        and selected_ok
        and token_purity_ok
        and controls_not_better
    ):
        conclusion = "通过" if late_better_than_all else "部分通过"
    elif shape_ok and prefix_ok and gensg_var_ok and not controls_not_better:
        conclusion = "证据不足"
    elif shape_ok and prefix_ok and gensg_var_ok:
        conclusion = "部分通过"
    else:
        conclusion = "不通过"

    def top_by(score_name: str, n: int = 5) -> list[dict[str, Any]]:
        return [r for r in metric_rows if r["score_name"] == score_name][:n]

    report_lines = [
        "# GenSG-IGC Stage1 Offline Sanity Check",
        "",
        f"- timestamp: {dt.datetime.now().isoformat()}",
        f"- host: {socket.gethostname()}",
        f"- CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}",
        f"- config: {args.config}",
        f"- checkpoint: {args.checkpoint}",
        f"- observations: {len(observations)}",
        f"- K: {args.k}",
        f"- num_steps: {args.num_steps}",
        f"- camera: {args.camera}",
        f"- prefix layers: {len(prefix_layers)}",
        f"- action layers: {len(action_layers)}",
        f"- conclusion: **{conclusion}**",
        "",
        "## What Was Tested",
        "",
        "This offline stage does not run online rollout success. It checks whether generation-time action attention can be combined with automatic token-level self-grounding from the frozen pi0 model. The score uses no simulator mask, object id, detector, segmentation model, tracker, prompt rewrite, or hand-written entity span.",
        "",
        "## Sanity Gates",
        "",
        f"- generation-time action-to-image and action-to-language preserve batch=K: {shape_ok}",
        f"- prefix token-to-image grounding captured for language tokens: {prefix_ok}",
        f"- token quality q is non-zero and separates tokens: {q_ok}",
        f"- token maps differ across tokens: {token_maps_ok}",
        f"- top token diagnostic is clean enough: {token_purity_ok} (top1 function/relation-like rate={top1_nonentity_rate:.3f}, top3 rate={top3_nonentity_rate:.3f})",
        f"- GenSG score has non-zero candidate variance: {gensg_var_ok}",
        f"- GenSG selected index is not fixed for at least one head/time: {selected_ok}",
        f"- GenSG is not weaker than random/shuffled/center controls at the recommended head/time: {controls_not_better}",
        f"- late-step aggregation is at least as discriminative as early/middle steps: {late_better_than_early}",
        f"- late-step aggregation is at least as discriminative as all-steps mean: {late_better_than_all}",
        "",
        "## Top GenSG Head/Time Rows",
        "",
    ]
    for row in top_gensg:
        report_lines.append(
            f"- rank {row['rank']}: {row['aggregation']} layer {row['layer']} head {row['head']}, "
            f"var={float(row['score_variance_mean']):.4g}, selected_entropy={float(row['selected_index_entropy']):.4g}, "
            f"counts={row['selected_index_counts']}"
        )
    report_lines += ["", "## Recommended Stage2 Configuration", ""]
    if phase2_recommendation is None:
        report_lines.append("- No valid GenSG row found; do not enter Stage2.")
    else:
        report_lines.extend(
            [
                f"- action head: layer {phase2_recommendation['layer']} head {phase2_recommendation['head']}",
                f"- time aggregation: {phase2_recommendation['aggregation']}",
                f"- prefix_top_heads: {args.prefix_top_heads}",
                f"- camera: {args.camera}",
                f"- execution_action_tokens: {args.execution_action_tokens}",
                f"- K: {args.k}",
                f"- recommendation source: top GenSG row constrained to last1/last3_mean, rank {phase2_recommendation['rank']}",
            ]
        )
        report_lines.append(
            f"- same-head control max variance: {recommended_control_max:.4g}; GenSG variance: {recommended_gensg_var:.4g}"
        )
    report_lines += ["", "## Time Aggregation Comparison", ""]
    for aggregation in AGGREGATIONS:
        rows = [r for r in gensg_rows if r["aggregation"] == aggregation][:3]
        report_lines.append(f"- {aggregation}:")
        for row in rows:
            report_lines.append(
                f"  - layer {row['layer']} head {row['head']}, "
                f"var={float(row['score_variance_mean']):.4g}, selected_entropy={float(row['selected_index_entropy']):.4g}"
            )
    report_lines += ["", "## Pure 2.2 Score References", ""]
    for name in PURE_SCORE_NAMES:
        rows = top_by(name, 3)
        report_lines.append(f"- {name}:")
        for row in rows:
            report_lines.append(
                f"  - {row['aggregation']} layer {row['layer']} head {row['head']}, "
                f"var={float(row['score_variance_mean']):.4g}, selected_entropy={float(row['selected_index_entropy']):.4g}"
            )
    report_lines += ["", "## GenSG Ablations and Control Score Rows", ""]
    for name in ("no_q", "no_W", "random_token", "shuffled_token", "random_prefix_head", "center_prior"):
        rows = top_by(name, 3)
        report_lines.append(f"- {name}:")
        for row in rows:
            report_lines.append(
                f"  - {row['aggregation']} layer {row['layer']} head {row['head']}, "
                f"var={float(row['score_variance_mean']):.4g}, selected_entropy={float(row['selected_index_entropy']):.4g}"
            )
    if phase2_recommendation is not None:
        report_lines += ["", "## Same-Head Controls for Recommended Configuration", ""]
        for name in ("gensg", "no_q", "no_W", *control_score_names):
            row = find_metric(
                name,
                phase2_recommendation["aggregation"],
                phase2_recommendation["layer"],
                phase2_recommendation["head"],
            )
            if row is None:
                continue
            report_lines.append(
                f"- {name}: var={float(row['score_variance_mean']):.4g}, "
                f"selected_entropy={float(row['selected_index_entropy']):.4g}, counts={row['selected_index_counts']}"
            )
        if not controls_not_better:
            report_lines.append(
                "- Negative controls are comparable to or stronger than GenSG at the recommended configuration; "
                "阶段一证据不足，不能进入大规模阶段二。"
            )
    report_lines += ["", "## Top Token Quality Examples", ""]
    for row in obs_rows[: min(8, len(obs_rows))]:
        report_lines.append(
            f"- obs {row['obs_index']} {row['suite']} task {row['task_id']}: {row['top_tokens']}"
        )
    report_lines += [
        "",
        "## Outputs",
        "",
        "- results/stage1_gensg_head_time_metrics.csv",
        "- results/stage1_gensg_token_quality.csv",
        "- results/stage1_gensg_controls.csv",
        "- results/stage1_gensg_obs_summaries.json",
        "- results/stage1_gensg_obs*.npz",
        "- figures/stage1_gensg_obs*_l17h0_last3.png",
    ]
    report_path = reports_dir / "stage1_gensg_sanity_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary = {
        "timestamp": dt.datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "conclusion": conclusion,
        "sanity": {
            "shape_ok": shape_ok,
            "prefix_ok": prefix_ok,
            "q_ok": q_ok,
            "token_maps_ok": token_maps_ok,
            "token_purity_ok": token_purity_ok,
            "top1_function_or_relation_like_rate": top1_nonentity_rate,
            "top3_function_or_relation_like_rate": top3_nonentity_rate,
            "gensg_var_ok": gensg_var_ok,
            "selected_ok": selected_ok,
            "controls_not_better": controls_not_better,
            "recommended_control_max": recommended_control_max,
            "late_better_than_early": late_better_than_early,
            "late_better_than_all": late_better_than_all,
        },
        "top_gensg": top_gensg,
        "recommended_stage2": phase2_recommendation,
        "outputs": {
            "report": str(report_path.relative_to(root)),
            "metrics": str((results_dir / "stage1_gensg_head_time_metrics.csv").relative_to(root)),
            "token_quality": str((results_dir / "stage1_gensg_token_quality.csv").relative_to(root)),
        },
    }
    with (results_dir / "stage1_gensg_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
