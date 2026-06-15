from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import socket
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import torch

from openpi.models import model as _model
from openpi.policies import policy as _policy

from igc_select.stage1_offline_sanity import encode_prefix_context, sample_k_candidates, score_candidates_with_attention
from igc_gensg.scripts.stage1_gensg_sanity import (
    aggregate_steps,
    compute_grounding_quality,
    compute_pure_scores,
    content_token_mask,
    encode_prefix_context_with_grounding,
    gensg_score,
    repeat_cache,
    repeat_tensor_batch,
    sample_k_with_generation_image_language_attention,
)


BASELINE_METHODS = {"pi0_k1", "k4_first", "k4_random"}
RESCORE_METHODS = {"2.1_rescore_best"}
PURE_22_METHODS = {"2.2_pure_last1_imc", "2.2_pure_last3_imc"}
GENSG_METHODS = {"gensg_last1", "gensg_last3", "gensg_last3_no_q", "gensg_last3_no_W"}
STAGE3_CONTROL_METHODS = {
    "gensg_early",
    "gensg_middle",
    "gensg_all_steps",
    "gensg_last3_random_token_map",
    "gensg_last3_shuffled_token_map",
    "gensg_last3_random_prefix_head",
    "gensg_last3_random_action_head",
    "gensg_last3_bottom_action_head",
    "gensg_last3_prefix_only",
    "gensg_last3_score_mismatch",
}
ALL_METHODS = BASELINE_METHODS | RESCORE_METHODS | PURE_22_METHODS | GENSG_METHODS | STAGE3_CONTROL_METHODS


def parse_heads(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            layer_s, head_s = part.split(":", 1)
        elif "/" in part:
            layer_s, head_s = part.split("/", 1)
        else:
            raise ValueError(f"Head spec must be layer:head, got {part!r}")
        out.append((int(layer_s), int(head_s)))
    if not out:
        raise ValueError("At least one layer/head is required")
    return out


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(type(obj).__name__)


def _pairwise_l2(actions: np.ndarray) -> tuple[float, float, float]:
    flat = np.asarray(actions, dtype=np.float64).reshape(actions.shape[0], -1)
    vals = []
    for i in range(flat.shape[0]):
        for j in range(i + 1, flat.shape[0]):
            vals.append(float(np.linalg.norm(flat[i] - flat[j])))
    if not vals:
        return 0.0, 0.0, 0.0
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def _selected_index_entropy(indices: list[int], k: int) -> float:
    if not indices or k <= 1:
        return 0.0
    counts = Counter(indices)
    total = sum(counts.values())
    probs = np.asarray([counts.get(i, 0) / total for i in range(k)], dtype=np.float64)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum() / math.log(k))


def _method_time(method: str) -> str:
    if method in {"2.2_pure_last1_imc", "gensg_last1"}:
        return "last1"
    if method in {
        "2.2_pure_last3_imc",
        "gensg_last3",
        "gensg_last3_no_q",
        "gensg_last3_no_W",
        "gensg_last3_random_token_map",
        "gensg_last3_shuffled_token_map",
        "gensg_last3_random_prefix_head",
        "gensg_last3_random_action_head",
        "gensg_last3_bottom_action_head",
        "gensg_last3_prefix_only",
        "gensg_last3_score_mismatch",
    }:
        return "last3_mean"
    if method == "gensg_early":
        return "early1"
    if method == "gensg_middle":
        return "middle1"
    if method == "gensg_all_steps":
        return "all_steps_mean"
    return ""


@dataclass
class GenSGIGCConfig:
    method: str = "gensg_last3"
    generation_heads: tuple[tuple[int, int], ...] = ((17, 0),)
    rescore_heads: tuple[tuple[int, int], ...] = ((17, 0),)
    k: int = 4
    num_steps: int = 10
    scoring_timestep: float = 1e-3
    camera: str = "base_0_rgb"
    seed: int = 0
    prefix_top_heads: int = 4
    execution_action_tokens: int = 5
    attention_dir: pathlib.Path = pathlib.Path("igc_gensg/figures/stage2_gensg_attention")
    save_attention_npz: bool = True


class GenSGIGCPolicy(_policy.BasePolicy):
    """Online GenSG-IGC policy wrapper.

    GenSG methods score candidates from attention captured during the denoising
    generation loop. The 2.1 method remains a post-hoc rescoring baseline only.
    """

    def __init__(self, base_policy: _policy.Policy, config: GenSGIGCConfig):
        if not getattr(base_policy, "_is_pytorch_model", False):
            raise ValueError("GenSGIGCPolicy supports PyTorch policies only")
        if config.method not in ALL_METHODS:
            raise ValueError(f"Unknown method {config.method!r}; expected one of {sorted(ALL_METHODS)}")
        if config.method == "pi0_k1" and config.k != 1:
            config = GenSGIGCConfig(**{**config.__dict__, "k": 1})
        self._base = base_policy
        self._model = base_policy._model
        self._input_transform = base_policy._input_transform
        self._output_transform = base_policy._output_transform
        self._device = base_policy._pytorch_device
        self._config = config
        self._query_index = 0
        self._rng = np.random.default_rng(config.seed)
        self._selected_history: list[int] = []
        self._metadata = dict(base_policy.metadata)
        self._metadata.update(
            {
                "server_type": "gensg_igc_stage2",
                "method": config.method,
                "generation_heads": [list(x) for x in config.generation_heads],
                "rescore_heads": [list(x) for x in config.rescore_heads],
                "k": int(config.k),
                "num_steps": int(config.num_steps),
                "scoring_timestep": float(config.scoring_timestep),
                "camera": config.camera,
                "prefix_top_heads": int(config.prefix_top_heads),
                "execution_action_tokens": int(config.execution_action_tokens),
                "action_horizon": int(self._model.config.action_horizon),
                "action_dim": int(self._model.config.action_dim),
                "pi05": bool(getattr(self._model.config, "pi05", False)),
            }
        )
        config.attention_dir.mkdir(parents=True, exist_ok=True)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _prepare_observation(self, obs: dict[str, Any]) -> tuple[_model.Observation, dict[str, Any]]:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._device)[None, ...], inputs)
        return _model.Observation.from_dict(inputs), inputs

    def _to_env_actions(self, selected_actions: torch.Tensor, inputs: dict[str, Any]) -> np.ndarray:
        outputs = {"state": inputs["state"], "actions": selected_actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        outputs = self._output_transform(outputs)
        return np.asarray(outputs["actions"], dtype=np.float32)

    def _repeat_prefix_context(self, prefix_context: dict[str, Any], k: int) -> dict[str, Any]:
        return {
            "state": repeat_tensor_batch(prefix_context["state"], k),
            "prefix_pad_masks": repeat_tensor_batch(prefix_context["prefix_pad_masks"], k),
            "past_key_values": repeat_cache(prefix_context["past_key_values"], k),
            "prefix_out": None
            if prefix_context.get("prefix_out") is None
            else repeat_tensor_batch(prefix_context["prefix_out"], k),
            "prefix_spans": prefix_context["prefix_spans"],
            "prefix_len": prefix_context["prefix_len"],
            "image_keys": prefix_context["image_keys"],
        }

    def _save_attention(
        self,
        *,
        context: dict[str, Any],
        method: str,
        selected_heads: tuple[tuple[int, int], ...],
        arrays: dict[str, np.ndarray],
        metadata: dict[str, Any],
    ) -> str | None:
        if not self._config.save_attention_npz:
            return None
        if not bool(context.get("save_attention", self._config.save_attention_npz)):
            return None
        task = str(context.get("task", "task")).replace("/", "_").replace(" ", "_")[:80]
        ep = int(context.get("episode_id", -1))
        q = int(context.get("query_index", self._query_index))
        job_id = str(context.get("job_id", "")).replace("/", "_").replace(" ", "_")[:100]
        prefix = f"{job_id}_" if job_id else ""
        out = self._config.attention_dir / f"{prefix}{method}_{task}_ep{ep:03d}_q{q:03d}.npz"
        payload = {
            "selected_heads": np.asarray(selected_heads, dtype=np.int32),
            "metadata_json": json.dumps(metadata, ensure_ascii=False, default=_json_default),
            **arrays,
        }
        np.savez_compressed(out, **payload)
        return str(out)

    @torch.no_grad()
    def _score_21(
        self,
        prefix_context: dict[str, Any],
        actions_k: torch.Tensor,
        context: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        layers = sorted({layer for layer, _ in self._config.rescore_heads})
        repeated_context = self._repeat_prefix_context(prefix_context, int(actions_k.shape[0]))
        t0 = time.monotonic()
        score_out = score_candidates_with_attention(
            self._model,
            repeated_context,
            actions_k,
            camera=self._config.camera,
            timestep=self._config.scoring_timestep,
            layers=layers,
            action_token_count=self._config.execution_action_tokens,
        )
        scoring_ms = (time.monotonic() - t0) * 1000.0
        per_head_scores = []
        arrays = {}
        for layer, head in self._config.rescore_heads:
            rec = score_out["records"][(layer, head)]
            per_head_scores.append(np.asarray(rec["score_imc"], dtype=np.float64))
            arrays[f"rescore_l{layer}_h{head}"] = np.asarray(rec["grids"], dtype=np.float32)
        scores = np.mean(np.stack(per_head_scores, axis=0), axis=0)
        attention_path = self._save_attention(
            context=context,
            method="2.1_rescore_best",
            selected_heads=self._config.rescore_heads,
            arrays={**arrays, "scores": scores.astype(np.float32)},
            metadata={
                "score_name": "score_imc",
                "time_aggregation": "t0_rescore",
                "action_token_count": int(self._config.execution_action_tokens),
            },
        )
        return scores, {
            "scoring_ms": scoring_ms,
            "attention_map_path": attention_path,
            "attention_shape_first_layer": score_out.get("attention_shape_first_layer"),
            "layer": int(self._config.rescore_heads[0][0]),
            "head": int(self._config.rescore_heads[0][1]),
            "time_aggregation": "t0_rescore",
            "score_name": "score_imc",
            "action_token_count": int(self._config.execution_action_tokens),
        }

    def _token_summaries(
        self,
        *,
        token_text: list[str],
        token_mask: np.ndarray,
        q_values: np.ndarray,
        w_values: np.ndarray | None = None,
        contrib_values: np.ndarray | None = None,
        n: int = 8,
    ) -> dict[str, list[dict[str, Any]]]:
        valid = np.asarray(token_mask, dtype=bool)

        def top(values: np.ndarray | None) -> list[dict[str, Any]]:
            if values is None:
                return []
            arr = np.asarray(values, dtype=np.float64)
            arr = np.where(valid, arr, -np.inf)
            idxs = [int(x) for x in np.argsort(arr)[::-1] if valid[int(x)]][:n]
            return [{"idx": idx, "text": str(token_text[idx]), "value": float(values[idx])} for idx in idxs]

        return {
            "top_tokens_by_q": top(q_values),
            "top_tokens_by_W": top(w_values),
            "top_contributing_tokens": top(contrib_values),
        }

    def _build_grounding(self, prefix_context: dict[str, Any]) -> dict[str, Any]:
        raw_token_mask = np.asarray(prefix_context["token_mask"], dtype=bool)
        token_text = list(prefix_context["token_text"])
        token_ids = np.asarray(prefix_context["token_ids"])
        token_mask = content_token_mask(token_ids, token_text, raw_token_mask)
        grounding = compute_grounding_quality(
            prefix_context["prefix_grounding"],
            token_mask,
            prefix_layers=prefix_context["prefix_layers"],
            token_ids=token_ids,
            top_heads=self._config.prefix_top_heads,
            seed=int(self._config.seed + self._query_index * 17),
        )
        grounding["token_mask"] = token_mask
        grounding["raw_token_mask"] = raw_token_mask
        grounding["token_ids"] = token_ids
        grounding["token_text"] = token_text
        return grounding

    def _score_generation(
        self,
        run: dict[str, Any],
        grounding: dict[str, Any],
        context: dict[str, Any],
        method: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        aggregation = _method_time(method)
        if not aggregation:
            raise ValueError(f"not a generation method: {method}")
        agg_img = aggregate_steps(run["action_image"], aggregation)  # [L,K,H,G,G]
        agg_lang = aggregate_steps(run["action_language"], aggregation)  # [L,K,H,J]
        layer_indices = list(run["layers"])
        token_mask = np.asarray(grounding["token_mask"], dtype=bool)
        q_values = np.asarray(grounding["q"], dtype=np.float64)
        token_maps = np.asarray(grounding["maps"], dtype=np.float32)
        token_text = list(grounding["token_text"])
        per_head_scores = []
        arrays: dict[str, np.ndarray] = {}
        head_details = []
        selected_score_name = "gensg"
        selected_contrib = None
        selected_w = None
        selected_grids = None
        selected_maps_for_score = token_maps
        for layer, head in self._config.generation_heads:
            local_layer = layer_indices.index(layer)
            grids = np.asarray(agg_img[local_layer, :, head], dtype=np.float32)
            langs = np.asarray(agg_lang[local_layer, :, head], dtype=np.float32)
            raw_scores = None
            if method in PURE_22_METHODS:
                pure = compute_pure_scores(grids)
                scores = np.asarray(pure["score_imc"], dtype=np.float64)
                contrib = np.zeros((scores.shape[0], token_mask.shape[0]), dtype=np.float64)
                selected_score_name = "score_imc"
            else:
                q_for_score = np.ones_like(q_values, dtype=np.float64) if method in {"gensg_last3_no_q"} else q_values
                lang_for_score = np.ones_like(langs, dtype=np.float32) if method in {"gensg_last3_no_W"} else langs
                maps_for_score = token_maps
                if method == "gensg_last3_random_token_map":
                    maps_for_score = np.asarray(grounding["random_token_maps"], dtype=np.float32)
                elif method == "gensg_last3_shuffled_token_map":
                    maps_for_score = np.asarray(grounding["shuffled_maps"], dtype=np.float32)
                elif method == "gensg_last3_random_prefix_head":
                    maps_for_score = np.asarray(grounding["random_maps"], dtype=np.float32)
                    q_for_score = np.asarray(grounding["random_q"], dtype=np.float64)
                if method == "gensg_last3_prefix_only":
                    static_score = float(np.sum(np.asarray(q_for_score, dtype=np.float64)[token_mask]))
                    seed_text = json.dumps(
                        {
                            "job_id": context.get("job_id"),
                            "episode_id": context.get("episode_id"),
                            "query_index": context.get("query_index", self._query_index),
                            "method": method,
                        },
                        sort_keys=True,
                    )
                    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
                    # Prefix-only has no candidate-conditioned information. Use a
                    # deterministic random tie-breaker so this control behaves like
                    # static saliency, not like an accidental k4_first baseline.
                    scores = static_score + np.random.default_rng(seed).random(grids.shape[0]) * 1e-9
                    contrib = np.zeros((grids.shape[0], token_mask.shape[0]), dtype=np.float64)
                else:
                    scores, contrib = gensg_score(grids, lang_for_score, maps_for_score, q_for_score, token_mask)
                if method == "gensg_last3_score_mismatch":
                    raw_scores = np.asarray(scores, dtype=np.float64)
                    scores = np.roll(raw_scores, 1)
                selected_score_name = method
            per_head_scores.append(np.asarray(scores, dtype=np.float64))
            arrays[f"{method}_{aggregation}_l{layer}_h{head}_A"] = grids
            arrays[f"{method}_{aggregation}_l{layer}_h{head}_W"] = langs
            if raw_scores is not None:
                arrays[f"{method}_{aggregation}_l{layer}_h{head}_raw_scores"] = raw_scores.astype(np.float32)
            head_details.append({"layer": int(layer), "head": int(head), "scores": [float(x) for x in scores.tolist()]})
            if selected_contrib is None:
                selected_contrib = contrib
                selected_w = langs
                selected_grids = grids
                selected_maps_for_score = maps_for_score if "maps_for_score" in locals() else token_maps
        scores = np.mean(np.stack(per_head_scores, axis=0), axis=0)
        selected = int(np.argmax(scores))
        w_selected = None if selected_w is None else np.asarray(selected_w[selected], dtype=np.float64)
        contrib_selected = None if selected_contrib is None else np.asarray(selected_contrib[selected], dtype=np.float64)
        token_summary = self._token_summaries(
            token_text=token_text,
            token_mask=token_mask,
            q_values=q_values,
            w_values=w_selected,
            contrib_values=contrib_selected,
        )
        metadata = {
            "method": method,
            "aggregation": aggregation,
            "score_name": selected_score_name,
            "selected_index": selected,
            "head_details": head_details,
            **token_summary,
            "selected_prefix_heads": grounding.get("selected_prefix_heads"),
            "action_token_count": int(run.get("action_token_count", self._config.execution_action_tokens)),
        }
        arrays.update(
            {
                "scores": scores.astype(np.float32),
                "G": token_maps.astype(np.float32),
                "G_used": np.asarray(selected_maps_for_score, dtype=np.float32),
                "q": q_values.astype(np.float32),
                "token_mask": token_mask.astype(np.bool_),
            }
        )
        if selected_grids is not None:
            arrays["selected_A"] = np.asarray(selected_grids[selected], dtype=np.float32)
        if w_selected is not None:
            arrays["selected_W"] = np.asarray(w_selected, dtype=np.float32)
        attention_path = self._save_attention(
            context=context,
            method=method,
            selected_heads=self._config.generation_heads,
            arrays=arrays,
            metadata=metadata,
        )
        return scores, {
            "scoring_ms": 0.0,
            "generation_attention_runtime_ms": float(run.get("runtime_ms", 0.0)),
            "attention_map_path": attention_path,
            "G_j_path": attention_path,
            "A_k_path": attention_path,
            "attention_shape_first_layer": list(run["action_image"].shape),
            "action_language_shape": list(run["action_language"].shape),
            "layer": int(self._config.generation_heads[0][0]),
            "head": int(self._config.generation_heads[0][1]),
            "time_aggregation": aggregation,
            "score_name": selected_score_name,
            **token_summary,
            "selected_prefix_heads": grounding.get("selected_prefix_heads"),
            "action_token_count": int(run.get("action_token_count", self._config.execution_action_tokens)),
        }

    @torch.no_grad()
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        obs = dict(obs)
        context = dict(obs.pop("__stage2_context", {}) or {})
        query_start = time.monotonic()
        observation, inputs = self._prepare_observation(obs)
        method = self._config.method
        k = int(self._config.k)
        sample_seed = int(self._config.seed + self._query_index * 9973 + int(context.get("episode_id", 0)) * 1009)

        prefix_start = time.monotonic()
        needs_grounding = method in PURE_22_METHODS or method in GENSG_METHODS or method in STAGE3_CONTROL_METHODS
        if needs_grounding:
            prefix_context = encode_prefix_context_with_grounding(
                self._model,
                observation,
                camera=self._config.camera,
                prefix_layers=None,
            )
            grounding = self._build_grounding(prefix_context)
        else:
            prefix_context = encode_prefix_context(self._model, observation)
            grounding = None
        prefix_ms = (time.monotonic() - prefix_start) * 1000.0

        sample_start = time.monotonic()
        if method in PURE_22_METHODS or method in GENSG_METHODS or method in STAGE3_CONTROL_METHODS:
            run = sample_k_with_generation_image_language_attention(
                self._model,
                prefix_context,
                k=k,
                device=self._device,
                num_steps=self._config.num_steps,
                seed=sample_seed,
                camera=self._config.camera,
                action_layers=sorted({layer for layer, _ in self._config.generation_heads}),
                action_token_count=self._config.execution_action_tokens,
            )
            actions_k = run["actions"]
        else:
            actions_k, _noise_k, _repeated_context = sample_k_candidates(
                self._model,
                prefix_context,
                k=k,
                device=self._device,
                num_steps=self._config.num_steps,
                seed=sample_seed,
            )
            run = {
                "actions": actions_k,
                "runtime_ms": (time.monotonic() - sample_start) * 1000.0,
                "attention_shape": [],
            }
        sample_ms = (time.monotonic() - sample_start) * 1000.0
        norm_actions = actions_k.detach().float().cpu().numpy()
        pair_mean, pair_min, pair_max = _pairwise_l2(norm_actions)

        scores: np.ndarray | None = None
        score_meta: dict[str, Any] = {
            "scoring_ms": 0.0,
            "generation_attention_runtime_ms": float(run.get("runtime_ms", 0.0)),
            "attention_shape_first_layer": list(run.get("attention_shape", [])),
            "layer": int(self._config.generation_heads[0][0]),
            "head": int(self._config.generation_heads[0][1]),
            "time_aggregation": "",
            "score_name": "",
        }
        if method in PURE_22_METHODS or method in GENSG_METHODS or method in STAGE3_CONTROL_METHODS:
            assert grounding is not None
            scores, score_meta = self._score_generation(run, grounding, context, method)
            selected_idx = int(np.argmax(scores))
        elif method == "2.1_rescore_best":
            scores, score_meta = self._score_21(prefix_context, actions_k, context)
            selected_idx = int(np.argmax(scores))
        elif method in {"pi0_k1", "k4_first"}:
            selected_idx = 0
        elif method == "k4_random":
            selected_idx = int(self._rng.integers(0, k))
        else:
            digest = hashlib.sha256(f"{context}|{self._query_index}".encode("utf-8")).hexdigest()
            selected_idx = int(digest[:8], 16) % max(k, 1)

        selected = actions_k[selected_idx : selected_idx + 1]
        env_actions = self._to_env_actions(selected, inputs)
        runtime_ms = (time.monotonic() - query_start) * 1000.0
        try:
            gpu_memory_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            gpu_memory_mb = None
        self._selected_history.append(selected_idx)
        scores_list = [] if scores is None else [float(x) for x in scores.tolist()]
        score_var = 0.0 if scores is None else float(np.var(scores))
        candidate_norms = [float(np.linalg.norm(x.reshape(-1))) for x in norm_actions]
        query_record = {
            "hostname": socket.gethostname(),
            "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES", self._device),
            "task": context.get("task"),
            "instruction": context.get("task"),
            "suite": context.get("suite"),
            "task_id": context.get("task_id"),
            "episode_id": context.get("episode_id"),
            "method": method,
            "seed": context.get("seed"),
            "query_index": context.get("query_index", self._query_index),
            "K": k,
            "num_steps": int(self._config.num_steps),
            "generation_heads": [list(x) for x in self._config.generation_heads],
            "rescore_heads": [list(x) for x in self._config.rescore_heads],
            "layer": int(score_meta.get("layer", self._config.generation_heads[0][0])),
            "head": int(score_meta.get("head", self._config.generation_heads[0][1])),
            "time_aggregation": score_meta.get("time_aggregation", ""),
            "score_name": score_meta.get("score_name", ""),
            "all_candidate_scores": scores_list,
            "all_scores": scores_list,
            "selected_index": int(selected_idx),
            "selected_index_entropy_so_far": _selected_index_entropy(self._selected_history, k),
            "score_variance": score_var,
            "candidate_action_norm": candidate_norms,
            "candidate_action_pairwise_distance": pair_mean,
            "candidate_action_pairwise_l2_mean": pair_mean,
            "candidate_action_pairwise_l2_min": pair_min,
            "candidate_action_pairwise_l2_max": pair_max,
            "generation_attention_runtime_ms": float(score_meta.get("generation_attention_runtime_ms", run.get("runtime_ms", 0.0))),
            "total_policy_runtime_ms": runtime_ms,
            "runtime_ms": runtime_ms,
            "prefix_ms": prefix_ms,
            "sample_ms": sample_ms,
            "scoring_ms": float(score_meta.get("scoring_ms", 0.0)),
            "gpu_memory_mb": gpu_memory_mb,
            "attention_map_path": score_meta.get("attention_map_path"),
            "G_j_path": score_meta.get("G_j_path"),
            "A_k_path": score_meta.get("A_k_path"),
            "attention_shape_first_layer": score_meta.get("attention_shape_first_layer"),
            "action_language_shape": score_meta.get("action_language_shape"),
            "attention_camera": run.get("attention_camera", self._config.camera),
            "action_token_count": int(score_meta.get("action_token_count", run.get("action_token_count", self._config.execution_action_tokens))),
            "attention_camera_span": run.get("camera_span"),
            "language_span": run.get("language_span"),
            "prefix_image_keys": prefix_context.get("image_keys"),
            "prefix_spans_meta": dict(prefix_context.get("prefix_spans", {}).get("__meta__", {})),
            "top_tokens_by_q": score_meta.get("top_tokens_by_q", []),
            "top_tokens_by_W": score_meta.get("top_tokens_by_W", []),
            "top_contributing_tokens": score_meta.get("top_contributing_tokens", []),
            "selected_prefix_heads": score_meta.get("selected_prefix_heads", []),
            "success": None,
        }
        self._query_index += 1
        return {
            "actions": env_actions,
            "igc_select": query_record,
            "policy_timing": {"infer_ms": runtime_ms},
        }
