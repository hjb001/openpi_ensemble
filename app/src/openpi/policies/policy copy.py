from collections.abc import Sequence
import inspect
import logging
import pathlib
import time
from typing import Any, TypeAlias
import copy
import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

# Test-time scaling (Policy.tts=True): longer denoising + multi-sample mean for self-consistency.
_TTS_NUM_STEPS_FLOOR = 50
_TTS_SELF_CONSISTENCY_SAMPLES = 5
_TTS_MAX_DECODING_STEPS_FLOOR = 384


def _blend_overlapping_chunk(
    prev: np.ndarray | None, curr: np.ndarray, te: int, beta: float
) -> np.ndarray:
    """Blend the first `te` steps of `curr` with the tail of `prev` (ACT-style overlap).

    After executing `te` steps from the previous prediction `prev`, index `prev[te + k]`
    targets the same instant as `curr[k]`. Overlap exists for `k < T - te` where `T = len(curr)`.
    `beta` is the weight on the **new** chunk `curr` (1 - beta on `prev`).
    """
    curr = np.asarray(curr)
    te = int(min(te, curr.shape[0]))
    if te <= 0:
        return curr[:0]

    out = np.asarray(curr[:te], dtype=np.float64).copy()
    if prev is None:
        return out.astype(curr.dtype, copy=False)

    prev = np.asarray(prev)
    if prev.shape[0] < te or prev.shape[1:] != curr.shape[1:]:
        return np.asarray(curr[:te], dtype=curr.dtype)

    t = curr.shape[0]
    n_blend = min(te, max(0, t - te))
    if n_blend > 0:
        pb = prev[te : te + n_blend].astype(np.float64, copy=False)
        cb = curr[:n_blend].astype(np.float64, copy=False)
        out[:n_blend] = beta * cb + (1.0 - beta) * pb
    return out.astype(curr.dtype, copy=False)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tts: bool = False,
        partial_execution_steps: int | None = None,
        temporal_ensemble: bool = False,
        temporal_ensemble_beta: float = 0.5,
    ):
        if temporal_ensemble and partial_execution_steps is None:
            raise ValueError("temporal_ensemble requires partial_execution_steps to be set")
        if partial_execution_steps is not None and partial_execution_steps < 1:
            raise ValueError("partial_execution_steps must be >= 1 when set")
        if not (0.0 <= temporal_ensemble_beta <= 1.0):
            raise ValueError("temporal_ensemble_beta must be in [0, 1]")

        self._sample_action_param_names = set(inspect.signature(model.sample_actions).parameters.keys())
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = dict(metadata or {})

        self._tts = tts
        self._partial_execution_steps = partial_execution_steps

        if partial_execution_steps is not None:
            self._metadata["partial_execution_steps"] = partial_execution_steps


    def _effective_sample_kwargs(self) -> dict[str, Any]:
        """Kwargs for one `sample_actions` call. When tts is on, extend supported knobs only."""
        kw = dict(self._sample_kwargs)
        if not self._tts:
            return kw
        if "num_steps" in self._sample_action_param_names:
            base = int(kw.get("num_steps", 10))
            kw["num_steps"] = max(base, _TTS_NUM_STEPS_FLOOR)
        if "max_decoding_steps" in self._sample_action_param_names:
            base = int(kw.get("max_decoding_steps", 256))
            kw["max_decoding_steps"] = max(base, _TTS_MAX_DECODING_STEPS_FLOOR)
        return kw

    @staticmethod
    def _merge_self_consistent_float(samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Average float `actions` / `coarse_actions` across samples; other keys from the first sample."""
        out = dict(samples[0])
        for key in ("actions", "coarse_actions"):
            if key not in out:
                continue
            stacked = jnp.stack([s[key] for s in samples], axis=0)
            if stacked.dtype.kind != "f":
                logging.warning(
                    "TTS self-consistency: key %r is not float; using first sample only (no averaging).", key
                )
                continue
            out[key] = jnp.mean(stacked, axis=0)
        return out

    def _sample_to_batched_outputs(
        self, sample_rng: at.KeyArrayLike, inputs: dict[str, Any], sample_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        outputs: dict[str, Any] = {"state": inputs["state"]}
        result = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **sample_kwargs)
        if isinstance(result, dict):
            outputs.update(result)
        else:
            outputs["actions"] = result
        return outputs

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        sample_kwargs = self._effective_sample_kwargs()

        if not self._tts:
            self._rng, sample_rng = jax.random.split(self._rng)
            outputs = self._sample_to_batched_outputs(sample_rng, inputs, sample_kwargs)
        else:
            batched_samples: list[dict[str, Any]] = []
            for _ in range(_TTS_SELF_CONSISTENCY_SAMPLES):
                self._rng, sample_rng = jax.random.split(self._rng)
                batched_samples.append(self._sample_to_batched_outputs(sample_rng, inputs, sample_kwargs))
            outputs = self._merge_self_consistent_float(batched_samples)

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        timing: dict[str, Any] = {"infer_ms": model_time * 1000}
        if self._tts:
            timing["tts"] = True
            timing["tts_num_samples"] = _TTS_SELF_CONSISTENCY_SAMPLES
            if "num_steps" in sample_kwargs:
                timing["tts_num_steps"] = sample_kwargs["num_steps"]
            if "max_decoding_steps" in sample_kwargs:
                timing["tts_max_decoding_steps"] = sample_kwargs["max_decoding_steps"]
        if self._partial_execution_steps is not None:
            timing["partial_execution_steps"] = self._partial_execution_steps
        outputs["policy_timing"] = timing
        outputs = self.post_process(obs, outputs)
        return self._apply_partial_execution_and_ensemble(outputs)

    def _apply_partial_execution_and_ensemble(self, outputs: dict[str, Any]) -> dict[str, Any]:
        pass
        return out

    @override
    def reset(self) -> None:
        self._prev_actions = None
        self._prev_coarse_actions = None

    def post_process(self, obs: dict, outputs: dict) -> dict:
        task_name_requiring_waist = ["sorting_packages", "sorting_packages_continuous"]
        task_name = jax.tree.map(lambda x: x, obs).get("task_name", None)

        if task_name is None:
            return outputs

        print(f"Policy infering for task: {task_name}, with inference time: {outputs['policy_timing']['infer_ms']:.3f} ms")
        if task_name not in task_name_requiring_waist:
            # cut off waist actions for tasks that don't require it
            outputs["actions"] = outputs["actions"][:, :16]

        else:
            raw_state = jax.tree.map(lambda x: x, obs).get("state", None)
            assert raw_state is not None, "State is required for post-processing waist actions"
            # freeze four waist actions to the current state, utilizing only the last action for policy output
            outputs["actions"][:, 16:20] = raw_state[16:20]

        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def reset(self) -> None:
        self._policy.reset()

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
