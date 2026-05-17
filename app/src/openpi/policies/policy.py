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

# Optional top-level obs key: per-call TTS (e.g. task-routed ensemble injects without duplicating weights).
POLICY_INFER_TTS_KEY = "policy_infer_tts"
POLICY_INFER_TTS_V2_KEY = "policy_infer_tts_v2"
POLICY_INFER_TTS_V2_NUM_STEPS_KEY = "policy_infer_tts_v2_num_steps"


def infer_tts_override_from_obs(obs: Any) -> bool | None:
    """If ``obs`` is a dict containing ``POLICY_INFER_TTS_KEY``, return its bool value; else ``None``."""
    flat = jax.tree.map(lambda x: x, obs)
    if not isinstance(flat, dict) or POLICY_INFER_TTS_KEY not in flat:
        return None
    v = flat[POLICY_INFER_TTS_KEY]
    if isinstance(v, (bytes, bytearray)):
        v = v.decode()
    if hasattr(v, "item") and callable(getattr(v, "item", None)):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes")
    return bool(v)


def infer_tts_v2_override_from_obs(obs: Any) -> bool | None:
    """If ``obs`` contains ``POLICY_INFER_TTS_V2_KEY``, return its bool value; else ``None``."""
    flat = jax.tree.map(lambda x: x, obs)
    if not isinstance(flat, dict) or POLICY_INFER_TTS_V2_KEY not in flat:
        return None
    v = flat[POLICY_INFER_TTS_V2_KEY]
    if isinstance(v, (bytes, bytearray)):
        v = v.decode()
    if hasattr(v, "item") and callable(getattr(v, "item", None)):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes")
    return bool(v)


def infer_tts_v2_num_steps_from_obs(obs: Any) -> int | None:
    """If ``obs`` contains ``POLICY_INFER_TTS_V2_NUM_STEPS_KEY``, return its int value; else ``None``."""
    flat = jax.tree.map(lambda x: x, obs)
    if not isinstance(flat, dict) or POLICY_INFER_TTS_V2_NUM_STEPS_KEY not in flat:
        return None
    v = flat[POLICY_INFER_TTS_V2_NUM_STEPS_KEY]
    if hasattr(v, "item") and callable(getattr(v, "item", None)):
        try:
            v = v.item()
        except Exception:
            pass
    return int(v)


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
        partial_execution_steps: int | None = None,
        tts: bool = False,
    ):
        if partial_execution_steps is not None and partial_execution_steps < 1:
            raise ValueError("partial_execution_steps must be >= 1 when set")

        # tts change here
        self._sample_action_param_names = set(inspect.signature(model.sample_actions).parameters.keys())
        
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = dict(metadata or {})

        self._partial_execution_steps = partial_execution_steps
        self._tts = tts

        if partial_execution_steps is not None:
            self._metadata["partial_execution_steps"] = partial_execution_steps
        self._metadata["policy_tts"] = self._tts

    # 为了加上tts就改了这里
    def _effective_sample_kwargs(self, *, tts: bool) -> dict[str, Any]:
        """Kwargs for one `sample_actions` call. When tts is on, extend supported knobs only."""
        kw = dict(self._sample_kwargs)
        if not tts:
            return kw
        if "num_steps" in self._sample_action_param_names:
            base = int(kw.get("num_steps", 10))
            kw["num_steps"] = max(base, _TTS_NUM_STEPS_FLOOR)
        if "max_decoding_steps" in self._sample_action_param_names:
            base = int(kw.get("max_decoding_steps", 256))
            kw["max_decoding_steps"] = max(base, _TTS_MAX_DECODING_STEPS_FLOOR)
        return kw

    def _effective_sample_kwargs_v2(self, *, num_steps: int) -> dict[str, Any]:
        """TTS v2: only increase num_steps (no self-consistency averaging)."""
        kw = dict(self._sample_kwargs)
        if "num_steps" in self._sample_action_param_names:
            base = int(kw.get("num_steps", 10))
            kw["num_steps"] = max(base, num_steps)
        if "max_decoding_steps" in self._sample_action_param_names:
            base = int(kw.get("max_decoding_steps", 256))
            kw["max_decoding_steps"] = max(base, num_steps)
        return kw

    # 为了加上tts就改了这里
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

    # 为了加上tts就改了这里
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

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        tts_override = infer_tts_override_from_obs(obs)
        effective_tts = self._tts if tts_override is None else tts_override

        # TTS v2: steps-only, no self-consistency averaging
        tts_v2_override = infer_tts_v2_override_from_obs(obs)
        effective_tts_v2 = tts_v2_override if tts_v2_override is not None else False
        tts_v2_num_steps = infer_tts_v2_num_steps_from_obs(obs) if effective_tts_v2 else None
        # tts_v2 takes priority over tts
        if effective_tts_v2:
            effective_tts = False

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        _control_keys = {POLICY_INFER_TTS_KEY, POLICY_INFER_TTS_V2_KEY, POLICY_INFER_TTS_V2_NUM_STEPS_KEY}
        if isinstance(inputs, dict):
            inputs = {k: v for k, v in inputs.items() if k not in _control_keys}
        inputs = self._input_transform(inputs)

        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        """
        self._rng, sample_rng = jax.random.split(self._rng)         
        outputs = {
            "state": inputs["state"]
        }
        result = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs)

        if isinstance(result, dict):
            outputs.update(result)    
        else:
            outputs["actions"] = result
        # outputs["actions"] = inputs["actions"]
        """
        ########## WE CHANGE HERE ##########
        if effective_tts_v2:
            # TTS v2: only increase steps, single forward pass (no averaging)
            sample_kwargs = self._effective_sample_kwargs_v2(num_steps=tts_v2_num_steps or _TTS_NUM_STEPS_FLOOR)
            self._rng, sample_rng = jax.random.split(self._rng)
            outputs = self._sample_to_batched_outputs(sample_rng, inputs, sample_kwargs)
        elif effective_tts:
            sample_kwargs = self._effective_sample_kwargs(tts=True)
            batched_samples: list[dict[str, Any]] = []
            for _ in range(_TTS_SELF_CONSISTENCY_SAMPLES):
                self._rng, sample_rng = jax.random.split(self._rng)
                batched_samples.append(self._sample_to_batched_outputs(sample_rng, inputs, sample_kwargs))
            outputs = self._merge_self_consistent_float(batched_samples)
        else:
            sample_kwargs = self._effective_sample_kwargs(tts=False)
            self._rng, sample_rng = jax.random.split(self._rng)
            outputs = self._sample_to_batched_outputs(sample_rng, inputs, sample_kwargs)
        ########## WE CHANGE HERE ##########

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        timing: dict[str, Any] = {"infer_ms": model_time * 1000}

        if effective_tts_v2:
            timing["tts_v2"] = True
            if "num_steps" in sample_kwargs:
                timing["tts_v2_num_steps"] = sample_kwargs["num_steps"]
            if "max_decoding_steps" in sample_kwargs:
                timing["tts_v2_max_decoding_steps"] = sample_kwargs["max_decoding_steps"]
        elif effective_tts:
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
        return self._apply_partial_execution(outputs)

    def _apply_partial_execution(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Keep only the first N steps of executable `actions` (receding horizon).

        `coarse_actions` is auxiliary model output and is left unchanged; only environments
        use `actions` for control.
        """
        if self._partial_execution_steps is None:
            return outputs

        actions = outputs.get("actions")
        if actions is None:
            return outputs

        te = self._partial_execution_steps
        actions_arr = np.asarray(actions)
        t = actions_arr.shape[0]
        te_eff = min(te, t)

        out = dict(outputs)
        out["actions"] = actions_arr[:te_eff]
        return out

    def post_process(self, obs: dict, outputs: dict) -> dict:
        task_name_requiring_waist = ["sorting_packages", "sorting_packages_continuous", "sorting_packages_real"]
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
            #outputs["actions"][:, 16:20] = raw_state[16:20]

        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def task_name_from_obs(obs: dict) -> str | None:
    """Read `task_name` from a (possibly nested) observation dict; mirrors `post_process` access pattern."""
    flat = jax.tree.map(lambda x: x, obs)
    tn = flat.get("task_name", None)
    if tn is None:
        return None
    if isinstance(tn, (bytes, bytearray)):
        return tn.decode()
    if hasattr(tn, "item") and callable(getattr(tn, "item", None)):
        try:
            tn = tn.item()
        except Exception:
            pass
    return str(tn)


class TaskRoutedEnsemblePolicy(BasePolicy):
    """Holds multiple `Policy` instances and dispatches `infer` by `obs['task_name']`.

    Use this to combine checkpoints that excel on different subtasks; set `routes` so each
    task name maps to the right model, and use `default_policy` when `task_name` is missing
    or not listed.

    Per-task TTS is applied by injecting ``policy_infer_tts`` into a shallow copy of ``obs`` so each
    unique checkpoint stays **one** loaded `Policy` (no duplicate weights for TTS on/off).

    Per-task tricks (``sorting_packages_continuous_prompt``, ``rule_based_sorting_correction``,
    ``clean_the_desktop_correction``) are injected as boolean obs keys so outer wrappers can
    activate per-request.
    """

    _OBS_INJECTABLE_TRICKS = frozenset({
        "sorting_packages_prompt",
        "sorting_packages_continuous_prompt",
        "sorting_packages_continuous_prompt_v2",
        "rule_based_sorting_correction",
        "clean_the_desktop_correction",
        "dynamic_tts_sorting",
    })

    def __init__(
        self,
        policies_by_task: dict[str, Policy],
        default_policy: Policy,
        *,
        tts_by_task: dict[str, bool],
        default_tts: bool,
        log_routing: bool = True,
        tricks_by_task: dict[str, list[str]] | None = None,
        tts_v2_by_task: dict[str, bool] | None = None,
        tts_v2_num_steps_by_task: dict[str, int | None] | None = None,
    ):
        if not policies_by_task:
            raise ValueError("policies_by_task must be non-empty")
        self._by_task = dict(policies_by_task)
        self._default = default_policy
        self._tts_by_task = dict(tts_by_task)
        self._default_tts = default_tts
        self._log_routing = log_routing
        self._tricks_by_task: dict[str, list[str]] = dict(tricks_by_task or {})
        self._tts_v2_by_task: dict[str, bool] = dict(tts_v2_by_task or {})
        self._tts_v2_num_steps_by_task: dict[str, int | None] = dict(tts_v2_num_steps_by_task or {})
        self._last_task: str | None = None  # for task-switch banner
        if set(self._tts_by_task) != set(self._by_task):
            raise ValueError("tts_by_task keys must match policies_by_task keys exactly")
        self._metadata: dict[str, Any] = {
            "ensemble": "task_routed",
            "routed_tasks": sorted(self._by_task.keys()),
            "per_task_tts": dict(self._tts_by_task),
            "per_task_tts_v2": dict(self._tts_v2_by_task),
            "per_task_tts_v2_num_steps": {k: v for k, v in self._tts_v2_num_steps_by_task.items() if v is not None},
            "per_task_tricks": dict(self._tricks_by_task),
            "default_tts": self._default_tts,
            "default_policy_metadata": dict(default_policy.metadata),
            "per_task_policy_metadata": {k: dict(v.metadata) for k, v in self._by_task.items()},
        }

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        task = task_name_from_obs(obs)
        if task is not None and task in self._by_task:
            policy = self._by_task[task]
            use_tts = self._tts_by_task[task]
            tricks = self._tricks_by_task.get(task, [])
            routed = True
        else:
            policy = self._default
            use_tts = self._default_tts
            tricks = []
            routed = False

        use_tts_v2 = self._tts_v2_by_task.get(task, False) if task else False
        tts_v2_num_steps = self._tts_v2_num_steps_by_task.get(task) if use_tts_v2 else None

        # Print a prominent banner when the task changes (new episode / task switch).
        if task != self._last_task:
            prev = self._last_task
            self._last_task = task
            print(flush=True)
            print("=" * 72, flush=True)
            print(f"[Ensemble] TASK SWITCH: {prev!r} -> {task!r}", flush=True)
            print(f"  tts        : {use_tts}", flush=True)
            print(f"  tts_v2     : {use_tts_v2} (num_steps={tts_v2_num_steps})", flush=True)
            print(f"  tricks     : {tricks}", flush=True)
            print(f"  routed     : {routed}", flush=True)
            print("=" * 72, flush=True)

        if self._log_routing:
            if routed:
                logging.info("TaskRoutedEnsemblePolicy: task_name=%r -> routed sub-policy tts=%r tts_v2=%r tricks=%s", task, use_tts, use_tts_v2, tricks)
            else:
                logging.info("TaskRoutedEnsemblePolicy: task_name=%r -> default sub-policy tts=%r", task, use_tts)
        if not isinstance(obs, dict):
            logging.warning(
                "TaskRoutedEnsemblePolicy: obs is not a dict; cannot set %r (using sub-policy defaults).",
                POLICY_INFER_TTS_KEY,
            )
            return policy.infer(obs)
        obs_merged = dict(obs)
        # Use setdefault so outer wrappers (e.g. DynamicTtsSortingWrapper) can override.
        obs_merged.setdefault(POLICY_INFER_TTS_KEY, use_tts)
        # Inject tts_v2 settings
        obs_merged.setdefault(POLICY_INFER_TTS_V2_KEY, use_tts_v2)
        if use_tts_v2 and tts_v2_num_steps is not None:
            obs_merged.setdefault(POLICY_INFER_TTS_V2_NUM_STEPS_KEY, tts_v2_num_steps)
        # Inject per-task trick flags so outer wrappers know whether to activate.
        for trick_name in self._OBS_INJECTABLE_TRICKS:
            obs_merged.setdefault(trick_name, trick_name in tricks)
        return policy.infer(obs_merged)

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
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
