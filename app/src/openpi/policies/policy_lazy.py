"""Lazy task-routed ensemble: load at most N checkpoints on GPU (LRU). Used by ``scripts/serve_policy_lazy.py``."""

from collections import OrderedDict
from collections.abc import Callable
import gc
import logging
import pathlib
import threading
from typing import Any, TypeAlias

import jax
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi.policies import policy as _policy
from openpi.training import config as _train_config

BasePolicy: TypeAlias = _base_policy.BasePolicy
Policy = _policy.Policy
POLICY_INFER_TTS_KEY = _policy.POLICY_INFER_TTS_KEY
POLICY_INFER_TTS_V2_KEY = _policy.POLICY_INFER_TTS_V2_KEY
POLICY_INFER_TTS_V2_NUM_STEPS_KEY = _policy.POLICY_INFER_TTS_V2_NUM_STEPS_KEY
task_name_from_obs = _policy.task_name_from_obs

CheckpointKey: TypeAlias = tuple[str, str]


class LazyTaskRoutedEnsemblePolicy(BasePolicy):
    """Task-routed ensemble that loads at most ``max_loaded`` checkpoints on GPU at once.

    Use when several unique ``(config, dir)`` checkpoints do not fit in VRAM together. The WebSocket server
    stays up; on each ``infer`` the needed weights are loaded from disk if absent, and least-recently-used
    entries are evicted (``del`` + ``gc.collect()`` + ``jax.clear_caches()``) when the cache is full.

    For more aggressive GPU release between swaps, consider ``XLA_PYTHON_CLIENT_PREALLOCATE=false``.
    """

    def __init__(
        self,
        task_to_key: dict[str, CheckpointKey],
        default_key: CheckpointKey,
        tts_by_task: dict[str, bool],
        default_tts: bool,
        load_policy_for_key: Callable[[CheckpointKey], Policy],
        *,
        key_to_static_metadata: dict[CheckpointKey, dict[str, Any]],
        log_routing: bool = True,
        max_loaded: int = 1,
        tricks_by_task: dict[str, list[str]] | None = None,
        tts_v2_by_task: dict[str, bool] | None = None,
        tts_v2_num_steps_by_task: dict[str, int | None] | None = None,
    ):
        if max_loaded < 1:
            raise ValueError("max_loaded must be >= 1")
        if not task_to_key:
            raise ValueError("task_to_key must be non-empty")
        if set(tts_by_task) != set(task_to_key):
            raise ValueError("tts_by_task keys must match task_to_key keys exactly")

        self._task_to_key = dict(task_to_key)
        self._default_key = default_key
        self._tts_by_task = dict(tts_by_task)
        self._default_tts = default_tts
        self._load_policy_for_key = load_policy_for_key
        self._key_to_static_metadata = dict(key_to_static_metadata)
        self._log_routing = log_routing
        self._max_loaded = max_loaded
        self._tricks_by_task: dict[str, list[str]] = dict(tricks_by_task or {})
        self._tts_v2_by_task: dict[str, bool] = dict(tts_v2_by_task or {})
        self._tts_v2_num_steps_by_task: dict[str, int | None] = dict(tts_v2_num_steps_by_task or {})

        self._lock = threading.Lock()
        self._loaded: OrderedDict[CheckpointKey, Policy] = OrderedDict()
        self._last_task: str | None = None  # for task-switch banner

        per_task_policy_metadata = {
            t: dict(self._key_to_static_metadata.get(k, {})) for t, k in self._task_to_key.items()
        }
        self._metadata: dict[str, Any] = {
            "ensemble": "task_routed_lazy",
            "lazy_load": True,
            "max_loaded_checkpoints": max_loaded,
            "routed_tasks": sorted(self._task_to_key.keys()),
            "per_task_tts": dict(self._tts_by_task),
            "per_task_tts_v2": dict(self._tts_v2_by_task),
            "per_task_tts_v2_num_steps": {k: v for k, v in self._tts_v2_num_steps_by_task.items() if v is not None},
            "per_task_tricks": dict(self._tricks_by_task),
            "default_tts": self._default_tts,
            "default_policy_metadata": dict(self._key_to_static_metadata.get(default_key, {})),
            "per_task_policy_metadata": per_task_policy_metadata,
        }

    def _ensure_loaded(self, key: CheckpointKey) -> Policy:
        with self._lock:
            if key in self._loaded:
                self._loaded.move_to_end(key)
                return self._loaded[key]

            while len(self._loaded) >= self._max_loaded:
                evict_key, evict_pol = self._loaded.popitem(last=False)
                logging.info(
                    "LazyTaskRoutedEnsemblePolicy: evicting checkpoint config=%r dir=%r (max_loaded=%d)",
                    evict_key[0],
                    evict_key[1],
                    self._max_loaded,
                )
                del evict_pol
                gc.collect()
                jax.clear_caches()

            logging.info(
                "LazyTaskRoutedEnsemblePolicy: loading checkpoint config=%r dir=%r",
                key[0],
                key[1],
            )
            pol = self._load_policy_for_key(key)
            self._loaded[key] = pol
            self._loaded.move_to_end(key)
            return pol

    # Trick names that get injected into obs as boolean flags so outer wrappers
    # can decide per-request whether to activate.
    _OBS_INJECTABLE_TRICKS = frozenset({
        "sorting_packages_prompt",
        "sorting_packages_continuous_prompt",
        "sorting_packages_continuous_prompt_v2",
        "rule_based_sorting_correction",
        "clean_the_desktop_correction",
        "dynamic_tts_sorting",
    })

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        task = task_name_from_obs(obs)
        if task is not None and task in self._task_to_key:
            key = self._task_to_key[task]
            use_tts = self._tts_by_task[task]
            tricks = self._tricks_by_task.get(task, [])
            routed = True
        else:
            key = self._default_key
            use_tts = self._default_tts
            tricks = []
            routed = False

        use_tts_v2 = self._tts_v2_by_task.get(task, False) if task else False
        tts_v2_num_steps = self._tts_v2_num_steps_by_task.get(task) if use_tts_v2 else None

        # Print a prominent banner when the task changes (new episode / task switch).
        if task != self._last_task:
            prev = self._last_task
            self._last_task = task
            ckpt_dir = key[1] if key else "?"
            print(flush=True)
            print("=" * 72, flush=True)
            print(f"[Ensemble] TASK SWITCH: {prev!r} -> {task!r}", flush=True)
            print(f"  checkpoint : {ckpt_dir}", flush=True)
            print(f"  config     : {key[0] if key else '?'}", flush=True)
            print(f"  tts        : {use_tts}", flush=True)
            print(f"  tts_v2     : {use_tts_v2} (num_steps={tts_v2_num_steps})", flush=True)
            print(f"  tricks     : {tricks}", flush=True)
            print(f"  routed     : {routed}", flush=True)
            print("=" * 72, flush=True)

        policy = self._ensure_loaded(key)

        if self._log_routing:
            if routed:
                logging.info("LazyTaskRoutedEnsemblePolicy: task_name=%r -> routed tts=%r tts_v2=%r tricks=%s", task, use_tts, use_tts_v2, tricks)
            else:
                logging.info("LazyTaskRoutedEnsemblePolicy: task_name=%r -> default tts=%r", task, use_tts)

        if not isinstance(obs, dict):
            logging.warning(
                "LazyTaskRoutedEnsemblePolicy: obs is not a dict; cannot set %r.",
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


def static_policy_metadata_for_checkpoint(config_name: str, checkpoint_dir: str) -> dict[str, Any]:
    """Lightweight policy metadata from train config only (no weights). For lazy ensemble client handshake."""
    cfg = _train_config.get_config(config_name)
    meta = dict(cfg.policy_metadata or {})
    meta.setdefault("train_config_name", config_name)
    meta["checkpoint_dir"] = str(pathlib.Path(checkpoint_dir).expanduser())
    return meta
