"""Policy WebSocket server — lazy task-routed ensemble variant.

Same protocol as ``serve_policy.py``; use subcommand ``policy:task-routed-ensemble-lazy`` to load at most
``max_loaded_checkpoints`` models on GPU (LRU). Other modes (default / single checkpoint) match ``serve_policy.py``.

routes.json now supports a ``tricks`` list per task. Recognised trick names:
  - ``tts``: test-time scaling (multi-sample self-consistency)
  - ``sorting_packages_continuous_prompt``: phase-based prompt rewriting for sorting_packages_continuous
  - ``rule_based_sorting_correction``: rule-based post-hoc action correction for sorting
  - ``clean_the_desktop_correction``: wrist-angle smoothing for clean_the_desktop
"""

import dataclasses
import enum
import json
import logging
import pathlib
import socket

import tyro


def _checkpoint_cache_key(config: str, dir: str) -> tuple[str, str]:
    """Normalize (config, dir) so duplicate routes / default share one loaded Policy."""
    d = pathlib.Path(dir).expanduser()
    try:
        resolved = d.resolve(strict=False)
    except OSError:
        resolved = d
    return (config.strip(), str(resolved))


from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import policy_lazy as _policy_lazy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
from policy_wrappers import (
    CleanDesktopCorrectionWrapper,
    DynamicTtsSortingWrapper,
    RuleBasedCleanDesktopCorrectionWrapper,
    RuleBasedSortingCorrectionWrapper,
    SortingPromptRequestWrapper,
    TrickFlagInjectorWrapper,
)

# All recognised trick names.  Order matters: wrappers are applied inside-out
# (first in list = innermost = runs first on each infer).
KNOWN_TRICKS = frozenset({
    "tts",
    "tts_v2",
    "sorting_packages_prompt",
    "sorting_packages_continuous_prompt",
    "sorting_packages_continuous_prompt_v2",
    "rule_based_sorting_correction",
    "clean_the_desktop_correction",
    "rule_based_clean_the_desktop_correction",
    "dynamic_tts_sorting",
})


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    VLABENCH = "vlabench"
    LIBEROPLUS = "liberoplus"
    G2SIM = "g2sim"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    dir: str
    tts: bool | None = None


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class TaskRoute:
    """When using task-routed ensemble: if `obs['task_name']` equals `task_name`, use this checkpoint."""

    task_name: str
    config: str
    dir: str


@dataclasses.dataclass
class TaskRoutedEnsembleLazy:
    """Task-routed ensemble with **lazy** checkpoint loading (see ``openpi.policies.policy_lazy``).

    At most ``max_loaded_checkpoints`` unique checkpoints stay on GPU; others are evicted LRU. Same ``routes.json``
    format as ``serve_policy.py`` ``policy:task-routed-ensemble``.

    Each route in ``routes.json`` may include a ``tricks`` list. Recognised values:
      ``tts``, ``sorting_packages_continuous_prompt``, ``rule_based_sorting_correction``,
      ``clean_the_desktop_correction``.
    """

    default: Checkpoint
    routes_json: pathlib.Path
    max_loaded_checkpoints: int = 1


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy_lazy script."""

    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    partial_execution_steps: int | None = None
    tts: bool = False
    policy: Checkpoint | Default | TaskRoutedEnsembleLazy = dataclasses.field(default_factory=Default)
    sorting_corrections: list[int] | None = None
    """Fallback: which rule-based sorting corrections (1-6) to enable. None = all.
    Per-task 'sorting_corrections' in routes.json takes priority over this.
    1: waist clamp (place_scan)  2: idle override (grab)
    3: arm-down nudge (place_scan)  4: release-retract (place_scan)
    5: force-rotate (grab_scan)  6: pre-turn lift (grab_scan)"""


def _resolved_tts(ckpt: Checkpoint, args: Args) -> bool:
    return ckpt.tts if ckpt.tts is not None else args.tts


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets-preview/checkpoints/pi05_may21_280k_v1",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="gs://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="acot_libero_action_cot_explicit_implicit_co_fusion",
        dir="./checkpoints/acot_libero_action_cot_explicit_implicit_co_fusion/exp_name/40000",
    ),
    EnvMode.LIBEROPLUS: Checkpoint(
        config="acot_libero_plus_action_cot_explicit_implicit_co_fusion",
        dir="./checkpoints/acot_libero_plus_action_cot_explicit_implicit_co_fusion/exp_name/100000",
    ),
    EnvMode.VLABENCH: Checkpoint(
        config="acot_vlabench_action_cot_explicit_implicit_co_fusion",
        dir="./checkpoints/acot_vlabench_action_cot_explicit_implicit_co_fusion/exp_name/60000",
    ),
    EnvMode.G2SIM: Checkpoint(
        config="acot_icra_simulation_challenge_reasoning_to_action",
        dir="./checkpoints/h3-continues-10000",
    ),
}


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    partial_execution_steps: int | None = None,
    tts: bool = False,
) -> _policy.Policy:
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            partial_execution_steps=partial_execution_steps,
            tts=tts,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def _policy_from_checkpoint(ckpt: Checkpoint, args: Args) -> _policy.Policy:
    return _policy_config.create_trained_policy(
        _config.get_config(ckpt.config),
        ckpt.dir,
        default_prompt=args.default_prompt,
        tts=_resolved_tts(ckpt, args),
        partial_execution_steps=args.partial_execution_steps,
    )


def _ensemble_base_policy_from_checkpoint(ckpt: Checkpoint, args: Args) -> _policy.Policy:
    return _policy_config.create_trained_policy(
        _config.get_config(ckpt.config),
        ckpt.dir,
        default_prompt=args.default_prompt,
        tts=False,
        partial_execution_steps=args.partial_execution_steps,
    )


def create_policy(args: Args) -> _policy.Policy | _policy_lazy.LazyTaskRoutedEnsemblePolicy:
    match args.policy:
        case Checkpoint():
            return _policy_from_checkpoint(args.policy, args)
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                partial_execution_steps=args.partial_execution_steps,
                tts=args.tts,
            )
        case TaskRoutedEnsembleLazy():
            raw = json.loads(args.policy.routes_json.read_text())
            if not isinstance(raw, list) or not raw:
                raise ValueError("routes_json must be a non-empty JSON array of {task_name, config, dir}")
            default_tts = _resolved_tts(args.policy.default, args)
            by_task: dict[str, Checkpoint] = {}
            tts_by_task: dict[str, bool] = {}
            tts_v2_by_task: dict[str, bool] = {}
            tts_v2_num_steps_by_task: dict[str, int | None] = {}
            tricks_by_task: dict[str, list[str]] = {}
            sorting_corrections_by_task: dict[str, list[int]] = {}
            for i, item in enumerate(raw):
                if not isinstance(item, dict):
                    raise ValueError(f"routes_json[{i}] must be an object")
                try:
                    r = TaskRoute(
                        task_name=str(item["task_name"]),
                        config=str(item["config"]),
                        dir=str(item["dir"]),
                    )
                except KeyError as e:
                    raise ValueError(f"routes_json[{i}] missing key {e}") from e

                # --- Parse tricks list (new) ---------------------------------
                route_tricks: list[str] = []
                if "tricks" in item:
                    tv = item["tricks"]
                    if not isinstance(tv, list):
                        raise ValueError(f"routes_json[{i}]['tricks'] must be a list, got {type(tv).__name__}")
                    for t in tv:
                        t_str = str(t).strip()
                        if t_str not in KNOWN_TRICKS:
                            raise ValueError(
                                f"routes_json[{i}]['tricks'] contains unknown trick {t_str!r}; "
                                f"known: {sorted(KNOWN_TRICKS)}"
                            )
                        route_tricks.append(t_str)

                # --- Parse sorting_corrections list (optional) -------------------
                if "sorting_corrections" in item:
                    sc = item["sorting_corrections"]
                    if not isinstance(sc, list):
                        raise ValueError(f"routes_json[{i}]['sorting_corrections'] must be a list, got {type(sc).__name__}")
                    sc_ints = [int(x) for x in sc]
                    for x in sc_ints:
                        if x not in RuleBasedSortingCorrectionWrapper.ALL_CORRECTIONS:
                            raise ValueError(
                                f"routes_json[{i}]['sorting_corrections'] contains invalid id {x}; "
                                f"valid: {sorted(RuleBasedSortingCorrectionWrapper.ALL_CORRECTIONS)}"
                            )
                    sorting_corrections_by_task[str(item["task_name"])] = sc_ints

                # Backwards compat: also accept standalone "tts" bool field.
                route_tts: bool | None
                if "tts" in item and "tricks" not in item:
                    tv2 = item["tts"]
                    if not isinstance(tv2, bool):
                        raise ValueError(f"routes_json[{i}]['tts'] must be a boolean, got {type(tv2).__name__}")
                    route_tts = tv2
                elif "tts" in route_tricks:
                    route_tts = True
                else:
                    route_tts = None

                # --- Parse tts_v2 ---
                route_tts_v2 = "tts_v2" in route_tricks
                route_tts_v2_num_steps: int | None = None
                if "tts_v2_num_steps" in item:
                    ns = item["tts_v2_num_steps"]
                    if not isinstance(ns, int) or ns < 1:
                        raise ValueError(f"routes_json[{i}]['tts_v2_num_steps'] must be a positive integer, got {ns!r}")
                    route_tts_v2_num_steps = ns

                if r.task_name in by_task:
                    logging.warning("Duplicate route for task_name=%r; using the last checkpoint", r.task_name)
                by_task[r.task_name] = Checkpoint(config=r.config, dir=r.dir)
                tts_by_task[r.task_name] = route_tts if route_tts is not None else args.tts
                tts_v2_by_task[r.task_name] = route_tts_v2
                tts_v2_num_steps_by_task[r.task_name] = route_tts_v2_num_steps
                tricks_by_task[r.task_name] = route_tricks

            unique_specs: dict[tuple[str, str], Checkpoint] = {}
            task_to_key: dict[str, tuple[str, str]] = {}
            for task_name, ckpt in by_task.items():
                key = _checkpoint_cache_key(ckpt.config, ckpt.dir)
                unique_specs.setdefault(key, ckpt)
                task_to_key[task_name] = key
            default_key = _checkpoint_cache_key(args.policy.default.config, args.policy.default.dir)
            unique_specs.setdefault(default_key, args.policy.default)

            max_loaded = max(1, int(args.policy.max_loaded_checkpoints))

            def _load_for_key(key: tuple[str, str]) -> _policy.Policy:
                return _ensemble_base_policy_from_checkpoint(unique_specs[key], args)

            key_to_static = {
                k: _policy_lazy.static_policy_metadata_for_checkpoint(c.config, c.dir) for k, c in unique_specs.items()
            }
            logging.info(
                "Task-routed ensemble (lazy): %d unique checkpoint(s) on disk, %d task route(s), "
                "max_loaded_checkpoints=%d",
                len(unique_specs),
                len(by_task),
                max_loaded,
            )
            for tn, tr in tricks_by_task.items():
                sc_info = sorting_corrections_by_task.get(tn, "all")
                v2_info = f"tts_v2={tts_v2_by_task[tn]}"
                if tts_v2_by_task[tn] and tts_v2_num_steps_by_task.get(tn) is not None:
                    v2_info += f"(num_steps={tts_v2_num_steps_by_task[tn]})"
                logging.info("  task=%r  tricks=%s  tts=%s  %s  sorting_corrections=%s", tn, tr, tts_by_task[tn], v2_info, sc_info)

            ensemble = _policy_lazy.LazyTaskRoutedEnsemblePolicy(
                task_to_key,
                default_key,
                tts_by_task=tts_by_task,
                default_tts=default_tts,
                load_policy_for_key=_load_for_key,
                key_to_static_metadata=key_to_static,
                max_loaded=max_loaded,
                tricks_by_task=tricks_by_task,
                tts_v2_by_task=tts_v2_by_task,
                tts_v2_num_steps_by_task=tts_v2_num_steps_by_task,
            )
            # Attach for main() to pass to TrickFlagInjectorWrapper.
            ensemble._sorting_corrections_by_task = sorting_corrections_by_task  # type: ignore[attr-defined]
            return ensemble


def _has_any_trick(policy, trick_name: str) -> bool:
    """Check if any task in the ensemble uses a given trick."""
    if not hasattr(policy, "_tricks_by_task"):
        return False
    return any(trick_name in tricks for tricks in policy._tricks_by_task.values())


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = dict(policy.metadata)

    # Grab tricks_by_task and sorting_corrections_by_task from the raw ensemble
    # BEFORE wrapping (outer wrappers won't expose them).
    tricks_by_task = dict(getattr(policy, "_tricks_by_task", {}) or {})
    sorting_corrections_by_task = dict(getattr(policy, "_sorting_corrections_by_task", {}) or {})

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # --- Wrap with trick-based wrappers (inside-out order) ---
    # Each wrapper checks per-request obs["<trick_name>"] to decide whether to apply.
    # TrickFlagInjectorWrapper (added last so it's outermost) populates those flags
    # from the matched task's tricks list so these wrappers can see them.
    def _any_task_has(trick: str) -> bool:
        return any(trick in trs for trs in tricks_by_task.values())

    if _any_task_has("dynamic_tts_sorting"):
        policy = DynamicTtsSortingWrapper(policy, default_enabled=False)
        logging.info("DynamicTtsSortingWrapper active (per-task via tricks; TTS off during place_scan)")

    if _any_task_has("sorting_packages_continuous_prompt") or _any_task_has("sorting_packages_continuous_prompt_v2") or _any_task_has("sorting_packages_prompt"):
        policy = SortingPromptRequestWrapper(policy, default_sorting_phase_prompt=False)
        logging.info("SortingPromptRequestWrapper active (per-task via tricks)")

    if _any_task_has("rule_based_sorting_correction"):
        corr_set = set(args.sorting_corrections) if args.sorting_corrections is not None else None
        policy = RuleBasedSortingCorrectionWrapper(policy, default_enabled=False, enabled_corrections=corr_set)
        logging.info(
            "RuleBasedSortingCorrectionWrapper active (per-task via tricks), corrections=%s",
            sorted(corr_set) if corr_set is not None else "all",
        )

    if _any_task_has("clean_the_desktop_correction"):
        policy = CleanDesktopCorrectionWrapper(policy, default_enabled=False)
        logging.info("CleanDesktopCorrectionWrapper active (per-task via tricks)")

    if _any_task_has("rule_based_clean_the_desktop_correction"):
        policy = RuleBasedCleanDesktopCorrectionWrapper(policy, default_enabled=False)
        logging.info("RuleBasedCleanDesktopCorrectionWrapper active (per-task via tricks)")

    # Outermost: inject trick flags into obs based on task_name so the above
    # wrappers actually see them (otherwise the ensemble would inject too late).
    if tricks_by_task:
        policy = TrickFlagInjectorWrapper(policy, tricks_by_task, sorting_corrections_by_task)
        logging.info("TrickFlagInjectorWrapper active (outermost; %d task(s))", len(tricks_by_task))

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (lazy script) (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
