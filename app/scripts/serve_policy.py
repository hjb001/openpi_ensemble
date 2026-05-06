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

# All recognised trick names.
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

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    # If set, overrides top-level ``--tts`` for this checkpoint only (ensemble routes / default / single checkpoint).
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
class TaskRoutedEnsemble:
    """Load several checkpoints; each step uses the policy whose `task_name` key matches `obs['task_name']`.

    If `task_name` is missing or unknown, `default` is used. Each sub-policy still runs its own transforms and
    `post_process` (e.g. waist handling) inside `Policy.infer`.

    `routes_json` must be a JSON array of objects with keys `task_name`, `config`, and `dir` (same fields as
    Checkpoint). Optional boolean `tts` per row overrides global ``--tts`` for that subtask's **inference** only;
    weights are still deduped by `(config, dir)` (TTS is applied per step via ``policy_infer_tts``).
    """

    default: Checkpoint
    routes_json: pathlib.Path


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Only expose / execute the first N steps of each predicted action chunk (receding horizon).
    partial_execution_steps: int | None = None

    # Test-time scaling：更长 num_steps（或 max_decoding_steps）+ 多次采样对动作做均值自洽。
    tts: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default | TaskRoutedEnsemble = dataclasses.field(default_factory=Default)

    sorting_corrections: list[int] | None = None
    """Fallback: which rule-based sorting corrections (1-6) to enable. None = all.
    Per-task 'sorting_corrections' in routes.json takes priority over this.
    1: waist clamp (place_scan)  2: idle override (grab)
    3: arm-down nudge (place_scan)  4: release-retract (place_scan)
    5: force-rotate (grab_scan)  6: pre-turn lift (grab_scan)"""


def _resolved_tts(ckpt: Checkpoint, args: Args) -> bool:
    """Per-checkpoint override, else global ``Args.tts``."""
    return ckpt.tts if ckpt.tts is not None else args.tts


# Default checkpoints that should be used for each environment.
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
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
            partial_execution_steps=partial_execution_steps,
            tts=tts,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def _policy_from_checkpoint(
    ckpt: Checkpoint,
    args: Args,
) -> _policy.Policy:
    return _policy_config.create_trained_policy(
        _config.get_config(ckpt.config),
        ckpt.dir,
        default_prompt=args.default_prompt,
        tts=_resolved_tts(ckpt, args),
        partial_execution_steps=args.partial_execution_steps,
    )


def _ensemble_base_policy_from_checkpoint(ckpt: Checkpoint, args: Args) -> _policy.Policy:
    """Single shared weights per checkpoint; per-task TTS via ``policy_infer_tts`` in ``Policy.infer``."""
    return _policy_config.create_trained_policy(
        _config.get_config(ckpt.config),
        ckpt.dir,
        default_prompt=args.default_prompt,
        tts=False,
        partial_execution_steps=args.partial_execution_steps,
    )


def create_policy(args: Args) -> _policy.Policy | _policy.TaskRoutedEnsemblePolicy:
    """Create a policy from the given arguments."""
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
        case TaskRoutedEnsemble():
            raw = json.loads(args.policy.routes_json.read_text())
            if not isinstance(raw, list) or not raw:
                raise ValueError("routes_json must be a non-empty JSON array of {task_name, config, dir}")
            # One Policy per unique (config, dir); TTS on/off does not duplicate weights.
            loaded: dict[tuple[str, str], _policy.Policy] = {}

            def _get_policy(ckpt: Checkpoint) -> _policy.Policy:
                key = _checkpoint_cache_key(ckpt.config, ckpt.dir)
                if key not in loaded:
                    loaded[key] = _ensemble_base_policy_from_checkpoint(ckpt, args)
                    logging.info(
                        "Loaded checkpoint into memory (unique key config=%r dir=%r)",
                        key[0],
                        key[1],
                    )
                return loaded[key]

            default_policy = _get_policy(args.policy.default)
            default_tts = _resolved_tts(args.policy.default, args)
            by_task: dict[str, _policy.Policy] = {}
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

                # --- Parse tricks list ---
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
                by_task[r.task_name] = _get_policy(Checkpoint(config=r.config, dir=r.dir))
                tts_by_task[r.task_name] = route_tts if route_tts is not None else args.tts
                tts_v2_by_task[r.task_name] = route_tts_v2
                tts_v2_num_steps_by_task[r.task_name] = route_tts_v2_num_steps
                tricks_by_task[r.task_name] = route_tricks
            logging.info(
                "Task-routed ensemble: %d unique checkpoint(s) in memory, %d task route(s)",
                len(loaded),
                len(by_task),
            )
            for tn, tr in tricks_by_task.items():
                sc_info = sorting_corrections_by_task.get(tn, "all")
                v2_info = f"tts_v2={tts_v2_by_task[tn]}"
                if tts_v2_by_task[tn] and tts_v2_num_steps_by_task.get(tn) is not None:
                    v2_info += f"(num_steps={tts_v2_num_steps_by_task[tn]})"
                logging.info("  task=%r  tricks=%s  tts=%s  %s  sorting_corrections=%s", tn, tr, tts_by_task[tn], v2_info, sc_info)
            ensemble = _policy.TaskRoutedEnsemblePolicy(
                by_task,
                default_policy,
                tts_by_task=tts_by_task,
                default_tts=default_tts,
                tricks_by_task=tricks_by_task,
                tts_v2_by_task=tts_v2_by_task,
                tts_v2_num_steps_by_task=tts_v2_num_steps_by_task,
            )
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

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # --- Wrap with trick-based wrappers (inside-out order) ---
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
    # wrappers actually see them.
    if tricks_by_task:
        policy = TrickFlagInjectorWrapper(policy, tricks_by_task, sorting_corrections_by_task)
        logging.info("TrickFlagInjectorWrapper active (outermost; %d task(s))", len(tricks_by_task))

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

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
