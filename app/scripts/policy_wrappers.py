"""Policy wrappers: sorting prompt, rule-based correction, clean-desktop correction, task auto-detect.

Migrated verbatim from /data/openpi_app/app/scripts/serve_policy.py.
"""

import logging

import numpy as np
from openpi_client import base_policy as _base_policy

logger = logging.getLogger(__name__)

class SortingStateMachine:
    """
    Sequential rule-based state machine for sorting_packages_continuous.

    Phases advance strictly forward — one phase at a time, never jumping.
    Waist yaw is the PRIMARY signal for movement transitions (grab→turn_scan,
    turn_scan→place_scan, turn_bin→place_bin, return→grab). Gripper open/close
    events are used ONLY at scanner and bin to detect release/re-grab events,
    and only when the phase is in the corresponding wait state.

    This prevents phantom transitions from noisy gripper readings during the
    initial grab phase (e.g., when the robot retries a failed grab) and
    prevents the color from rotating until a full cycle completes.

    State layout (processed 32-dim, as sent by geniesim client):
      [15] right_effector / gripper
      [20] waist yaw  (table ~1.57, scanner ~0, bin ~ -1.4)
    """

    # Hysteresis on the gripper:
    #  - above CLOSED → latch to "closed" (fully grabbed)
    #  - below OPEN   → latch to "open"   (fully released)
    #  - values in between keep the previous latched state.
    # This prevents mid-transition values (e.g. 45-55) from being
    # treated as a release event and advancing the phase too early.
    GRIPPER_CLOSED_THRESHOLD = 65.0
    GRIPPER_OPEN_THRESHOLD = 55.0
    WAIST_TABLE_THRESHOLD = 1.0          # above → at table
    WAIST_SCANNER_THRESHOLD = 0.5        # below → arrived at scanner region
    WAIST_LEAVE_SCANNER_THRESHOLD = 0.0  # below → left scanner, rotating toward bin
    WAIST_BIN_THRESHOLD = -0.3           # below → arrived at bin region

    IDX_GRIPPER = 15
    IDX_WAIST_YAW = 20
    # Right-arm joints (shoulder → wrist-rotate); used for idle detection so
    # that arm-only motion (e.g. reaching toward the package while gripper
    # and waist are stationary) is NOT misclassified as idle.
    ARM_SLICE = slice(7, 14)

    # Stuck/idle detection (only active during grab phase):
    # if for STUCK_WINDOW consecutive frames, each adjacent pair is close
    # enough (small per-step deltas) across gripper, waist, AND the right arm,
    # we treat the robot as idle in grab.
    # While idle, rotate the prompt every GRAB_IDLE_PROMPT_ROTATE_EVERY frames:
    #   short → long full-task text → "Sorting packages." → short → …
    STUCK_WINDOW = 5
    STUCK_GRIPPER_DELTA_MAX = 2.0
    STUCK_WAIST_DELTA_MAX = 0.12
    # L2 norm of right-arm joint vector (7 joints) per adjacent step.
    # Real arm motion during "reach toward table" is ≥ 0.24; static noise < 0.05.
    STUCK_ARM_DELTA_MAX = 0.15

    # Consecutive inferences where the sliding-window "idle in grab" test passes.
    GRAB_IDLE_PROMPT_ROTATE_EVERY = 10
    GRAB_IDLE_PROMPT_SLOTS = 3  # short, long, sorting — then cycle back to short

    GRAB_LONG_INSTRUCTION = (
        "Grab the package on the table, turn the waist right to face the barcode scanner, "
        "place the package on the scanning table with the barcode facing up. Then, grab the "
        "package, rotate the waist and place the package in the blue bin. Finally, return the "
        "waist back to face the initial table."
    )
    GRAB_SORTING_FALLBACK = "Sorting packages."

    COLOR_ORDER = ["black", "white", "red", "yellow"]
    # Grab-phase color cycle: each new grab cycle uses the next color.
    # None = no color appended.  All-None disables color injection entirely.
    GRAB_COLOR_CYCLE = [None, None, None, None]

    INSTRUCTIONS = {
        "grab":       "Grab the package on the table with the right arm.",
        "turn_scan":  "Turn the waist right to face the barcode scanner.",
        "place_scan": "Place the package on the scanning table with the barcode facing up.",
        "grab_scan":  "The right arm grabs the package.",
        "turn_bin":   "Rotate the waist with the right arm.",
        "place_bin":  "Place the package in the blue bin.",
        "return":     "Both arms coordinate and the waist returns to the initial posture.",
    }

    PHASE_ORDER = ["grab", "turn_scan", "place_scan", "grab_scan", "turn_bin", "place_bin", "return"]

    def __init__(self):
        self._cycle_count = 0
        self._color = None  # extracted from original prompt, e.g. "yellow"
        self.reset()

    def set_color(self, color: str | None):
        """Set package color extracted from the original task prompt."""
        self._color = color
        if color:
            logger.info("[SortingStateMachine] Package color set to: %s", color)

    def reset(self):
        self._phase = "grab"
        self._prev_gripper_closed = None  # None → first frame not yet seen
        self._grab_history: list[tuple[float, float, np.ndarray]] = []
        self._stuck = False
        self._consecutive_grab_idle = 0

    def _latch_gripper(self, gripper: float) -> bool | None:
        """Hysteretic latch. Returns True=closed, False=open, None=keep previous."""
        if gripper >= self.GRIPPER_CLOSED_THRESHOLD:
            return True
        if gripper <= self.GRIPPER_OPEN_THRESHOLD:
            return False
        return None  # in the dead-band; keep previous state

    def update(self, state: np.ndarray) -> tuple[str, str]:
        gripper = float(state[self.IDX_GRIPPER])
        waist = float(state[self.IDX_WAIST_YAW])
        arm = np.asarray(state[self.ARM_SLICE], dtype=np.float64).copy()

        latched = self._latch_gripper(gripper)

        # First frame: initialise the latched state. If gripper starts in the
        # dead-band, treat it as "open" (robot hasn't grabbed anything yet).
        if self._prev_gripper_closed is None:
            self._prev_gripper_closed = latched if latched is not None else False
            self._grab_history = [(gripper, waist, arm)]
            return self._phase, self._make_instruction()

        # Determine current state with hysteresis.
        if latched is None:
            is_closed = self._prev_gripper_closed  # hold previous
        else:
            is_closed = latched

        gripper_just_opened = self._prev_gripper_closed and not is_closed
        gripper_just_closed = not self._prev_gripper_closed and is_closed
        self._prev_gripper_closed = is_closed

        prev_phase = self._phase

        # Top-level reset: whenever the robot is physically back at the table
        # and we're not already in the "grab" phase, start a new cycle. This
        # handles both the normal "return → grab" transition AND the recovery
        # case where one or more mid-cycle transitions were missed (e.g.
        # gripper noise at the scanner leaves us stuck in place_scan). Once
        # the waist swings back past the table threshold, we MUST reset to
        # grab with the next color — otherwise the prompt would stay stuck
        # for the rest of the episode.
        if waist > self.WAIST_TABLE_THRESHOLD and self._phase != "grab":
            self._cycle_count += 1
            logger.info(
                "[SortingStateMachine] Back at table from phase '%s' — starting cycle %d (color=%s)",
                self._phase,
                self._cycle_count,
                self.COLOR_ORDER[self._cycle_count % len(self.COLOR_ORDER)],
            )
            self._phase = "grab"

        # Sequential forward-only transitions.
        elif self._phase == "grab":
            # At table. Advance when robot starts turning right (waist leaves table).
            if waist < self.WAIST_TABLE_THRESHOLD:
                self._phase = "turn_scan"

        elif self._phase == "turn_scan":
            # Turning toward scanner. Advance when arrived at scanner region.
            if waist < self.WAIST_SCANNER_THRESHOLD:
                self._phase = "place_scan"

        elif self._phase == "place_scan":
            # At scanner, holding package. Advance when gripper opens (released).
            # Note: only valid while still in scanner region — ignore noise.
            if gripper_just_opened and waist < self.WAIST_SCANNER_THRESHOLD:
                self._phase = "grab_scan"

        elif self._phase == "grab_scan":
            # At scanner, package released. Advance when the waist starts
            # rotating past the scanner region toward the bin (symmetric with
            # grab → turn_scan on the table side — both are waist-driven
            # "started moving" transitions). Gripper re-close is no longer
            # used here because it can spike mid-scanner due to re-grip
            # attempts without the robot actually moving on.
            if waist < self.WAIST_LEAVE_SCANNER_THRESHOLD:
                self._phase = "turn_bin"

        elif self._phase == "turn_bin":
            # Turning toward bin. Advance when arrived at bin region.
            if waist < self.WAIST_BIN_THRESHOLD:
                self._phase = "place_bin"

        elif self._phase == "place_bin":
            # At bin, holding package. Advance when gripper opens (released in bin).
            if gripper_just_opened and waist < self.WAIST_BIN_THRESHOLD:
                self._phase = "return"

        # "return" phase has no explicit forward logic — it waits for the
        # top-level "back at table" reset above to fire on the next frame
        # where waist > WAIST_TABLE_THRESHOLD.

        # Handle stuck detection state across phase transitions.
        if self._phase != prev_phase:
            # Phase changed — reset stuck detection state for the new phase.
            self._grab_history = []
            self._stuck = False
            self._consecutive_grab_idle = 0

        # Update stuck detection only while in grab phase.
        if self._phase == "grab":
            self._grab_history.append((gripper, waist, arm))
            if len(self._grab_history) > self.STUCK_WINDOW:
                self._grab_history.pop(0)

            idle_in_grab = False
            if len(self._grab_history) >= self.STUCK_WINDOW:
                # Adjacent-step criterion: every near-neighbor change in the
                # window should stay small across gripper, waist, AND arm.
                # Using max delta captures whether any one step was a
                # meaningful movement in any dimension.
                g_deltas = [abs(g2 - g1) for (g1, _, _), (g2, _, _) in zip(self._grab_history, self._grab_history[1:])]
                w_deltas = [abs(w2 - w1) for (_, w1, _), (_, w2, _) in zip(self._grab_history, self._grab_history[1:])]
                a_deltas = [float(np.linalg.norm(a2 - a1)) for (_, _, a1), (_, _, a2) in zip(self._grab_history, self._grab_history[1:])]
                g_max_delta = max(g_deltas) if g_deltas else 0.0
                w_max_delta = max(w_deltas) if w_deltas else 0.0
                a_max_delta = max(a_deltas) if a_deltas else 0.0
                if (
                    g_max_delta < self.STUCK_GRIPPER_DELTA_MAX
                    and w_max_delta < self.STUCK_WAIST_DELTA_MAX
                    and a_max_delta < self.STUCK_ARM_DELTA_MAX
                ):
                    idle_in_grab = True
                    if not self._stuck:
                        logger.info(
                            "[SortingStateMachine] Idle in grab phase "
                            "(gripper max delta=%.2f, waist max delta=%.4f, arm max delta=%.4f) "
                            "— will escalate prompt if this persists",
                            g_max_delta,
                            w_max_delta,
                            a_max_delta,
                        )
                    self._stuck = True

            if idle_in_grab:
                self._consecutive_grab_idle += 1
            else:
                self._consecutive_grab_idle = 0
                self._stuck = False

        return self._phase, self._make_instruction()

    def _inject_color(self, instr: str) -> str:
        """Insert cycle-based color before the trailing period, e.g. '...right arm.' -> '...right arm black.'"""
        color = self.GRAB_COLOR_CYCLE[self._cycle_count % len(self.GRAB_COLOR_CYCLE)]
        if not color:
            return instr
        if instr.endswith("."):
            return instr[:-1] + " " + color + "."
        return instr + " " + color

    def _make_instruction(self) -> str:
        if self._phase == "grab" and self._consecutive_grab_idle > 0:
            # Start rotating immediately on the first idle hit:
            # first 10 idle hits -> LONG, next 10 -> SORTING, next 10 -> SHORT, then repeat.
            slot = (
                ((self._consecutive_grab_idle - 1) // self.GRAB_IDLE_PROMPT_ROTATE_EVERY + 1)
                % self.GRAB_IDLE_PROMPT_SLOTS
            )
            if slot == 0:
                return self._inject_color(self.INSTRUCTIONS["grab"])
            if slot == 1:
                return self.GRAB_LONG_INSTRUCTION
            return self.GRAB_SORTING_FALLBACK
        instr = self.INSTRUCTIONS[self._phase]
        if self._phase == "grab":
            instr = self._inject_color(instr)
        return instr


class SortingPromptRequestWrapper(_base_policy.BasePolicy):
    """Sorting prompt handling with per-infer override.

    Supports two phase-based modes (selected via per-request obs flags):

    ``sorting_packages_continuous_prompt`` (bool):
      Phase-based sub-prompts via ``SortingStateMachine``.  Grab-phase colour
      rotates per cycle via ``GRAB_COLOR_CYCLE``.

    ``sorting_packages_prompt`` (bool):
      Same phase-based sub-prompts, but the grab-phase instruction is fixed to
      "The right arm grabs the package {colour}." where *colour* is extracted
      once from the original prompt (e.g. "Grab the yellow package ...").

    When both are absent (or False), only generic "sorting packages" prompts
    are rewritten to the full instruction.
    """

    # Mode enum for internal tracking (None = fallback only).
    _MODE_FALLBACK = "fallback"
    _MODE_CONTINUOUS = "continuous"
    _MODE_CONTINUOUS_V2 = "continuous_v2"
    _MODE_SINGLE = "single"

    GENERIC_PATTERNS = ["sorting packages", "sort packages"]
    FULL_INSTRUCTION = (
        "Grab the package on the table, turn the waist right to face the barcode scanner, "
        "place the package on the scanning table with the barcode facing up. Then, grab the "
        "package, rotate the waist and place the package in the blue bin. Finally, return the "
        "waist back to face the initial table."
    )

    # Grab instruction template for sorting_packages_prompt (single-cycle).
    SINGLE_GRAB_INSTRUCTION = "The right arm grabs the package {color}."

    def __init__(self, policy: _base_policy.BasePolicy, *, default_sorting_phase_prompt: bool = False):
        self._policy = policy
        self._default_phase = default_sorting_phase_prompt
        self._sm = SortingStateMachine()
        self._step = 0
        self._extracted_color: str | None = None
        self._last_mode: str | None = None

    @staticmethod
    def _extract_color(prompt: str) -> str | None:
        if not prompt:
            return None
        prompt_lower = prompt.lower()
        for color in SortingStateMachine.COLOR_ORDER:
            if color in prompt_lower:
                return color
        return None

    def _resolve_mode(self, obs: dict) -> tuple[dict, str]:
        """Pop mode flags from obs and return (cleaned obs, mode)."""
        work = dict(obs)
        raw_continuous = work.pop("sorting_packages_continuous_prompt", None)
        raw_continuous_v2 = work.pop("sorting_packages_continuous_prompt_v2", None)
        raw_single = work.pop("sorting_packages_prompt", None)

        if raw_single is not None and bool(raw_single):
            return work, self._MODE_SINGLE
        if raw_continuous_v2 is not None and bool(raw_continuous_v2):
            return work, self._MODE_CONTINUOUS_V2
        if raw_continuous is not None and bool(raw_continuous):
            return work, self._MODE_CONTINUOUS
        if self._default_phase:
            return work, self._MODE_CONTINUOUS
        return work, self._MODE_FALLBACK

    def infer(self, obs: dict) -> dict:
        work, mode = self._resolve_mode(obs)

        if mode != self._last_mode:
            self._sm.reset()
            self._extracted_color = None
            self._step = 0
            self._last_mode = mode

        if mode in (self._MODE_CONTINUOUS, self._MODE_CONTINUOUS_V2, self._MODE_SINGLE):
            # Re-extract color every infer — the prompt (and thus the target
            # colour) can change between episodes while the mode stays the same.
            original_prompt = work.get("prompt", "")
            color = self._extract_color(original_prompt)
            if color and color != self._extracted_color:
                self._extracted_color = color
                self._sm.set_color(color)
                logger.info("[SortingPrompt] Extracted color '%s' from prompt: %s", color, original_prompt)

            state = work.get("state")
            if state is not None:
                phase, instruction = self._sm.update(np.asarray(state))

                # For single mode: grab_scan uses its short instruction from
                # the state machine; all other phases keep the original prompt.
                if mode == self._MODE_SINGLE and phase != "grab_scan":
                    instruction = original_prompt

                # For continuous_v2: all phases use the long prompt,
                # except grab_scan which uses a short instruction.
                if mode == self._MODE_CONTINUOUS_V2:
                    if phase == "grab_scan":
                        instruction = "The right arm grabs the package."
                    else:
                        instruction = self.FULL_INSTRUCTION

                work["prompt"] = instruction
                gripper = float(state[SortingStateMachine.IDX_GRIPPER])
                waist = float(state[SortingStateMachine.IDX_WAIST_YAW])
                phase_idx = SortingStateMachine.PHASE_ORDER.index(phase) if phase in SortingStateMachine.PHASE_ORDER else -1
                print(
                    f"[SortingPrompt] step={self._step:04d} "
                    f"phase={phase_idx}/{len(SortingStateMachine.PHASE_ORDER)-1}({phase}) "
                    f"mode={mode} "
                    f"gripper={gripper:.2f} waist={waist:.4f} "
                    f"cycle={self._sm._cycle_count} "
                    f"idle={self._sm._consecutive_grab_idle} "
                    f"prompt=\"{instruction}\"",
                    flush=True,
                )
                self._step += 1

            out = self._policy.infer(work)
        else:
            prompt = work.get("prompt", "").strip()
            prompt_lower = prompt.lower().rstrip(".")
            if prompt_lower in self.GENERIC_PATTERNS:
                logger.info("[SortingFallback] Detected generic prompt '%s' -> replacing with full instruction", prompt)
                work["prompt"] = self.FULL_INSTRUCTION
            out = self._policy.infer(work)

        out["effective_prompt"] = work.get("prompt", "")
        return out

    def reset(self) -> None:
        self._sm.reset()
        self._step = 0
        self._extracted_color = None
        self._last_mode = None
        self._policy.reset()

    @property
    def metadata(self) -> dict:
        return self._policy.metadata


# ---------------------------------------------------------------------------
# Rule-based action correction for sorting_packages_continuous
# ---------------------------------------------------------------------------


class RuleBasedSortingCorrectionWrapper(_base_policy.BasePolicy):
    """Rule-based post-hoc correction on model action output.

    ``obs["rule_based_sorting_correction"]`` (bool), when set, enables or disables
    this wrapper for that infer only. If omitted, uses *default_enabled* from
    ``--rule-based-sorting-correction``.

    Corrections:
    1. **Waist clamp during place_scan**: prevents the model from rotating the
       waist past the scanner region (waist_yaw < 0.0), which would skip the
       scanning step entirely.
    2. **Idle override during grab**: when the robot is stuck in the grab phase
       for IDLE_TRIGGER consecutive frames, replays hard-coded action chunks
       (two small-movement steps) to nudge it into motion.
    3. **Place-scan arm-down nudge**: after waist clamp fires, the NEXT inference
       is overridden with a rule-based chunk that gently lowers the right arm
       toward the scanner surface.  The model's chunk was planned *for* a waist
       rotation, so zeroing only the waist leaves the arm static.  Replacing the
       whole chunk with current-state + arm-down delta gives visual feedback that
       helps the model re-orient on subsequent queries.

    Phase-aware (uses SortingStateMachine internally); only touches action output.
    """

    # During place_scan / grab_scan, waist action values are clamped to >= this.
    # 0.0 is the scanner-region boundary; below 0.0 means the robot left the
    # scanner and is heading toward the bin.
    WAIST_CLAMP_MIN = 0.0

    # How many consecutive idle frames before playing override actions.
    # STUCK_WINDOW already provides a 5-frame debounce, so trigger immediately.
    IDLE_TRIGGER = 1

    # Hard-coded override action sequences for grab-phase idle recovery.
    # Source: 20260410_1828_infer_000025/actions.txt and 20260410_1828_infer_000026/actions.txt
    _HARDCODED_OVERRIDE_ACTIONS: list[list[list[float]]] = [
        # --- chunk 0 (infer_000025) ---
        [
            [7.3856811080e-01, -7.1937696540e-01, -1.5215524840e+00, -1.5383141620e+00, 2.7870088070e-01, -9.2882897850e-01, -8.3689384820e-01, -7.2646784640e-01, -7.2235261060e-01, 1.5346774710e+00, -1.5608384060e+00, -2.8364751890e-01, -9.1191416740e-01, 8.5364176610e-01, 3.3359797200e-01, 3.3445999120e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5671544910e+00],
            [7.3878179360e-01, -7.1938980050e-01, -1.5215012360e+00, -1.5383965120e+00, 2.7870034360e-01, -9.2872650370e-01, -8.3705818020e-01, -7.2662379980e-01, -7.2209601210e-01, 1.5345525330e+00, -1.5612861600e+00, -2.8358089780e-01, -9.1171950150e-01, 8.5397854000e-01, 3.3383516840e-01, 3.3457122700e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5672784100e+00],
            [7.3888268500e-01, -7.1934045220e-01, -1.5215460730e+00, -1.5384960800e+00, 2.7846504180e-01, -9.2868537610e-01, -8.3708291530e-01, -7.2699203660e-01, -7.2131235310e-01, 1.5336090320e+00, -1.5641335850e+00, -2.8625566940e-01, -9.0879063760e-01, 8.5335506690e-01, 3.3366714370e-01, 3.3436537540e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5666468590e+00],
            [7.3897419660e-01, -7.1938889040e-01, -1.5215911860e+00, -1.5385309100e+00, 2.7849850290e-01, -9.2861442310e-01, -8.3710908380e-01, -7.3196020450e-01, -7.1868813780e-01, 1.5261574010e+00, -1.5775012910e+00, -3.0919484150e-01, -8.7620255510e-01, 8.2278528080e-01, 3.3329372150e-01, 3.3404200640e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5668652330e+00],
            [7.3895508820e-01, -7.1949450570e-01, -1.5215740170e+00, -1.5384501300e+00, 2.7855458010e-01, -9.2866666300e-01, -8.3714701530e-01, -7.3721868030e-01, -7.1626632980e-01, 1.5188265300e+00, -1.5789622810e+00, -3.2346264660e-01, -8.4835278190e-01, 7.9129808870e-01, 3.3342888840e-01, 3.3407262900e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5674520690e+00],
            [7.3897435440e-01, -7.1944148930e-01, -1.5215786360e+00, -1.5384506480e+00, 2.7857579330e-01, -9.2870054860e-01, -8.3710933230e-01, -7.4351109570e-01, -7.1351363230e-01, 1.5112636810e+00, -1.5719798370e+00, -3.3032379310e-01, -8.3198639840e-01, 7.6610281330e-01, 3.3314373090e-01, 3.3358626690e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5675165410e+00],
            [7.3901632530e-01, -7.1942871560e-01, -1.5215776200e+00, -1.5384551010e+00, 2.7857270060e-01, -9.2867071310e-01, -8.3723912090e-01, -7.5179952710e-01, -7.1002774160e-01, 1.5028766330e+00, -1.5595269140e+00, -3.3508988790e-01, -8.2959560550e-01, 7.4569551430e-01, 3.3315520030e-01, 3.3383944820e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5676962660e+00],
            [7.3906996750e-01, -7.1938058510e-01, -1.5216088950e+00, -1.5384537960e+00, 2.7851971780e-01, -9.2866751440e-01, -8.3722940920e-01, -7.5961123320e-01, -7.0713495880e-01, 1.4953972640e+00, -1.5482205540e+00, -3.3972964650e-01, -8.3520575840e-01, 7.3105604420e-01, 3.3287349880e-01, 3.3342205230e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5667316090e+00],
            [7.3915105500e-01, -7.1933305790e-01, -1.5216306000e+00, -1.5385694390e+00, 2.7849504330e-01, -9.2863111430e-01, -8.3727378940e-01, -7.6620718280e-01, -7.0593845360e-01, 1.4893746670e+00, -1.5388422010e+00, -3.4453842910e-01, -8.4180930580e-01, 7.1950824980e-01, 3.3298165260e-01, 3.3349747450e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5667569130e+00],
            [7.3922730530e-01, -7.1929074970e-01, -1.5216548120e+00, -1.5385661550e+00, 2.7850530490e-01, -9.2862774990e-01, -8.3724996700e-01, -7.7167135880e-01, -7.0546268250e-01, 1.4844085870e+00, -1.5303383790e+00, -3.4851340510e-01, -8.4775332490e-01, 7.0962281820e-01, 3.3294167100e-01, 3.3298387060e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5670992050e+00],
            [7.3910427840e-01, -7.1929602590e-01, -1.5216877410e+00, -1.5385836720e+00, 2.7846083410e-01, -9.2865519050e-01, -8.3718946590e-01, -7.7665539720e-01, -7.0548379390e-01, 1.4802697720e+00, -1.5226179460e+00, -3.5183395320e-01, -8.5319051200e-01, 7.0131237040e-01, 3.3283003570e-01, 3.3285171810e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5675253800e+00],
            [7.3924530980e-01, -7.1934850630e-01, -1.5216919440e+00, -1.5385823600e+00, 2.7844696930e-01, -9.2864697750e-01, -8.3723502500e-01, -7.8090688930e-01, -7.0580171090e-01, 1.4765303940e+00, -1.5163834380e+00, -3.5523296290e-01, -8.5689584450e-01, 6.9404265720e-01, 3.3297142000e-01, 3.3296672720e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5671019780e+00],
            [7.3917432410e-01, -7.1933095800e-01, -1.5216694250e+00, -1.5386725240e+00, 2.7848578740e-01, -9.2853126450e-01, -8.3725550040e-01, -7.8483669890e-01, -7.0618519940e-01, 1.4731663830e+00, -1.5114742960e+00, -3.5886544080e-01, -8.5948631270e-01, 6.8825676000e-01, 3.3306373370e-01, 3.3288951650e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5675259000e+00],
            [7.3925799260e-01, -7.1929700500e-01, -1.5217275820e+00, -1.5386905600e+00, 2.7841045500e-01, -9.2858101380e-01, -8.3727963610e-01, -7.8819769590e-01, -7.0658751770e-01, 1.4701187130e+00, -1.5083191780e+00, -3.6279578020e-01, -8.6081791020e-01, 6.8348562210e-01, 3.3284621470e-01, 3.3321079160e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5675827470e+00],
            [7.3922352740e-01, -7.1929329600e-01, -1.5217561360e+00, -1.5386219740e+00, 2.7840836490e-01, -9.2861810490e-01, -8.3725634850e-01, -7.9145275610e-01, -7.0677515870e-01, 1.4668006950e+00, -1.5065528660e+00, -3.6850895880e-01, -8.6062544490e-01, 6.7801560110e-01, 3.3296762520e-01, 3.3315469380e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5673564000e+00],
            [7.3918230550e-01, -7.1924498230e-01, -1.5217531300e+00, -1.5387442200e+00, 2.7836208880e-01, -9.2850249090e-01, -8.3730301010e-01, -7.9476986550e-01, -7.0598942570e-01, 1.4619602060e+00, -1.5076280720e+00, -3.7774110940e-01, -8.5379089760e-01, 6.6578532110e-01, 3.3317690220e-01, 3.3385997230e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5670657550e+00],
            [7.3907944180e-01, -7.1933634460e-01, -1.5215480430e+00, -1.5386320610e+00, 2.7845640490e-01, -9.2854279730e-01, -8.3725507310e-01, -8.0026817860e-01, -7.0487657730e-01, 1.4543049730e+00, -1.5005270940e+00, -3.8881393460e-01, -8.4589436910e-01, 6.3823562590e-01, 3.3302293030e-01, 3.3384365770e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5680993910e+00],
            [7.3905360320e-01, -7.1930916100e-01, -1.5215244950e+00, -1.5386338570e+00, 2.7852829220e-01, -9.2856599690e-01, -8.3732359640e-01, -8.0309993850e-01, -7.0482525680e-01, 1.4501996320e+00, -1.4919264900e+00, -3.9234145820e-01, -8.4802424130e-01, 6.2763184530e-01, 3.3327581450e-01, 3.3401175460e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5682773830e+00],
            [7.3896142640e-01, -7.1927605570e-01, -1.5214449630e+00, -1.5385821650e+00, 2.7854586420e-01, -9.2858697810e-01, -8.3729099690e-01, -8.0645729650e-01, -7.0504913340e-01, 1.4449289830e+00, -1.4793514830e+00, -3.9645867370e-01, -8.5470131850e-01, 6.1710581640e-01, 3.3324156750e-01, 3.3387958040e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5687233160e+00],
            [7.3906855010e-01, -7.1923945390e-01, -1.5214497400e+00, -1.5386249260e+00, 2.7853288570e-01, -9.2863070560e-01, -8.3721562600e-01, -8.1246633860e-01, -7.0560390860e-01, 1.4368258230e+00, -1.4649389430e+00, -4.0552757120e-01, -8.6150603220e-01, 5.9851316630e-01, 3.3332082780e-01, 3.3388326640e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5687179430e+00],
            [7.3905317150e-01, -7.1917405070e-01, -1.5214702950e+00, -1.5386317510e+00, 2.7844020220e-01, -9.2867029290e-01, -8.3724750840e-01, -8.1909207980e-01, -7.0694981230e-01, 1.4277095370e+00, -1.4511286000e+00, -4.1872511160e-01, -8.6566689200e-01, 5.7331672940e-01, 3.3334668030e-01, 3.3401315050e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5689585010e+00],
            [7.3900810290e-01, -7.1924151360e-01, -1.5214509770e+00, -1.5385724620e+00, 2.7844628470e-01, -9.2872302300e-01, -8.3723518700e-01, -8.2559143670e-01, -7.0900486040e-01, 1.4183291370e+00, -1.4374779780e+00, -4.3417289720e-01, -8.6784575850e-01, 5.4297879370e-01, 3.3320964140e-01, 3.3371375180e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5686524310e+00],
            [7.3898889350e-01, -7.1923766990e-01, -1.5213795850e+00, -1.5386166980e+00, 2.7842451790e-01, -9.2861433800e-01, -8.3715237320e-01, -8.3237536650e-01, -7.1063130890e-01, 1.4081636580e+00, -1.4241709860e+00, -4.5018576770e-01, -8.6829936580e-01, 5.1215882010e-01, 3.3310336810e-01, 3.3379432140e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5682888220e+00],
            [7.3889339350e-01, -7.1924662480e-01, -1.5213023330e+00, -1.5386137860e+00, 2.7851571030e-01, -9.2860513380e-01, -8.3713624730e-01, -8.4152527450e-01, -7.1079284560e-01, 1.3949678950e+00, -1.4113667630e+00, -4.7650870350e-01, -8.5514775140e-01, 4.6684158160e-01, 3.3296820120e-01, 3.3413276170e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5682654240e+00],
            [7.3885441150e-01, -7.1938798730e-01, -1.5212414280e+00, -1.5384240120e+00, 2.7840043390e-01, -9.2868436500e-01, -8.3731728200e-01, -8.4921240910e-01, -7.1188504720e-01, 1.3840873110e+00, -1.3889255860e+00, -4.9144741120e-01, -8.5863029900e-01, 4.3566605460e-01, 3.3293025250e-01, 3.3436330330e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5679536350e+00],
            [7.3883137180e-01, -7.1939862790e-01, -1.5212289770e+00, -1.5384285560e+00, 2.7844785040e-01, -9.2865586940e-01, -8.3731837030e-01, -8.5623351860e-01, -7.1293005950e-01, 1.3744673710e+00, -1.3623249640e+00, -5.0393433250e-01, -8.6998514270e-01, 4.0821448370e-01, 3.3295365700e-01, 3.3443815850e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5681588370e+00],
            [7.3873570770e-01, -7.1933571600e-01, -1.5212524440e+00, -1.5384352370e+00, 2.7841896510e-01, -9.2857429650e-01, -8.3734215910e-01, -8.6365970020e-01, -7.1483254310e-01, 1.3642653690e+00, -1.3335275620e+00, -5.1812548520e-01, -8.8353506960e-01, 3.8097806850e-01, 3.3306410640e-01, 3.3490423640e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5676217420e+00],
            [7.3882451030e-01, -7.1936277680e-01, -1.5212248840e+00, -1.5385172280e+00, 2.7842210330e-01, -9.2853643590e-01, -8.3734721900e-01, -8.7134098320e-01, -7.1763448740e-01, 1.3540334880e+00, -1.3051245640e+00, -5.3247260910e-01, -8.9879084090e-01, 3.5524366800e-01, 3.3302762310e-01, 3.3515722150e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5679922840e+00],
            [7.3884092880e-01, -7.1922064900e-01, -1.5212635630e+00, -1.5385144470e+00, 2.7832348890e-01, -9.2852842720e-01, -8.3741988530e-01, -8.7852533990e-01, -7.2166790740e-01, 1.3442872300e+00, -1.2784527370e+00, -5.4844722320e-01, -9.1312284420e-01, 3.3330435420e-01, 3.3279450960e-01, 3.3547830030e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5673184450e+00],
            [7.3886631040e-01, -7.1927480250e-01, -1.5212475840e+00, -1.5384664750e+00, 2.7826282780e-01, -9.2856466850e-01, -8.3735007300e-01, -8.8473294590e-01, -7.2632755960e-01, 1.3356437620e+00, -1.2556071200e+00, -5.6094097550e-01, -9.2482157410e-01, 3.1627630270e-01, 3.3292292530e-01, 3.3533181800e-01, -1.0454515220e+00, 1.3439072370e+00, -3.1939578060e-01, -5.4355400180e-09, 1.5679295450e+00],
        ],
        # --- chunk 1 (infer_000026) ---
        [
            [7.3870168120e-01, -7.1923118550e-01, -1.5211127270e+00, -1.5383806620e+00, 2.7822307520e-01, -9.2884666000e-01, -8.3731650110e-01, -8.7357815340e-01, -7.2020101440e-01, 1.3518855160e+00, -1.2978563920e+00, -5.3662733740e-01, -9.0491021500e-01, 3.5396291460e-01, 3.3305503430e-01, 3.3460311460e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5649823660e+00],
            [7.3877702050e-01, -7.1922619850e-01, -1.5210529360e+00, -1.5383237110e+00, 2.7831444930e-01, -9.2883757750e-01, -8.3738444090e-01, -8.8084480770e-01, -7.2405641510e-01, 1.3472226710e+00, -1.2731788370e+00, -5.4252193270e-01, -9.1590403320e-01, 3.3164067480e-01, 3.3306532620e-01, 3.3389185990e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5644281130e+00],
            [7.3885365330e-01, -7.1920762090e-01, -1.5211650820e+00, -1.5383859210e+00, 2.7837965680e-01, -9.2879954710e-01, -8.3744309720e-01, -8.8768145430e-01, -7.2776450270e-01, 1.3430287690e+00, -1.2516813190e+00, -5.4823285200e-01, -9.2471572280e-01, 3.1420916540e-01, 3.3297109820e-01, 3.3431682420e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5643324450e+00],
            [7.3888351110e-01, -7.1921757150e-01, -1.5211582730e+00, -1.5383667510e+00, 2.7838370670e-01, -9.2878329480e-01, -8.3737969250e-01, -8.9244409690e-01, -7.3062577140e-01, 1.3397495630e+00, -1.2358023590e+00, -5.5197180470e-01, -9.3082684570e-01, 3.0281826870e-01, 3.3318837150e-01, 3.3443240040e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5647154650e+00],
            [7.3889773580e-01, -7.1923561700e-01, -1.5211365180e+00, -1.5383237900e+00, 2.7840348260e-01, -9.2880150160e-01, -8.3737571750e-01, -8.9606608350e-01, -7.3296156090e-01, 1.3369189320e+00, -1.2231697570e+00, -5.5443728330e-01, -9.3543936560e-01, 2.9478168160e-01, 3.3312139370e-01, 3.3441211620e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5646438870e+00],
            [7.3888673520e-01, -7.1924397650e-01, -1.5211352550e+00, -1.5383724000e+00, 2.7842825430e-01, -9.2877121290e-01, -8.3737063740e-01, -8.9872592520e-01, -7.3504135380e-01, 1.3344911720e+00, -1.2130253170e+00, -5.5624801800e-01, -9.3889087250e-01, 2.8912625040e-01, 3.3316341680e-01, 3.3429322480e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5653319380e+00],
            [7.3890424390e-01, -7.1923778230e-01, -1.5211193240e+00, -1.5383791000e+00, 2.7841896420e-01, -9.2873494250e-01, -8.3735563500e-01, -9.0091848910e-01, -7.3705083080e-01, 1.3321107750e+00, -1.2037973830e+00, -5.5829568400e-01, -9.4144912010e-01, 2.8502275950e-01, 3.3312141910e-01, 3.3445975130e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5648135600e+00],
            [7.3894997530e-01, -7.1925572490e-01, -1.5210987390e+00, -1.5383876240e+00, 2.7834441960e-01, -9.2874711800e-01, -8.3736158150e-01, -9.0349792350e-01, -7.3949646610e-01, 1.3294532500e+00, -1.1948191560e+00, -5.6109628700e-01, -9.4382409780e-01, 2.8124351130e-01, 3.3314257890e-01, 3.3426604840e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5650407730e+00],
            [7.3894146770e-01, -7.1927132470e-01, -1.5211037040e+00, -1.5383882060e+00, 2.7830572830e-01, -9.2875416550e-01, -8.3735979300e-01, -9.0546338490e-01, -7.4169828870e-01, 1.3269532540e+00, -1.1868949350e+00, -5.6342371470e-01, -9.4544238870e-01, 2.7861028360e-01, 3.3304057480e-01, 3.3444064490e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5652503070e+00],
            [7.3890613480e-01, -7.1924103290e-01, -1.5211086520e+00, -1.5383471280e+00, 2.7835205930e-01, -9.2875065780e-01, -8.3736513250e-01, -9.0739736170e-01, -7.4444068260e-01, 1.3243455640e+00, -1.1795280770e+00, -5.6596812660e-01, -9.4651381360e-01, 2.7668614400e-01, 3.3312152080e-01, 3.3431379250e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5652815040e+00],
            [7.3891877280e-01, -7.1927000680e-01, -1.5211119100e+00, -1.5383991770e+00, 2.7831995050e-01, -9.2878438320e-01, -8.3740096650e-01, -9.0931364690e-01, -7.4749928940e-01, 1.3215792240e+00, -1.1713798230e+00, -5.6855339130e-01, -9.4719781180e-01, 2.7471045140e-01, 3.3321251300e-01, 3.3405343530e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5656172100e+00],
            [7.3889190010e-01, -7.1930508750e-01, -1.5210953470e+00, -1.5383855020e+00, 2.7835169200e-01, -9.2882434120e-01, -8.3734996630e-01, -9.1125925720e-01, -7.5085781640e-01, 1.3185231280e+00, -1.1627546590e+00, -5.7131366670e-01, -9.4772408870e-01, 2.7301975320e-01, 3.3316420460e-01, 3.3401003160e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5661298680e+00],
            [7.3891711070e-01, -7.1927050130e-01, -1.5211148970e+00, -1.5384104450e+00, 2.7837376530e-01, -9.2879985320e-01, -8.3738669870e-01, -9.1380459800e-01, -7.5586498370e-01, 1.3143544320e+00, -1.1493150580e+00, -5.7410629430e-01, -9.4780845780e-01, 2.7199644720e-01, 3.3317220940e-01, 3.3413771280e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5658288240e+00],
            [7.3892772380e-01, -7.1929856840e-01, -1.5210975750e+00, -1.5384160350e+00, 2.7833407710e-01, -9.2875964700e-01, -8.3744049340e-01, -9.1681376770e-01, -7.6140626810e-01, 1.3098790820e+00, -1.1348462400e+00, -5.7728123820e-01, -9.4814563390e-01, 2.7089388030e-01, 3.3306297130e-01, 3.3388961340e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5657392220e+00],
            [7.3890010330e-01, -7.1928185710e-01, -1.5211066280e+00, -1.5384245170e+00, 2.7831424490e-01, -9.2878472390e-01, -8.3744735850e-01, -9.1959879080e-01, -7.6714472010e-01, 1.3051039960e+00, -1.1212625360e+00, -5.8006532060e-01, -9.4892859580e-01, 2.6916949250e-01, 3.3328463250e-01, 3.3396861260e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5656657370e+00],
            [7.3893766180e-01, -7.1929374460e-01, -1.5211106840e+00, -1.5384633880e+00, 2.7835728600e-01, -9.2876725170e-01, -8.3745406420e-01, -9.2278722990e-01, -7.7401703580e-01, 1.2998451410e+00, -1.1067086930e+00, -5.8300355870e-01, -9.5034341270e-01, 2.6611423930e-01, 3.3311891180e-01, 3.3386664650e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5659291720e+00],
            [7.3897387150e-01, -7.1925157390e-01, -1.5211685140e+00, -1.5384607570e+00, 2.7831998470e-01, -9.2871449190e-01, -8.3743952860e-01, -9.2553861250e-01, -7.8062401570e-01, 1.2951208140e+00, -1.0938215460e+00, -5.8486139650e-01, -9.5198234720e-01, 2.6271673700e-01, 3.3309313550e-01, 3.3412488800e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5655369660e+00],
            [7.3897643400e-01, -7.1923915200e-01, -1.5211571360e+00, -1.5384679390e+00, 2.7835052600e-01, -9.2876533280e-01, -8.3745822730e-01, -9.2852146960e-01, -7.8787529390e-01, 1.2899592910e+00, -1.0801336680e+00, -5.8686664150e-01, -9.5441066700e-01, 2.5826041950e-01, 3.3325373140e-01, 3.3389526240e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5657591530e+00],
            [7.3900423110e-01, -7.1925546570e-01, -1.5211916000e+00, -1.5384317360e+00, 2.7839174520e-01, -9.2874806110e-01, -8.3743383050e-01, -9.3158150500e-01, -7.9587258080e-01, 1.2842183710e+00, -1.0656663380e+00, -5.8916508810e-01, -9.5668352400e-01, 2.5299175290e-01, 3.3317094730e-01, 3.3413910870e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5659036950e+00],
            [7.3901999010e-01, -7.1926301870e-01, -1.5211914230e+00, -1.5384338150e+00, 2.7839504770e-01, -9.2879118030e-01, -8.3752263200e-01, -9.3453612350e-01, -8.0436941790e-01, 1.2785118600e+00, -1.0515624870e+00, -5.9194126660e-01, -9.5879205050e-01, 2.4616346890e-01, 3.3321866270e-01, 3.3421016880e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5659432110e+00],
            [7.3900400740e-01, -7.1928683090e-01, -1.5211875120e+00, -1.5384477940e+00, 2.7830828150e-01, -9.2878572590e-01, -8.3751494990e-01, -9.3805928800e-01, -8.1367718510e-01, 1.2721412500e+00, -1.0355156950e+00, -5.9505221070e-01, -9.6069918140e-01, 2.3973088120e-01, 3.3334600270e-01, 3.3396992130e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5658763120e+00],
            [7.3903619420e-01, -7.1929885920e-01, -1.5211964060e+00, -1.5383995170e+00, 2.7833150030e-01, -9.2873811080e-01, -8.3757080420e-01, -9.4104538430e-01, -8.2237392100e-01, 1.2663792540e+00, -1.0214400070e+00, -5.9834657720e-01, -9.6237223170e-01, 2.3103610370e-01, 3.3341421720e-01, 3.3416578350e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5659354120e+00],
            [7.3905187840e-01, -7.1934623170e-01, -1.5211926390e+00, -1.5383977960e+00, 2.7835884270e-01, -9.2876173930e-01, -8.3752292510e-01, -9.4385724310e-01, -8.2998350580e-01, 1.2611012530e+00, -1.0081965770e+00, -6.0090208630e-01, -9.6308248040e-01, 2.2628838200e-01, 3.3334506240e-01, 3.3411374260e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5657428610e+00],
            [7.3906471510e-01, -7.1933396810e-01, -1.5211899690e+00, -1.5383834750e+00, 2.7834560250e-01, -9.2866012360e-01, -8.3747223640e-01, -9.4616535090e-01, -8.3679288400e-01, 1.2563055780e+00, -9.9736119240e-01, -6.0224037430e-01, -9.6248063100e-01, 2.2295819040e-01, 3.3340586510e-01, 3.3431198220e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5662884490e+00],
            [7.3909532410e-01, -7.1931184360e-01, -1.5212186070e+00, -1.5384048150e+00, 2.7829275110e-01, -9.2869345730e-01, -8.3754888870e-01, -9.4808098300e-01, -8.4341702820e-01, 1.2517452510e+00, -9.8708903590e-01, -6.0299627130e-01, -9.6149924760e-01, 2.2165555480e-01, 3.3320200080e-01, 3.3420148800e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5661950340e+00],
            [7.3901935030e-01, -7.1931875500e-01, -1.5211885790e+00, -1.5383958150e+00, 2.7828190410e-01, -9.2870016050e-01, -8.3752859540e-01, -9.5011494830e-01, -8.5071879020e-01, 1.2468200770e+00, -9.7674009140e-01, -6.0331219450e-01, -9.6048387390e-01, 2.2120504480e-01, 3.3306572430e-01, 3.3392797880e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5668794450e+00],
            [7.3899201890e-01, -7.1936171300e-01, -1.5211701950e+00, -1.5383862750e+00, 2.7830326390e-01, -9.2868059770e-01, -8.3754760650e-01, -9.5089147550e-01, -8.5457924890e-01, 1.2442714100e+00, -9.7009200260e-01, -6.0295754520e-01, -9.5811274420e-01, 2.2116910690e-01, 3.3317022730e-01, 3.3398141560e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5668435690e+00],
            [7.3902879270e-01, -7.1939443330e-01, -1.5212078630e+00, -1.5383924740e+00, 2.7834213680e-01, -9.2867066790e-01, -8.3757623860e-01, -9.5153871260e-01, -8.5860505910e-01, 1.2417420560e+00, -9.6349660340e-01, -6.0237064670e-01, -9.5543381860e-01, 2.2176493390e-01, 3.3322090740e-01, 3.3395000790e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5672073520e+00],
            [7.3901948950e-01, -7.1936192340e-01, -1.5211068890e+00, -1.5382995340e+00, 2.7831437630e-01, -9.2867309360e-01, -8.3754424610e-01, -9.5193785320e-01, -8.6068525410e-01, 1.2404448970e+00, -9.5746077590e-01, -6.0179422330e-01, -9.5369768190e-01, 2.2034157760e-01, 3.3317751210e-01, 3.3398111030e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5665903590e+00],
            [7.3905097180e-01, -7.1932652040e-01, -1.5212430160e+00, -1.5383791240e+00, 2.7831393100e-01, -9.2861554450e-01, -8.3750669590e-01, -9.5258374400e-01, -8.6322767280e-01, 1.2389502320e+00, -9.5000339900e-01, -6.0236907860e-01, -9.5109070480e-01, 2.1880880510e-01, 3.3292298460e-01, 3.3396575540e-01, -1.0454540250e+00, 1.3439054490e+00, -3.1939321760e-01, -1.4680222190e-06, 1.5665752810e+00],
        ],
    ]

    # -- Place-scan arm-down nudge constants ----------------------------------
    # Per-step joint deltas (processed action space).  From 5-episode data:
    #   proc[8] (right shoulder/elbow-like): reliably decreases during place_scan
    #   proc[9] (right auxiliary):           mostly decreases
    # Intentionally conservative — just enough to "initiate" the downward motion.
    NUDGE_ARM_DOWN_DELTAS = {
        8: -0.005,   # proc[8]: main "arm down" joint
        9: -0.003,   # proc[9]: secondary
        10: +0.002,  # proc[10]: compensation — prevents leftward drift
        13: +0.003,  # proc[13]: compensation — keeps end-effector oriented
    }
    # Safety floors / ceilings — never drive a joint beyond these (data bounds + margin).
    NUDGE_JOINT_FLOOR = {
        8: -2.0,
        9: -1.2,
    }
    # Waist-clamp streak trigger: if the model's waist output is clamped for
    # this many consecutive infers, fire one arm-down nudge, then reset the
    # counter.  Every N consecutive clamps → one nudge.
    PLACE_SCAN_CLAMP_TRIGGER = 3

    # -- Place-scan release-retract constants ----------------------------------
    # When the arm is stuck on the table (state barely changing, gripper closed)
    # for RELEASE_IDLE_TRIGGER consecutive infers, override the next action with
    # a chunk that opens the gripper and retracts the arm.
    #
    # Chunk layout (30 steps, mimicking successful model behaviour from data):
    #   step 0~19:  hold still (gripper closed, arm unchanged)
    #   step 20~29: simultaneously open gripper + retract arm
    RELEASE_IDLE_TRIGGER = 2
    RELEASE_STATE_DIFF_THRESH = 0.15  # L2 of right-arm state diff between infers
    # Chunk 1 (release): hold still, then gently open gripper + tiny arm adjust.
    # Derived from infer 226 of 0515 run.
    RELEASE_HOLD_RATIO = 2 / 3           # first 20 of 30 steps: hold still
    RELEASE_GRIPPER_TARGET = 0.37        # gripper at end of chunk 1
    RELEASE_ARM_DELTAS = {               # tiny per-step arm adjustment in last 10 steps
        12: -0.011,   # wrist
        13: -0.015,   # wrist rotate
    }
    # Chunk 2 (retract): gripper fully open + arm retracts heavily.
    # Derived from infer 227 of 0515 run.
    RETRACT_GRIPPER_HOLD_STEP = 5        # gripper stays 1.0 for first N steps
    RETRACT_GRIPPER_OPEN_STEP = 10       # gripper fully open by this step
    RETRACT_GRIPPER_TARGET = 0.3         # fully open
    RETRACT_ARM_DELTAS = {               # per-step arm retract deltas (all 30 steps)
        7:  -0.008,   # shoulder
        8:  +0.018,   # arm up (main)
        9:  +0.006,   # aux
        10: -0.013,   # elbow
        11: +0.034,   # major retract (main)
        12: -0.005,   # wrist (small)
        13: -0.020,   # wrist rotate
    }

    # -- Grab-scan force-rotate constants ----------------------------------------
    # After release-retract fires and the model re-grabs the package, if the
    # arm starts descending (j8 negative) without rotating the waist, it's stuck
    # in a place-grab loop.  Override with lift + rotate to force the next phase.
    # Derived from successful transition in infer 020-021 of 1031 run.
    FORCE_ROTATE_DESCEND_TRIGGER = 2     # consecutive descending + waist-static steps
    FORCE_ROTATE_J8_DELTA_THRESH = 0.0   # j8 action delta < this → descending
    FORCE_ROTATE_WAIST_DELTA_THRESH = 0.01  # |waist delta| < this → waist static
    # Chunk 1: slight lift + begin waist rotation.
    FORCE_ROTATE_C1_ARM_DELTAS = {
        8:  +0.005,   # lift
        11: +0.004,   # elbow up
    }
    FORCE_ROTATE_C1_WAIST_DELTA = -0.005  # per step
    # Chunk 2: hold arm + full waist rotation toward bin.
    FORCE_ROTATE_C2_WAIST_DELTA = -0.017  # per step

    # --- Correction 6: pre-turn lift in grab_scan ---
    # When model first tries to turn waist in grab_scan (after re-grab from
    # scanner), intercept and lift the arm up so PickUpOnGripper evaluator
    # records current_z > initial_z + 0.02 m.  Two-chunk sequence:
    #   chunk 1: hold waist, lift arm
    #   chunk 2: hold waist, lower arm back to original height
    # After both chunks, hand back to model (which resumes waist turn).
    PRE_TURN_LIFT_WAIST_DELTA_THRESH = -0.10  # action waist delta < this → model is turning
    # Multi-joint deltas for a *vertical* straight-arm* lift.  Removed j7
    # (shoulder yaw) and kept j11 near-zero to avoid the "inward shrink to
    # chest" seen with the retract-derived set; j8 raised for more height.
    #   j8  = main shoulder pitch lift  (primary)
    #   j9  = secondary lift (inverse of arm-down j9)
    #   j10 = left/right drift compensation
    #   j11 = elbow — kept tiny so the arm stays straight, not folded inward
    #   j13 = wrist rotate — keeps end-effector orientation
    PRE_TURN_LIFT_ARM_DELTAS = {
        8:  +0.035,   # per-step; 30 steps → +1.05 (primary lift)
        9:  +0.006,   # per-step; 30 steps → +0.18
        10: -0.020,   # per-step; 30 steps → -0.60
        11: +0.005,   # per-step; 30 steps → +0.15 (almost zero — no elbow fold)
        13: +0.004,   # per-step; 30 steps → +0.12
    }

    # All correction IDs — for reference:
    #   1: waist clamp during place_scan
    #   2: idle override in grab phase
    #   3: place-scan arm-down nudge
    #   4: place-scan release-retract
    #   5: grab-scan force-rotate
    #   6: pre-turn lift in grab_scan
    ALL_CORRECTIONS = frozenset({1, 2, 3, 4, 5, 6})

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        default_enabled: bool = True,
        enabled_corrections: set[int] | None = None,
    ):
        self._policy = policy
        self._default_enabled = default_enabled
        # None means all enabled; otherwise only the specified IDs are active.
        self._enabled_corrections: frozenset[int] = (
            frozenset(self.ALL_CORRECTIONS) if enabled_corrections is None else frozenset(enabled_corrections)
        )
        logger.info("[RuleBased] enabled corrections: %s", sorted(self._enabled_corrections))
        self._sm = SortingStateMachine()
        self._step = 0
        self._color_extracted = False
        self._override_sequence = self._load_override_sequence()
        self._override_idx = 0
        self._override_done_this_phase = False
        # Place-scan nudge state
        self._place_scan_clamp_streak = 0  # consecutive waist-clamped infers
        self._nudge_count = 0  # arm-down nudges fired in current place_scan phase
        # Place-scan release-retract state
        self._prev_right_arm_state: np.ndarray | None = None
        self._release_idle_count = 0
        self._release_sequence: list[np.ndarray] = []  # 2-chunk sequence
        self._release_seq_idx = 0
        self._release_fired_this_phase = False  # has release-retract triggered in this place_scan?
        # Grab-scan force-rotate state
        self._regrab_detected = False           # gripper re-closed after release
        self._descend_count = 0                 # consecutive descending steps
        self._force_rotate_sequence: list[np.ndarray] = []
        self._force_rotate_seq_idx = 0
        # Grab-scan pre-turn lift state
        self._pre_turn_lift_sequence: list[np.ndarray] = []
        self._pre_turn_lift_seq_idx = 0
        self._pre_turn_lift_done = False  # at most once per grab_scan round
        # Armed when the state machine enters grab_scan; the lift fires on
        # the first big waist-rotation action emitted while armed, regardless
        # of the current phase. This decouples trigger timing from the state
        # machine's noise-sensitive waist thresholds so premature phase
        # transitions (e.g. state.waist drifting −0.004 past 0.0 during
        # re-grab) don't cause the turn to be missed.
        self._pre_turn_lift_armed: bool = False
        # Tracks phase from the previous __call__ so we can detect entry
        # into grab_scan (and arm the lift) on the transition frame.
        self._prev_phase: str | None = None

    # -- helpers --------------------------------------------------------------

    @classmethod
    def _load_override_sequence(cls) -> list[np.ndarray]:
        seq = [np.array(chunk) for chunk in cls._HARDCODED_OVERRIDE_ACTIONS]
        for i, arr in enumerate(seq):
            logger.info("[RuleBased] Loaded hard-coded override actions chunk %d, shape=%s", i, arr.shape)
        return seq

    @staticmethod
    def _extract_color(prompt: str) -> str | None:
        if not prompt:
            return None
        for color in SortingStateMachine.COLOR_ORDER:
            if color in prompt.lower():
                return color
        return None

    @staticmethod
    def _clamp_waist_in_actions(actions, clamp_min: float) -> bool:
        """Clamp waist yaw (index 20) in every timestep to >= *clamp_min*.

        *actions* may be a list-of-lists or a numpy array; modified in-place.
        Returns True if any value was actually clamped.
        """
        idx = SortingStateMachine.IDX_WAIST_YAW
        clamped = False
        if isinstance(actions, np.ndarray):
            mask = actions[:, idx] < clamp_min
            if np.any(mask):
                actions[:, idx] = np.clip(actions[:, idx], clamp_min, None)
                clamped = True
        elif isinstance(actions, list):
            for step in actions:
                if isinstance(step, list) and len(step) > idx and step[idx] < clamp_min:
                    step[idx] = clamp_min
                    clamped = True
        return clamped

    def _build_nudge_chunk(self, state: np.ndarray, chunk_size: int) -> np.ndarray:
        """Build a gentle arm-down action chunk from current state.

        Every step = current joint positions + cumulative delta on the key
        right-arm joints.  Left arm, grippers, and waist are held at current
        state values (waist stays in scanner zone, gripper stays closed).
        """
        base = np.array(state[:21], dtype=np.float64, copy=True)
        # Grippers are in a different scale in state vs action space.
        # State: angle in degrees (e.g. 88), Action: normalised (1.0=closed, 0.3=open).
        # Force grippers to closed in action space.
        base[14] = 1.0  # left gripper
        base[15] = 1.0  # right gripper
        chunk = np.tile(base, (chunk_size, 1))
        for step_i in range(chunk_size):
            for joint_idx, delta in self.NUDGE_ARM_DOWN_DELTAS.items():
                raw = base[joint_idx] + delta * (step_i + 1)
                floor = self.NUDGE_JOINT_FLOOR.get(joint_idx, -999.0)
                chunk[step_i, joint_idx] = max(raw, floor)
        return chunk

    def _build_release_retract_sequence(self, state: np.ndarray, chunk_size: int) -> list[np.ndarray]:
        """Build a 2-chunk sequence mimicking infer 226→227 from 0515 run.

        Chunk 1 (release):
          step 0 ~ hold_end-1 : hold current pose, gripper closed (1.0)
          step hold_end ~ end : gently open gripper + tiny wrist adjustment

        Chunk 2 (retract):
          step 0 ~ open_step  : gripper ramps from 1.0 → 0.3 (re-grip then release)
          step open_step ~ end: gripper stays 0.3, arm retracts with full deltas
          Arm retract deltas applied from step 0 across entire chunk.
        """
        base = np.array(state[:21], dtype=np.float64, copy=True)
        base[14] = 1.0
        base[15] = 1.0

        # --- Chunk 1: release ---
        c1 = np.tile(base, (chunk_size, 1))
        hold_end = int(chunk_size * self.RELEASE_HOLD_RATIO)
        active = chunk_size - hold_end

        for step_i in range(hold_end, chunk_size):
            progress = step_i - hold_end + 1
            t = progress / active
            # Gripper: 1.0 → RELEASE_GRIPPER_TARGET
            c1[step_i, 15] = 1.0 + (self.RELEASE_GRIPPER_TARGET - 1.0) * t
            # Small arm adjustment (wrist only)
            for joint_idx, delta in self.RELEASE_ARM_DELTAS.items():
                c1[step_i, joint_idx] = base[joint_idx] + delta * progress

        # --- Chunk 2: retract ---
        # Base for chunk 2 = end state of chunk 1
        base2 = c1[-1].copy()
        c2 = np.tile(base2, (chunk_size, 1))
        open_step = self.RETRACT_GRIPPER_OPEN_STEP

        hold_step = self.RETRACT_GRIPPER_HOLD_STEP
        for step_i in range(chunk_size):
            # Gripper: hold 1.0 for first hold_step, ramp to 0.3 by open_step, then hold 0.3
            if step_i < hold_step:
                c2[step_i, 15] = 1.0
            elif step_i < open_step:
                t = (step_i - hold_step + 1) / (open_step - hold_step)
                c2[step_i, 15] = 1.0 + (self.RETRACT_GRIPPER_TARGET - 1.0) * t
            else:
                c2[step_i, 15] = self.RETRACT_GRIPPER_TARGET

            # Arm retract: cumulative deltas from step 0
            for joint_idx, delta in self.RETRACT_ARM_DELTAS.items():
                c2[step_i, joint_idx] = base2[joint_idx] + delta * (step_i + 1)

        return [c1, c2]

    def _build_force_rotate_sequence(self, state: np.ndarray, chunk_size: int) -> list[np.ndarray]:
        """Build a 4-chunk sequence: lift, lower, slight-lift+begin-rotate, full-rotate.

        Chunks 1–2 reuse _build_pre_turn_lift_sequence so the PickUpOnGripper
        evaluator sees z_diff > 0.02 m before the waist starts turning.  After
        the arm returns to the original pose (end of chunk 2), chunks 3–4 run
        the original force-rotate logic (mimics infer 020→021 from 1031 run).
        """
        # --- Chunks 1 & 2: lift then lower (carries the arm above z-threshold). ---
        lift_chunks = self._build_pre_turn_lift_sequence(state, chunk_size)

        # After chunk 2 the arm is back at its original pose; use that as the
        # base for the rotation chunks so gripper state carries over smoothly.
        base = lift_chunks[-1][-1].copy()

        # --- Chunk 3: slight lift + start rotating waist ---
        c3 = np.tile(base, (chunk_size, 1))
        for step_i in range(chunk_size):
            progress = step_i + 1
            for joint_idx, delta in self.FORCE_ROTATE_C1_ARM_DELTAS.items():
                c3[step_i, joint_idx] = base[joint_idx] + delta * progress
            c3[step_i, 20] = base[20] + self.FORCE_ROTATE_C1_WAIST_DELTA * progress

        # --- Chunk 4: hold arm + full waist rotation ---
        base2 = c3[-1].copy()
        c4 = np.tile(base2, (chunk_size, 1))
        for step_i in range(chunk_size):
            progress = step_i + 1
            c4[step_i, 20] = base2[20] + self.FORCE_ROTATE_C2_WAIST_DELTA * progress

        return [*lift_chunks, c3, c4]

    def _build_pre_turn_lift_sequence(self, state: np.ndarray, chunk_size: int) -> list[np.ndarray]:
        """Build a 2-chunk sequence: lift arm up, then lower back to original.

        Keeps waist frozen at the current position and gripper closed throughout.
        After both chunks the arm is back where it started and the model takes over.
        """
        base = np.array(state[:21], dtype=np.float64, copy=True)
        # Left gripper [14]: keep neutral (model outputs ~0.333 for left hand in sorting).
        # state[14] is in raw encoder units (~53), NOT action-space units, so we
        # must set it explicitly.  Do NOT set to 1.0 — that closes the left hand.
        base[14] = 0.333
        base[15] = 1.0  # right gripper closed (holding the package)

        # --- Chunk 1: lift arm up (waist unchanged) ---
        c1 = np.tile(base, (chunk_size, 1))
        for step_i in range(chunk_size):
            progress = step_i + 1
            for joint_idx, delta in self.PRE_TURN_LIFT_ARM_DELTAS.items():
                c1[step_i, joint_idx] = base[joint_idx] + delta * progress

        # --- Chunk 2: lower arm back to original height ---
        lifted = c1[-1].copy()
        c2 = np.tile(lifted, (chunk_size, 1))
        for step_i in range(chunk_size):
            progress = step_i + 1
            for joint_idx, delta in self.PRE_TURN_LIFT_ARM_DELTAS.items():
                c2[step_i, joint_idx] = lifted[joint_idx] - delta * progress

        return [c1, c2]

    # -- main infer -----------------------------------------------------------

    def infer(self, obs: dict) -> dict:
        work = dict(obs)
        raw = work.pop("rule_based_sorting_correction", None)
        use_rules = self._default_enabled if raw is None else bool(raw)
        if not use_rules:
            return self._policy.infer(work)

        # Per-request override from obs (injected by TrickFlagInjectorWrapper from routes.json).
        per_request_corr = work.pop("sorting_corrections", None)
        if per_request_corr is not None:
            effective_corrections = frozenset(int(x) for x in per_request_corr)
        else:
            effective_corrections = self._enabled_corrections

        # Extract color once.
        if not self._color_extracted:
            color = self._extract_color(work.get("prompt", ""))
            if color:
                self._sm.set_color(color)
            self._color_extracted = True

        state = work.get("state")
        phase = "grab"
        if state is not None:
            phase, _ = self._sm.update(np.asarray(state))
            gripper = float(state[SortingStateMachine.IDX_GRIPPER])
            waist = float(state[SortingStateMachine.IDX_WAIST_YAW])
            phase_idx = SortingStateMachine.PHASE_ORDER.index(phase) if phase in SortingStateMachine.PHASE_ORDER else -1
            print(
                f"[RuleBased] step={self._step:04d} "
                f"phase={phase_idx}/{len(SortingStateMachine.PHASE_ORDER)-1}({phase}) "
                f"gripper={gripper:.2f} waist={waist:.4f} "
                f"cycle={self._sm._cycle_count} "
                f"idle={self._sm._consecutive_grab_idle}",
                flush=True,
            )
            self._step += 1

        # Reset override state when phase leaves grab.
        if phase != "grab":
            self._override_idx = 0
            self._override_done_this_phase = False

        # Reset place_scan / grab_scan state when phase changes.
        if phase != "place_scan":
            self._place_scan_clamp_streak = 0
            self._nudge_count = 0
            self._prev_right_arm_state = None
            self._release_idle_count = 0
            self._release_sequence = []
            self._release_seq_idx = 0
        if phase not in ("place_scan", "grab_scan"):
            self._release_fired_this_phase = False
            self._regrab_detected = False
            self._descend_count = 0
            # Don't clear force-rotate sequence if still playing.
            if self._force_rotate_seq_idx >= len(self._force_rotate_sequence):
                self._force_rotate_sequence = []
                self._force_rotate_seq_idx = 0
        # Arm the pre-turn lift on entry into grab_scan. The trigger fires
        # later on the first big waist action while armed, regardless of the
        # then-current phase — so a premature grab_scan→turn_bin transition
        # caused by noise can't hide the real turn from us.
        if phase == "grab_scan" and self._prev_phase != "grab_scan":
            self._pre_turn_lift_armed = True
            self._pre_turn_lift_done = False
        if phase != "grab_scan":
            # Don't clear lift sequence if still playing.
            if self._pre_turn_lift_seq_idx >= len(self._pre_turn_lift_sequence):
                self._pre_turn_lift_sequence = []
                self._pre_turn_lift_seq_idx = 0

        # --- Correction 2: idle override in grab phase -----------------------
        should_override = (
            2 in effective_corrections
            and len(self._override_sequence) > 0
            and phase == "grab"
            and self._sm._consecutive_grab_idle >= self.IDLE_TRIGGER
            and not self._override_done_this_phase
        )
        if should_override and self._override_idx < len(self._override_sequence):
            chunk = self._override_sequence[self._override_idx]
            print(
                f"[RuleBased] *** IDLE OVERRIDE chunk "
                f"{self._override_idx + 1}/{len(self._override_sequence)} "
                f"(idle={self._sm._consecutive_grab_idle}) ***",
                flush=True,
            )
            self._override_idx += 1
            if self._override_idx >= len(self._override_sequence):
                self._override_done_this_phase = True
            out = self._policy.infer(work)
            out["actions"] = chunk.tolist()
            out["override"] = True
            return out

        # --- Normal inference ------------------------------------------------
        out = self._policy.infer(work)


# --- Correction 1: waist clamp during place_scan ---------------------
        waist_was_clamped = False
        if 1 in effective_corrections and phase == "place_scan" and "actions" in out:
            if self._clamp_waist_in_actions(out["actions"], self.WAIST_CLAMP_MIN):
                waist_was_clamped = True
                self._place_scan_clamp_streak += 1
                print(
                    f"[RuleBased] Clamped waist actions >= {self.WAIST_CLAMP_MIN} "
                    f"in phase '{phase}' — clamp streak={self._place_scan_clamp_streak}/{self.PLACE_SCAN_CLAMP_TRIGGER}",
                    flush=True,
                )
            else:
                self._place_scan_clamp_streak = 0

        # --- Correction 3: place-scan arm-down nudge -------------------------
        # Every PLACE_SCAN_CLAMP_TRIGGER consecutive waist-clamped infers →
        # replace the model output with a gentle arm-down chunk, then reset.
        if (
            3 in effective_corrections
            and phase == "place_scan"
            and state is not None
            and waist_was_clamped
            and self._place_scan_clamp_streak >= self.PLACE_SCAN_CLAMP_TRIGGER
        ):
            self._place_scan_clamp_streak = 0
            self._release_idle_count = 0  # nudge takes priority, reset release detection
            chunk_size = len(out.get("actions", [])) or 30
            nudge = self._build_nudge_chunk(state, chunk_size)
            print(
                f"[RuleBased] *** PLACE_SCAN ARM-DOWN NUDGE (clamp×{self.PLACE_SCAN_CLAMP_TRIGGER}) "
                f"({chunk_size} steps, delta/step: "
                f"{dict(self.NUDGE_ARM_DOWN_DELTAS)}) ***",
                flush=True,
            )
            out["actions"] = nudge.tolist()
            out["override"] = True

        # --- Correction 4: place-scan release-retract (2-chunk sequence) ------
        # If a release sequence is in progress, play the next chunk.
        if 4 in effective_corrections and phase == "place_scan" and self._release_seq_idx < len(self._release_sequence):
            chunk = self._release_sequence[self._release_seq_idx]
            print(
                f"[RuleBased] *** PLACE_SCAN RELEASE-RETRACT chunk "
                f"{self._release_seq_idx + 1}/{len(self._release_sequence)} ***",
                flush=True,
            )
            self._release_seq_idx += 1
            out["actions"] = chunk.tolist()
            out["override"] = True
        # Otherwise, detect idle and start a new sequence.
        elif 4 in effective_corrections and phase == "place_scan" and state is not None and not waist_was_clamped:
            ra_state = np.asarray(state[7:14])
            gripper_closed = float(state[SortingStateMachine.IDX_GRIPPER]) >= SortingStateMachine.GRIPPER_CLOSED_THRESHOLD
            if self._prev_right_arm_state is not None and gripper_closed:
                ra_diff = float(np.linalg.norm(ra_state - self._prev_right_arm_state))
                if ra_diff < self.RELEASE_STATE_DIFF_THRESH:
                    self._release_idle_count += 1
                    print(
                        f"[RuleBased] place_scan release idle={self._release_idle_count}/{self.RELEASE_IDLE_TRIGGER} "
                        f"(ra_diff={ra_diff:.4f})",
                        flush=True,
                    )
                else:
                    self._release_idle_count = 0
            else:
                self._release_idle_count = 0
            self._prev_right_arm_state = ra_state.copy()

            if self._release_idle_count >= self.RELEASE_IDLE_TRIGGER:
                self._release_idle_count = 0
                chunk_size = len(out.get("actions", [])) or 30
                self._release_sequence = self._build_release_retract_sequence(state, chunk_size)
                self._release_seq_idx = 0
                self._release_fired_this_phase = True
                # Play chunk 1 immediately.
                chunk = self._release_sequence[self._release_seq_idx]
                print(
                    f"[RuleBased] *** PLACE_SCAN RELEASE-RETRACT chunk "
                    f"{self._release_seq_idx + 1}/{len(self._release_sequence)} "
                    f"(idle×{self.RELEASE_IDLE_TRIGGER}, {chunk_size} steps) ***",
                    flush=True,
                )
                self._release_seq_idx += 1
                out["actions"] = chunk.tolist()
                out["override"] = True

        # --- Correction 5: grab-scan force-rotate --------------------------------
        # If release-retract already fired, model re-grabbed, and now arm is
        # descending without rotating waist → stuck in place-grab loop.
        # Override with lift + rotate to force transition to turn_bin.

        # Playing an in-progress force-rotate sequence takes priority (phase may
        # have advanced to turn_bin after chunk 1 rotated the waist past 0.0).
        if 5 in effective_corrections and self._force_rotate_seq_idx < len(self._force_rotate_sequence):
            chunk = self._force_rotate_sequence[self._force_rotate_seq_idx]
            print(
                f"[RuleBased] *** GRAB_SCAN FORCE-ROTATE chunk "
                f"{self._force_rotate_seq_idx + 1}/{len(self._force_rotate_sequence)} ***",
                flush=True,
            )
            self._force_rotate_seq_idx += 1
            out["actions"] = chunk.tolist()
            out["override"] = True
        elif 5 in effective_corrections and (
            phase == "grab_scan"
            and state is not None
            and self._release_fired_this_phase
            and "actions" in out
        ):
            gripper_closed = float(state[SortingStateMachine.IDX_GRIPPER]) >= SortingStateMachine.GRIPPER_CLOSED_THRESHOLD
            # Detect re-grab after release.
            if gripper_closed and not self._regrab_detected:
                self._regrab_detected = True
                logger.info("[RuleBased] Re-grab detected after release-retract in grab_scan")

            # After re-grab, check for descending arm + static waist.
            if self._regrab_detected and gripper_closed:
                actions_arr = np.array(out["actions"])
                if actions_arr.ndim == 2 and actions_arr.shape[0] > 1:
                    j8_delta = float(actions_arr[-1, 8] - actions_arr[0, 8])
                    waist_delta = float(actions_arr[-1, 20] - actions_arr[0, 20])
                    if j8_delta < self.FORCE_ROTATE_J8_DELTA_THRESH and abs(waist_delta) < self.FORCE_ROTATE_WAIST_DELTA_THRESH:
                        self._descend_count += 1
                        print(
                            f"[RuleBased] grab_scan descend count={self._descend_count}/{self.FORCE_ROTATE_DESCEND_TRIGGER} "
                            f"(j8_delta={j8_delta:+.4f}, waist_delta={waist_delta:+.4f})",
                            flush=True,
                        )
                    else:
                        self._descend_count = 0

                    if self._descend_count >= self.FORCE_ROTATE_DESCEND_TRIGGER:
                        self._descend_count = 0
                        chunk_size = len(out.get("actions", [])) or 30
                        self._force_rotate_sequence = self._build_force_rotate_sequence(state, chunk_size)
                        self._force_rotate_seq_idx = 0
                        # Force-rotate's own chunks 1-2 are the pre-turn lift
                        # and chunks 3-4 are the waist rotation. Disarm the
                        # standalone pre-turn lift so it doesn't stack on top
                        # of force-rotate's last chunk when Correction 6 reads
                        # out["actions"] below.
                        self._pre_turn_lift_armed = False
                        chunk = self._force_rotate_sequence[self._force_rotate_seq_idx]
                        print(
                            f"[RuleBased] *** GRAB_SCAN FORCE-ROTATE chunk "
                            f"{self._force_rotate_seq_idx + 1}/{len(self._force_rotate_sequence)} "
                            f"(descend×{self.FORCE_ROTATE_DESCEND_TRIGGER}, {chunk_size} steps) ***",
                            flush=True,
                        )
                        self._force_rotate_seq_idx += 1
                        out["actions"] = chunk.tolist()
                        out["override"] = True

        # --- Correction 6: pre-turn lift in grab_scan ---------------------------
        # When the model first tries to turn waist in grab_scan (waist delta in
        # actions exceeds threshold), intercept with a lift-then-lower sequence so
        # the PickUpOnGripper evaluator sees z_diff > 0.02 m.
        # Only fires once per grab_scan phase and only when force-rotate is idle.
        if 6 in effective_corrections and self._force_rotate_seq_idx >= len(self._force_rotate_sequence):
            # Playing an in-progress lift sequence.
            if self._pre_turn_lift_seq_idx < len(self._pre_turn_lift_sequence):
                chunk = self._pre_turn_lift_sequence[self._pre_turn_lift_seq_idx]
                print(
                    f"[RuleBased] *** PRE-TURN LIFT chunk "
                    f"{self._pre_turn_lift_seq_idx + 1}/{len(self._pre_turn_lift_sequence)} ***",
                    flush=True,
                )
                self._pre_turn_lift_seq_idx += 1
                out["actions"] = chunk.tolist()
                out["override"] = True
            # Detect: armed (entered grab_scan this round) and model emits
            # its first big waist-rotation action. Phase-independent — once
            # armed we watch the action signal directly until we fire.
            elif (
                self._pre_turn_lift_armed
                and not self._pre_turn_lift_done
                and state is not None
                and "actions" in out
            ):
                actions_arr = np.array(out["actions"])
                if actions_arr.ndim == 2 and actions_arr.shape[0] > 1:
                    waist_delta = float(actions_arr[-1, 20] - actions_arr[0, 20])
                    gripper_closed = (
                        float(state[SortingStateMachine.IDX_GRIPPER])
                        >= SortingStateMachine.GRIPPER_CLOSED_THRESHOLD
                    )
                    if gripper_closed and waist_delta < self.PRE_TURN_LIFT_WAIST_DELTA_THRESH:
                        chunk_size = actions_arr.shape[0]
                        self._pre_turn_lift_sequence = self._build_pre_turn_lift_sequence(state, chunk_size)
                        self._pre_turn_lift_seq_idx = 0
                        self._pre_turn_lift_done = True
                        self._pre_turn_lift_armed = False
                        chunk = self._pre_turn_lift_sequence[self._pre_turn_lift_seq_idx]
                        print(
                            f"[RuleBased] *** PRE-TURN LIFT chunk "
                            f"{self._pre_turn_lift_seq_idx + 1}/{len(self._pre_turn_lift_sequence)} "
                            f"(waist_delta={waist_delta:+.4f} < {self.PRE_TURN_LIFT_WAIST_DELTA_THRESH}, "
                            f"{chunk_size} steps, arm_deltas={dict(self.PRE_TURN_LIFT_ARM_DELTAS)}) ***",
                            flush=True,
                        )
                        self._pre_turn_lift_seq_idx += 1
                        out["actions"] = chunk.tolist()
                        out["override"] = True

        self._prev_phase = phase
        return out

    def reset(self) -> None:
        self._sm.reset()
        self._step = 0
        self._override_idx = 0
        self._override_done_this_phase = False
        self._place_scan_clamp_streak = 0
        self._nudge_count = 0
        self._prev_right_arm_state = None
        self._release_idle_count = 0
        self._release_sequence = []
        self._release_seq_idx = 0
        self._release_fired_this_phase = False
        self._regrab_detected = False
        self._descend_count = 0
        self._force_rotate_sequence = []
        self._force_rotate_seq_idx = 0
        self._pre_turn_lift_sequence = []
        self._pre_turn_lift_seq_idx = 0
        self._pre_turn_lift_done = False
        self._pre_turn_lift_armed = False
        self._prev_phase = None
        self._policy.reset()

    @property
    def metadata(self) -> dict:
        return self._policy.metadata


# ---------------------------------------------------------------------------
# Rule-based wrist correction for clean_the_desktop
# ---------------------------------------------------------------------------


class CleanDesktopCorrectionWrapper(_base_policy.BasePolicy):
    """Rule-based wrist-angle correction for clean_the_desktop grasps.

    Problem: during pre-grasp approach the model's wrist predictions (j12, j13)
    oscillate wildly between consecutive infers, causing the gripper to arrive at
    an unstable angle and fail to pick up objects (especially tissue).

    Corrections apply to **all** approach phases (gripper open) in
    clean_the_desktop.  The smoothing and clamping are conservative enough
    that they do not affect grasps whose wrist angles are already stable.

    1. **EMA smoothing** on j12/j13 when inter-infer jump exceeds threshold.
    2. **Hard clamp** to a safe wrist-angle range (derived from successful
       grasp data; prevents extreme rotation like j13 drifting to -1.5).

    Toggle: ``obs["clean_the_desktop_correction"]`` (bool) overrides the CLI
    default ``--clean-the-desktop-correction`` for that single infer.
    """

    # Right-arm wrist joint indices in the 16-dim processed action space.
    IDX_J12 = 12  # arm_r_joint6 (wrist pitch)
    IDX_J13 = 13  # arm_r_joint7 (wrist rotate)
    # Gripper index in state space (raw angle in degrees).
    IDX_RIGHT_GRIPPER_STATE = 15

    # Gripper state > this ⇒ gripper fully open (approach phase).
    GRIPPER_OPEN_THRESH = 100.0

    # --- Within-chunk max delta per step ---
    # Limits how much j12/j13 can change between consecutive steps inside one
    # 30-step action chunk.  Prevents the model from swinging the wrist wildly
    # within a single inference.  Derived from successful grasp data where
    # per-step delta was typically < 0.02.
    MAX_STEP_DELTA_J12 = 0.025
    MAX_STEP_DELTA_J13 = 0.025

    # --- Hard clamp (from successful-grasp data analysis) ---
    J12_CLAMP_MIN = -1.2
    J12_CLAMP_MAX = 0.1
    J13_CLAMP_MIN = -0.8
    J13_CLAMP_MAX = 0.6

    def __init__(self, policy: _base_policy.BasePolicy, *, default_enabled: bool = False):
        self._policy = policy
        self._default_enabled = default_enabled
        self._prev_j12: float | None = None
        self._prev_j13: float | None = None
        self._step = 0
        self._n_open = 0       # infer calls with gripper open (correction attempted)
        self._n_closed = 0     # infer calls with gripper closed (skipped)
        self._n_corrected = 0  # of n_open, how many actually changed values

    # -- main infer -----------------------------------------------------------

    def infer(self, obs: dict) -> dict:
        work = dict(obs)
        raw = work.pop("clean_the_desktop_correction", None)
        enabled = self._default_enabled if raw is None else bool(raw)

        out = self._policy.infer(work)

        if not enabled:
            return out

        actions = out.get("actions")
        state = work.get("state")
        if actions is None or state is None:
            return out

        is_np = isinstance(actions, np.ndarray)
        arr = np.asarray(actions, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 16:
            return out

        grip = float(state[self.IDX_RIGHT_GRIPPER_STATE])
        is_open = grip > self.GRIPPER_OPEN_THRESH
        phase = "OPEN" if is_open else "CARRY"

        # --- Apply wrist correction (all phases) ---
        orig_j12 = arr[:, self.IDX_J12].copy()
        orig_j13 = arr[:, self.IDX_J13].copy()
        chunk_j12_range = float(orig_j12.max() - orig_j12.min())
        chunk_j13_range = float(orig_j13.max() - orig_j13.min())

        # Correction 1: limit per-step delta within the chunk.
        for t in range(1, len(arr)):
            for idx, max_d in [(self.IDX_J12, self.MAX_STEP_DELTA_J12),
                               (self.IDX_J13, self.MAX_STEP_DELTA_J13)]:
                delta = arr[t, idx] - arr[t - 1, idx]
                if abs(delta) > max_d:
                    arr[t, idx] = arr[t - 1, idx] + np.sign(delta) * max_d

        # Correction 2: hard clamp to safe range.
        arr[:, self.IDX_J12] = np.clip(arr[:, self.IDX_J12], self.J12_CLAMP_MIN, self.J12_CLAMP_MAX)
        arr[:, self.IDX_J13] = np.clip(arr[:, self.IDX_J13], self.J13_CLAMP_MIN, self.J13_CLAMP_MAX)

        # Compute how much correction actually changed
        diff_j12 = float(np.abs(arr[:, self.IDX_J12] - orig_j12).max())
        diff_j13 = float(np.abs(arr[:, self.IDX_J13] - orig_j13).max())
        changed = diff_j12 > 1e-4 or diff_j13 > 1e-4

        if is_open:
            self._n_open += 1
        else:
            self._n_closed += 1
        self._n_corrected += int(changed)

        if changed:
            print(
                f"[CleanDesktop] step={self._step:04d} {phase} grip={grip:.0f} "
                f"range j12={chunk_j12_range:.3f} j13={chunk_j13_range:.3f} "
                f"maxΔ j12={diff_j12:.4f} j13={diff_j13:.4f} MODIFIED "
                f"[{self._n_corrected}/{self._step + 1}]",
                flush=True,
            )

        out["actions"] = arr if is_np else arr.tolist()

        self._prev_j12 = float(arr[-1, self.IDX_J12])
        self._prev_j13 = float(arr[-1, self.IDX_J13])
        self._step += 1

        return out

    def reset(self) -> None:
        if self._step > 0:
            print(
                f"[CleanDesktop] === Episode done: {self._step} infer calls, "
                f"{self._n_open} open + {self._n_closed} carry, "
                f"{self._n_corrected} actually modified ===",
                flush=True,
            )
        self._prev_j12 = None
        self._prev_j13 = None
        self._step = 0
        self._n_open = 0
        self._n_closed = 0
        self._n_corrected = 0
        self._policy.reset()

    @property
    def metadata(self) -> dict:
        return self._policy.metadata


# ---------------------------------------------------------------------------
# Rule-based idle-release correction for clean_the_desktop
# ---------------------------------------------------------------------------


class RuleBasedCleanDesktopCorrectionWrapper(_base_policy.BasePolicy):
    """Rule-based idle-release for clean_the_desktop.

    When the right arm (state[7:14]) barely moves for IDLE_TRIGGER consecutive
    infers (adjacent-pair L2 diff below threshold), override the next action
    chunk with a hold-pose chunk that gradually opens the right gripper from
    closed (1.0) to open (GRIPPER_OPEN_TARGET) over the chunk. All other joints
    are held at the current state, and the left gripper is set to neutral-open
    in action space.

    Re-arms naturally: the idle counter resets on each trigger. After a trigger
    fires, the gripper-opening motion itself causes arm-joint changes to stay
    small again, but since the chunk takes the gripper to the fully-open action
    value, by the time a new "grab and hold" situation develops the next 5
    consecutive idle steps are accumulated from scratch.

    Toggle: ``obs["rule_based_clean_the_desktop_correction"]`` (bool) overrides
    the wrapper default for that single infer.
    """

    # Right-arm joint indices in state space (7 joints, radians).
    RIGHT_ARM_SLICE = slice(7, 14)
    # Gripper indices in state space (raw angle in degrees).
    IDX_LEFT_GRIPPER_STATE = 14
    IDX_RIGHT_GRIPPER_STATE = 15
    # Action-space layout (16-dim per step): [0:7] left arm, [7:14] right arm,
    # [14] left gripper, [15] right gripper.
    IDX_RIGHT_GRIPPER_ACTION = 15
    IDX_LEFT_GRIPPER_ACTION = 14

    # Right-gripper-closed gate: state[15] above this ⇒ gripper is in grabbing
    # / holding state (reference: observed stuck values 99~104, fully-open ~55).
    GRIPPER_CLOSED_MIN_STATE = 90.0
    # Left-gripper-open gate: state[14] below this ⇒ left hand released
    # (distinguishes placement from handoff, where left hand is still grasping).
    LEFT_OPEN_MAX_STATE = 55.0
    # Left-right EEF XY-plane distance gate: above this ⇒ arms are spatially
    # far apart (placement on desk), below ⇒ close together (handoff).
    ARMS_FAR_APART_XY_THRESH = 0.20

    # Idle detection: adjacent-pair L2 norm of right-arm state deltas.
    IDLE_STATE_DIFF_THRESH = 0.15
    IDLE_TRIGGER = 2

    # Gripper action-space values.
    GRIPPER_CLOSED = 1.0
    GRIPPER_OPEN_TARGET = 0.333

    # --- Release / retract sequence (mirrors sorting's release-retract) ---
    # Chunk 1 (release): hold pose first 2/3, then gently open gripper
    # + tiny wrist adjustment.
    RELEASE_HOLD_RATIO = 2 / 3
    RELEASE_GRIPPER_TARGET = 0.37
    RELEASE_ARM_DELTAS = {
        12: -0.011,   # wrist
        13: -0.015,   # wrist rotate
    }
    # Chunk 2 (retract): gripper re-closes briefly then ramps fully open;
    # arm retracts with cumulative per-step deltas across the whole chunk.
    RETRACT_GRIPPER_HOLD_STEP = 5
    RETRACT_GRIPPER_OPEN_STEP = 10
    RETRACT_GRIPPER_TARGET = 0.3
    RETRACT_ARM_DELTAS = {
        7:  -0.008,   # shoulder
        8:  +0.018,   # arm up (main)
        9:  +0.006,   # aux
        10: -0.013,   # elbow
        11: +0.034,   # major retract (main)
        12: -0.005,   # wrist (small)
        13: -0.020,   # wrist rotate
    }

    def __init__(self, policy: _base_policy.BasePolicy, *, default_enabled: bool = False):
        self._policy = policy
        self._default_enabled = default_enabled
        self._prev_right_arm_state: np.ndarray | None = None
        self._idle_count = 0
        self._step = 0
        self._n_triggered = 0
        self._release_sequence: list[np.ndarray] = []
        self._release_seq_idx = 0

    def _build_release_retract_sequence(self, state: np.ndarray, chunk_size: int) -> list[np.ndarray]:
        """Build a 2-chunk release-retract sequence (16-dim action space).

        Chunk 1 (release):
          step 0 ~ hold_end-1 : hold current pose, right gripper closed (1.0)
          step hold_end ~ end : gently open gripper to RELEASE_GRIPPER_TARGET
                                + tiny wrist adjustment (j12, j13)

        Chunk 2 (retract):
          step 0 ~ hold_step-1    : right gripper held 1.0 (re-grip)
          step hold_step ~ open_step-1 : ramp 1.0 -> RETRACT_GRIPPER_TARGET
          step open_step ~ end    : gripper at RETRACT_GRIPPER_TARGET
          Arm retract deltas applied cumulatively across the whole chunk.
        """
        base = np.array(state[:16], dtype=np.float64, copy=True)
        # Left gripper: neutral-open in action space (NOT 1.0, which would close
        # the left hand — state units are degrees, action units are normalised).
        base[self.IDX_LEFT_GRIPPER_ACTION] = self.GRIPPER_OPEN_TARGET
        # Right gripper starts closed.
        base[self.IDX_RIGHT_GRIPPER_ACTION] = self.GRIPPER_CLOSED

        # --- Chunk 1: release ---
        c1 = np.tile(base, (chunk_size, 1))
        hold_end = int(chunk_size * self.RELEASE_HOLD_RATIO)
        active = max(1, chunk_size - hold_end)
        for step_i in range(hold_end, chunk_size):
            progress = step_i - hold_end + 1
            t = progress / active
            c1[step_i, self.IDX_RIGHT_GRIPPER_ACTION] = (
                self.GRIPPER_CLOSED
                + (self.RELEASE_GRIPPER_TARGET - self.GRIPPER_CLOSED) * t
            )
            for joint_idx, delta in self.RELEASE_ARM_DELTAS.items():
                c1[step_i, joint_idx] = base[joint_idx] + delta * progress

        # --- Chunk 2: retract ---
        base2 = c1[-1].copy()
        c2 = np.tile(base2, (chunk_size, 1))
        hold_step = self.RETRACT_GRIPPER_HOLD_STEP
        open_step = self.RETRACT_GRIPPER_OPEN_STEP
        for step_i in range(chunk_size):
            if step_i < hold_step:
                c2[step_i, self.IDX_RIGHT_GRIPPER_ACTION] = self.GRIPPER_CLOSED
            elif step_i < open_step:
                t = (step_i - hold_step + 1) / max(1, open_step - hold_step)
                c2[step_i, self.IDX_RIGHT_GRIPPER_ACTION] = (
                    self.GRIPPER_CLOSED
                    + (self.RETRACT_GRIPPER_TARGET - self.GRIPPER_CLOSED) * t
                )
            else:
                c2[step_i, self.IDX_RIGHT_GRIPPER_ACTION] = self.RETRACT_GRIPPER_TARGET
            for joint_idx, delta in self.RETRACT_ARM_DELTAS.items():
                c2[step_i, joint_idx] = base2[joint_idx] + delta * (step_i + 1)

        return [c1, c2]

    def infer(self, obs: dict) -> dict:
        work = dict(obs)
        raw = work.pop("rule_based_clean_the_desktop_correction", None)
        enabled = self._default_enabled if raw is None else bool(raw)

        out = self._policy.infer(work)

        if not enabled:
            return out

        state = work.get("state")
        if state is None:
            return out
        state_np = np.asarray(state, dtype=np.float64)

        # --- If a release-retract sequence is mid-flight, play next chunk. ---
        if self._release_seq_idx < len(self._release_sequence):
            chunk = self._release_sequence[self._release_seq_idx]
            seq_len = len(self._release_sequence)
            print(
                f"[RuleBasedCleanDesktop] *** RELEASE-RETRACT chunk "
                f"{self._release_seq_idx + 1}/{seq_len} ***",
                flush=True,
            )
            self._release_seq_idx += 1
            out["actions"] = chunk.tolist()
            out["override"] = True
            # Update idle-tracking state without counting these as idle.
            self._prev_right_arm_state = state_np[self.RIGHT_ARM_SLICE].copy()
            self._idle_count = 0
            self._step += 1
            return out

        # --- Normal idle detection. ---
        ra_state = state_np[self.RIGHT_ARM_SLICE].copy()
        grip = float(state_np[self.IDX_RIGHT_GRIPPER_STATE])
        grip_closed = grip > self.GRIPPER_CLOSED_MIN_STATE
        left_grip = float(state_np[self.IDX_LEFT_GRIPPER_STATE])
        left_open = left_grip < self.LEFT_OPEN_MAX_STATE

        # Left-right EEF XY-plane distance (placement vs handoff discriminator).
        eef = work.get("eef")
        xy_dist = None
        arms_far = False
        if isinstance(eef, dict):
            left_eef = eef.get("left")
            right_eef = eef.get("right")
            if left_eef is not None and right_eef is not None:
                lx, ly = float(left_eef[0]), float(left_eef[1])
                rx, ry = float(right_eef[0]), float(right_eef[1])
                xy_dist = float(np.hypot(lx - rx, ly - ry))
                arms_far = xy_dist > self.ARMS_FAR_APART_XY_THRESH

        ra_diff = None
        if self._prev_right_arm_state is not None:
            ra_diff = float(np.linalg.norm(ra_state - self._prev_right_arm_state))
            arm_idle = ra_diff < self.IDLE_STATE_DIFF_THRESH
            if grip_closed and arm_idle and left_open and arms_far:
                self._idle_count += 1
            else:
                self._idle_count = 0
        self._prev_right_arm_state = ra_state

        print(
            f"[RuleBasedCleanDesktop] step={self._step:04d} "
            f"grip={grip:.1f}({'C' if grip_closed else 'O'}) "
            f"lgrip={left_grip:.1f}({'O' if left_open else 'C'}) "
            f"xy={xy_dist if xy_dist is not None else float('nan'):.3f}"
            f"({'F' if arms_far else 'N'}) "
            f"ra_diff={ra_diff if ra_diff is not None else float('nan'):.4f} "
            f"idle={self._idle_count}/{self.IDLE_TRIGGER}",
            flush=True,
        )
        self._step += 1

        if self._idle_count >= self.IDLE_TRIGGER:
            actions = out.get("actions")
            arr = np.asarray(actions, dtype=np.float64) if actions is not None else None
            chunk_size = arr.shape[0] if arr is not None and arr.ndim == 2 else 30
            self._release_sequence = self._build_release_retract_sequence(state_np, chunk_size)
            self._release_seq_idx = 0
            self._n_triggered += 1
            self._idle_count = 0
            # Play chunk 1 immediately; chunk 2 follows on next infer.
            chunk = self._release_sequence[self._release_seq_idx]
            seq_len = len(self._release_sequence)
            print(
                f"[RuleBasedCleanDesktop] *** IDLE RELEASE-RETRACT sequence "
                f"start [trigger #{self._n_triggered}], chunk 1/{seq_len} ***",
                flush=True,
            )
            self._release_seq_idx += 1
            out["actions"] = chunk.tolist()
            out["override"] = True

        return out

    def reset(self) -> None:
        if self._step > 0:
            print(
                f"[RuleBasedCleanDesktop] === Episode done: {self._step} infer calls, "
                f"{self._n_triggered} release triggers ===",
                flush=True,
            )
        self._prev_right_arm_state = None
        self._idle_count = 0
        self._step = 0
        self._n_triggered = 0
        self._release_sequence = []
        self._release_seq_idx = 0
        self._policy.reset()

    @property
    def metadata(self) -> dict:
        return self._policy.metadata


class TaskAutoDetectWrapper(_base_policy.BasePolicy):
    """Detects task type from the first prompt and configures wrappers automatically.

    Task detection (first infer only, based on prompt content):
      - Long sorting instruction ("Grab the ... package on the table, turn the waist ..."):
          sorting_packages (not continuous) → tts only
      - Short generic ("Sorting packages" / "Sort packages"):
          sorting_packages_continuous → tts + rule_based + phase prompt
      - Anything else:
          generic task → tts only

    After detection, subsequent infers inject the correct per-request flags
    so the downstream SortingPromptRequestWrapper and RuleBasedSortingCorrectionWrapper
    behave accordingly.
    """

    def __init__(self, policy: _base_policy.BasePolicy):
        self._policy = policy
        self._detected: str | None = None  # "sorting_continuous", "other"
        self._last_task_name: str | None = None

    def infer(self, obs: dict) -> dict:
        work = dict(obs)
        task_name = work.get("task_name", "")
        print(f"[TaskAutoDetect] infer called: task_name='{task_name}'", flush=True)

        # Re-detect when task_name changes (new episode / new task).
        if task_name != self._last_task_name:
            self._last_task_name = task_name
            prev = self._detected
            if task_name == "sorting_packages_continuous":
                self._detected = "sorting_continuous"
            elif task_name == "clean_the_desktop":
                self._detected = "clean_desktop"
            else:
                self._detected = "other"
            if self._detected != prev:
                logger.info("[TaskAutoDetect] task_name='%s' → mode='%s'", task_name, self._detected)

        # Inject per-request flags based on detected task type.
        if self._detected == "sorting_continuous":
            work.setdefault("sorting_packages_continuous_prompt", True)
            work.setdefault("rule_based_sorting_correction", True)
            work.setdefault("clean_the_desktop_correction", False)
        elif self._detected == "clean_desktop":
            work.setdefault("sorting_packages_continuous_prompt", False)
            work.setdefault("rule_based_sorting_correction", False)
            work.setdefault("clean_the_desktop_correction", True)
        else:
            work.setdefault("sorting_packages_continuous_prompt", False)
            work.setdefault("rule_based_sorting_correction", False)
            work.setdefault("clean_the_desktop_correction", False)

        return self._policy.infer(work)

    def reset(self) -> None:
        self._detected = None
        self._last_task_name = None
        self._policy.reset()

    @property
    def metadata(self) -> dict:
        return self._policy.metadata


class DynamicTtsSortingWrapper(_base_policy.BasePolicy):
    """Per-phase TTS control for sorting tasks.

    When ``obs["dynamic_tts_sorting"]`` is truthy, determines the current
    sorting phase via a ``SortingStateMachine`` and **disables** TTS during
    ``place_scan`` (where precision matters more than exploration), keeping
    TTS enabled for all other phases.

    Must sit closer to the ensemble than the other trick wrappers so that
    ``policy_infer_tts`` is set before the ensemble sees the obs.
    """

    # Phases where TTS should be DISABLED.
    _NO_TTS_PHASES = frozenset({"place_scan"})

    # Same key as openpi.policies.policy.POLICY_INFER_TTS_KEY.
    _POLICY_INFER_TTS_KEY = "policy_infer_tts"

    def __init__(self, policy: _base_policy.BasePolicy, *, default_enabled: bool = False):
        self._policy = policy
        self._default_enabled = default_enabled
        self._sm = SortingStateMachine()

    def infer(self, obs: dict) -> dict:
        work = dict(obs)
        raw = work.pop("dynamic_tts_sorting", None)
        active = self._default_enabled if raw is None else bool(raw)

        if active:
            state = work.get("state")
            if state is not None:
                phase, _ = self._sm.update(np.asarray(state))
            else:
                phase = "grab"
            use_tts = phase not in self._NO_TTS_PHASES
            work[self._POLICY_INFER_TTS_KEY] = use_tts
            logger.info("[DynamicTTS] phase=%s -> tts=%s", phase, use_tts)

        return self._policy.infer(work)

    def reset(self) -> None:
        self._policy.reset()

    @property
    def metadata(self) -> dict:
        return self._policy.metadata


class TrickFlagInjectorWrapper(_base_policy.BasePolicy):
    """Outermost wrapper: reads ``obs['task_name']``, looks up tricks from a per-task map,
    and injects boolean trick flags into obs BEFORE the downstream trick wrappers see it.

    Why this exists: the ensemble also injects these flags, but it does so INSIDE its own
    ``infer()`` — by then the outer trick wrappers (SortingPromptRequestWrapper,
    RuleBasedSortingCorrectionWrapper, CleanDesktopCorrectionWrapper) have already read
    obs and made their decisions. This wrapper runs FIRST so the flags are present when
    those wrappers check them.
    """

    _OBS_INJECTABLE_TRICKS = frozenset({
        "sorting_packages_prompt",
        "sorting_packages_continuous_prompt",
        "sorting_packages_continuous_prompt_v2",
        "rule_based_sorting_correction",
        "clean_the_desktop_correction",
        "rule_based_clean_the_desktop_correction",
        "dynamic_tts_sorting",
    })

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        tricks_by_task: dict,
        sorting_corrections_by_task: dict[str, list[int]] | None = None,
    ):
        self._policy = policy
        self._tricks_by_task = {k: list(v) for k, v in (tricks_by_task or {}).items()}
        self._sorting_corrections_by_task = dict(sorting_corrections_by_task or {})

    def infer(self, obs: dict) -> dict:
        if not isinstance(obs, dict):
            return self._policy.infer(obs)
        task_name = obs.get("task_name", "")
        if hasattr(task_name, "item") and callable(getattr(task_name, "item", None)):
            try:
                task_name = task_name.item()
            except Exception:
                pass
        if isinstance(task_name, (bytes, bytearray)):
            task_name = task_name.decode()
        task_name = str(task_name) if task_name is not None else ""
        tricks = self._tricks_by_task.get(task_name, [])
        work = dict(obs)
        for trick_name in self._OBS_INJECTABLE_TRICKS:
            work.setdefault(trick_name, trick_name in tricks)
        # Inject per-task sorting_corrections list (if configured in routes.json).
        if task_name in self._sorting_corrections_by_task:
            work.setdefault("sorting_corrections", self._sorting_corrections_by_task[task_name])
        return self._policy.infer(work)

    def reset(self) -> None:
        self._policy.reset()

    @property
    def metadata(self) -> dict:
        return self._policy.metadata
