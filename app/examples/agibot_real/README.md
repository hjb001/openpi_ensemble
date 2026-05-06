# Agibot G02 Real-Robot Client

A WebSocket client that drives a physical Agibot G02 robot using the
existing `serve_policy.py --env=g2sim` server. The client packs camera
frames + joint state into the schema the server's `Go2ACOTInputs` /
`LerobotACOTGo2DataConfig` already expects, so the server runs unchanged.

## Hardware setup

1. Connect dev machine to the robot via Ethernet. Set the dev machine to
   the static IP `10.42.1.102` (the robot is `10.42.1.101`).
2. Verify connectivity: `ping 10.42.1.101`.
3. Install GDK on the dev machine:
   ```bash
   curl -sSL http://10.42.1.101:8849/install.sh | bash
   ```
4. Build and install the GDK Python wheel:
   ```bash
   cd ~/.cache/agibot/app/gdk/build_dep/python/pybind/
   pip install . --no-build-isolation
   ```
5. Source the env so `agibot_gdk` is importable:
   ```bash
   source ~/.cache/agibot/app/env.sh
   ```

## Server side (no change to existing flow)

```bash
python scripts/serve_policy.py --env=g2sim \
    policy:checkpoint --policy.config=acot_icra_simulation_challenge_reasoning_to_action \
                      --policy.dir=./sim_real_action
```

(or simply `python scripts/serve_policy.py --env=g2sim`, since `./sim_real_action`
is the new default checkpoint dir for `EnvMode.G2SIM`.)

## Client side

Dry run first (verifies pipeline without moving the robot):

```bash
python -m examples.agibot_real.main \
    --host=127.0.0.1 --port=8000 \
    --prompt="Pour the workpiece into the box" \
    --dry-run
```

After a clean 30 s dry run, drop `--dry-run` and turn `--speed-scale` up
gradually (`0.3 → 0.5 → 1.0`):

```bash
python -m examples.agibot_real.main \
    --host=127.0.0.1 --port=8000 \
    --prompt="Pour the workpiece into the box" \
    --speed-scale=0.3 \
    --reset-pose 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
```

## STATE_LAYOUT alignment (do this once)

The 32-dim state vector layout in `constants.STATE_LAYOUT` must match the
order the model was trained on. Confirm by running:

```bash
python -m examples.agibot_real.dump_state_layout \
    --repo-id=/mnt/public/E6/lerobot/8169/2026021101/gripper/task_6167
```

If the printed joint order differs from `STATE_LAYOUT`, edit the constant
before deploying.

## Safety checklist

- [ ] E-stop reachable; tested before each session.
- [ ] First run after any change is `--dry-run`.
- [ ] First non-dry run uses `--speed-scale=0.3`.
- [ ] `reset_pose` was reviewed; every value within joint limits in `constants.JOINT_LIMITS`.
- [ ] Surrounding workspace clear of obstacles.

## Tests

```bash
python -m pytest examples/agibot_real/tests -v
```

All tests use a fake `agibot_gdk` (`tests/fakes/agibot_gdk.py`) — no
hardware required.

## Out of scope (future)

- 100 Hz dual-thread control (current client is 30 Hz, relies on GDK
  interpolation).
- Dexterous-hand support (only omnipicker grippers are wired up; range
  `(-0.785, 0)`).
- ROS2 deployment.
- Server-side schema changes.

See `docs/superpowers/specs/2026-05-06-agibot-real-client-design.md` for
the full design rationale.
