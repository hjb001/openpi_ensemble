# Action Chain-of-Thought-VLA

This repository is the **official implementation** of:

**ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models**  
[[Paper]](https://arxiv.org/abs/2601.11404)

In addition, this repo also serves as the **official baseline implementation for the AGIBot ICRA Simulation Challenge**.

---

## 🔥 Overview

We introduce **Action Chain-of-Thought (ACoT)** — a reasoning paradigm where the decision process of a robot is formulated as a structured sequence of **coarse action intents**, which explicitly guides downstream policy learning.

Building upon this idea, we propose **ACoT-VLA**, a vision-language-action framework composed of two complementary reasoning modules:

- **Explicit Action Reasoner (EAR)**  
  Generates coarse action trajectories as explicit reasoning steps.

- **Implicit Action Reasoner (IAR)**  
  Extracts latent action priors from multimodal internal representations.

Together, EAR and IAR co-form an Action Chain-of-Thought that conditions the final action head, enabling grounded and long-horizon policy learning.

![framework](docs/framework.png)

---

## 🏆 AGIBot ICRA Simulation Challenge Baseline

Besides being the official research implementation, this repository also provides:

- A **training baseline**
- A **deployment & inference pipeline**
- Example configs for the **Genie Sim ICRA Simulation Challenge**

The overall workflow is:

```

Dataset (LeRobot format)
↓
Compute normalization statistics
↓
Train ACoT-VLA
↓
Launch policy server
↓
Run Genie Sim evaluation client

````

---

## TODO

- [x] Release training code  
- [x] Release inference code  
- [ ] Release model weights on simulation benchmarks  

---

## 🚀 Get Started

### 1. Clone repository

```bash
git clone <your_repo_url>
git submodule update --init --recursive
````

---

### 2. Install dependencies

We use **uv** to manage Python environments.

Installation guide:
[https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

---

## 🧠 Base Model

ACoT-VLA is built upon **π0.5**.

Please refer to the official OpenPI repository for checkpoint usage and model structure:

[https://github.com/Physical-Intelligence/openpi/tree/main](https://github.com/Physical-Intelligence/openpi/tree/main)

---

## 📊 Training

### 1. Dataset Preparation

We adopt the **LeRobot dataset format** for post-training.

Here is an example file to convert your dataset first:

```bash
examples/libero/convert_libero_data_to_lerobot.py
```

---

### 2. Training Config

Training configurations are defined in:

```
src/openpi/training/config.py
```

Below is the **ICRA Simulation Challenge baseline configuration**:

```python
TrainConfig(
    name="acot_icra_simulation_challenge_reasoning_to_action",

    # Both coarse and fine action horizons are set to 30
    # Utilize both EAR and IAR
    model=acot_vla.ACOTConfig(
        coarse_action_horizon=30,
        action_horizon=30,
        paligemma_variant="gemma_2b_lora",
        adopt_explicit_action_reasoner=True,
        adopt_implicit_action_reasoner=True,
        downsample_based_implicit_extractor=True,
    ),

    data=LerobotACOTGo2DataConfig(
        default_prompt="This is the icra simulation challenge baseline config. Please refer to the README for details.",

        # ===== Training tasks =====
        repo_id=[
            "/mnt/public/E6/lerobot/7819/task_5833",
            "/mnt/public/E6/lerobot/7820/task_5832",
            "/mnt/public/E6/lerobot/8153",
            "/mnt/public/E6/lerobot/7821/task_5829",
            "/mnt/public/E6/lerobot/7837/task_5441",
            "/mnt/public/E6/lerobot/7944/task_6100",
            "/mnt/public/E6/lerobot/7818/task_5853",
            "/mnt/public/E6/lerobot/8169/2026021101/gripper/task_6167",
            "/mnt/public/E6/lerobot/7878/task_5828",
        ],

        assets=AssetsConfig(
            assets_dir=None,
            asset_id="/mnt/zhonglinqing/data/datasets/genie_sim_icra_datasets/nine_dataset_merge_assets",
        ),

        # Brief prompt replacement for instruction diversity
        prompt_map_inject_to_training={
            "Unload workpiece_icra_SIM": ("Pour the workpiece into the box", 0.5),
            "Turn the doorknob": ("Turn the doorknob and push the door", 0.5),
            "Make popcorn": ("Scoop the popcorn and pour it into the popcorn bucket", 0.5),
            "Carry the pot": ("Grasp the two handles of the pot and place it on the stove", 0.5),
        },

        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.images.top_head",
                            "hand_left": "observation.images.hand_left",
                            "hand_right": "observation.images.hand_right",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                        "task": "task",
                        "episode_index": "episode_index",
                    }
                )
            ]
        ),

        base_config=DataConfig(
            dataloader_sampler="subtask",
            prompt_from_hl_instruction=True,
        ),

        joint_action_shifts=(2, 1),
        extra_delta_transform=(True, True),
        delta_action_mask=_transforms.make_bool_mask(14, -18),
    ),

    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),

    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,

    weight_loader=weight_loaders.ACOTCheckpointWeightLoader(
        "/mnt/public/zhonglinqing/pkgs/pi05_model/params"
    ),

    num_train_steps=50_000,
    save_interval=5000,
    num_workers=24,
    batch_size=256,

    freeze_filter=acot_vla.ACOTConfig(
        paligemma_variant="gemma_2b_lora"
    ).get_freeze_filter(
        freeze_vision=False,
        freeze_llm=True,
        freeze_llm_embedder=True,
        freeze_dual_ae=[False, False],
    ),
)
```

---

### 3. Compute Normalization Statistics

Before training:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name acot_icra_simulation_challenge_reasoning_to_action
```

---

### 4. Train

```bash
bash scripts/train.sh CONFIG_NAME EXP_NAME
```

Checkpoints will be saved under:

```
checkpoints/
```

---

## 🧪 Inference & Evaluation (Genie Sim)

Evaluation follows a **server-client architecture**.

### Step 1 — Prepare Genie Sim environment

See GenieSim 3.0

---

### Step 2 — Configure model

Edit:

```
scripts/serve_policy.py
```

Specify:
* config name
* checkpoint path

Here is an example:
```
EnvMode.G2SIM: Checkpoint(
    config="acot_icra_simulation_challenge_reasoning_to_action,
    dir="./checkpoints/acot_icra_simulation_challenge_reasoning_to_action/exp_name/30000",
)
```

---

### Step 3 — Launch model server

Example:

```bash
bash scripts/server.sh 0 8000
```

* `0`: GPU id
* `8000`: communication port

---

### Step 4 — Run evaluation client
See GenieSim 3.0

---

## 📈 Results

We report representative results on the ICRA simulation tasks.

| Model               | Average Success Rate |
| ------------------- | -------------------- |
| ACoT-VLA (Baseline) | TBD                  |

(Results will be updated after benchmark release.)

---

## 🧩 Design Notes

* ACoT treats action generation with **action space guidance**.
* Both explicit and implicit reasoning are important for robotic manipulation tasks.
* Teacher forcing is key for stable optimization.

---

## 🙏 Acknowledgements

Part of the codebase is built upon the OpenPI framework.
We sincerely thank the authors for their excellent work.
