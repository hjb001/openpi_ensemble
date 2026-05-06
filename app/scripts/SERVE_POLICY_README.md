# Policy 服务启动说明（`serve_policy.py`）

本文说明如何用 `uv run` 启动策略推理服务，以及本仓库为 **按 `task_name` 切换多 checkpoint（task-routed ensemble）** 所做的代码改动。

---

## 1. 前置条件

- 在 **`openpi` 项目根目录**（包含 `pyproject.toml`、`src/`、`scripts/` 的目录，本 fork 下一般为 `app/`）内执行命令。
- 已用 `uv` 同步依赖（例如 `uv sync`），且本机 JAX 能访问 **GPU** 时，模型权重与计算通常落在 GPU 上；否则在 CPU 上运行。
- Checkpoint 路径需存在或可下载（`gs://` 等由 `openpi` 的下载逻辑处理）。

---

## 2. 启动方式概览

Tyro 将 `policy` 字段做成 **子命令**。不写子命令时等价于 **`policy:default`**：按 `--env` 选择内置默认 checkpoint，**只加载一个模型**。

通用参数（与子命令无关）示例：

| 参数 | 说明 |
|------|------|
| `--env` | 仅在 `policy:default` 时使用，选择环境对应的默认 checkpoint |
| `--port` | WebSocket 监听端口（默认 `8000`） |
| `--default-prompt` | 当数据里没有 `prompt` 时注入的默认语言指令 |
| `--record` | 将每次 `infer` 的输入输出存到 `policy_records/` |
| `--partial-execution-steps` | 只返回动作 chunk 的前 N 步（receding horizon） |
| `--tts` | Test-time scaling（更长去噪步数 + 多样本均值等，见 `Policy` 实现）；task-routed ensemble 下可作为 **未单独指定时的默认** |

**参数顺序（Tyro）：** 使用 `policy:checkpoint` / `policy:task-routed-ensemble` 等子命令时，**全局参数必须写在子命令前面**。例如 `--port`、`--env`、`--tts` 属于最外层；若写在子命令后面，会被当成子命令自己的参数，从而出现 **`Unrecognized options: --port`**。正确示例：

```bash
uv run scripts/serve_policy.py --port 8999 policy:task-routed-ensemble \
  --policy.routes-json ./checkpoints/routes.json \
  --policy.default.config YOUR_CONFIG \
  --policy.default.dir YOUR_DIR
```

查看完整帮助：

```bash
cd /path/to/openpi/app   # 含 pyproject.toml 的目录
uv run scripts/serve_policy.py --help
```

---

## 3. 方式 A：默认策略（最常用）

按环境加载 **唯一** 一份 `DEFAULT_CHECKPOINT` 中的模型。

**示例（G2SIM + 指定端口）：**

```bash
uv run scripts/serve_policy.py --env G2SIM --port 8999
```

等价于显式写默认子命令：

```bash
uv run scripts/serve_policy.py policy:default --env G2SIM --port 8999
```

`G2SIM` 对应的配置名与 checkpoint 目录在 `scripts/serve_policy.py` 的 `DEFAULT_CHECKPOINT` 字典中定义（`EnvMode.G2SIM` → `config` + `dir`）。

---

## 4. 方式 B：指定单个 checkpoint

不依赖 `--env` 的默认表，直接指定训练配置名与 checkpoint 目录：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi0_aloha_sim \
  --policy.dir checkpoints/pi0_aloha_sim/exp/10000 \
  --port 8000
```

---

## 5. 方式 C：Task-routed ensemble（按 `task_name` 多模型）

每次 `infer(obs)` 根据 **`obs["task_name"]`** 选择子模型；若缺失或不在路由表中，则使用 **default** 对应的 checkpoint。

**显存与加载次数（重要）：**

- **不是「每个 task 必占一份独立权重」**。实现上按 **`(config, dir)` 去重**（目录会做 `expanduser` + `resolve`）：多条 `routes.json` 只要指向 **同一训练配置名 + 同一 checkpoint 路径**，只会 **`create_trained_policy` 一次**，多个 `task_name` **共享同一个 `Policy` 实例**。
- **`default`** 若与某条 route 的 `(config, dir)` 相同，也会 **复用同一份已加载模型**，不会为 default 再加载第二遍。
- 你只有 **两个** 不同 checkpoint 时，无论有多少个 `task_name`，通常 **最多占两份模型显存**（外加各自推理时的临时激活等）。
- 启动日志里会打印：`X unique checkpoint(s) in memory, Y task route(s)`，其中 `X` 才是实际加载份数。

### 5.1 准备 `routes.json`

非空 JSON **数组**，每个元素为对象，字段与单 checkpoint 一致：`task_name`、`config`、`dir`。可选 **`tts`**（布尔）：该子任务是否走 TTS 推理；省略时与命令行顶层的 **`--tts`** 一致。

示例 `routes.json`：

```json
[
  {
    "task_name": "sorting_packages",
    "config": "acot_icra_simulation_challenge_reasoning_to_action",
    "dir": "./checkpoints/model_for_sorting",
    "tts": true
  },
  {
    "task_name": "other_task",
    "config": "acot_icra_simulation_challenge_reasoning_to_action",
    "dir": "./checkpoints/model_for_other",
    "tts": false
  }
]
```

`task_name` 必须与客户端传入 observation 里的字符串 **完全一致**（脚本侧会用 `task_name_from_obs()` 规范化 `bytes` / numpy 标量等为 `str`）。

**TTS 与显存：** 去重键只有 **`(config, dir)`**。同一 checkpoint 上不同子任务可以 `tts: true/false` 混用，**不会**为 TTS 再加载第二份权重；服务在每次 `infer` 里注入内部字段 **`policy_infer_tts`**（由 `TaskRoutedEnsemblePolicy` 写入，会在进模型前剥掉）。顶层 `--tts` / `routes.json` 里的 `tts` / `--policy.default.tts` 只决定「这一步是否走 TTS 计算路径」，不增加模型份数。

**ensemble 的 default：** `--policy.default.tts`（或顶层 `--tts`）只控制 **未命中路由** 时的 TTS；同样与 default 的 `(config, dir)` 共用一份权重。

**高级：** 若直接调用单个 `Policy.infer`（非 ensemble），也可在 `obs` 顶层设置 **`policy_infer_tts`**（布尔）做单次覆盖；勿与训练数据里真实字段冲突。

### 5.2 启动命令

```bash
uv run scripts/serve_policy.py --port 8999 policy:task-routed-ensemble \
  --policy.routes-json /absolute/or/relative/path/to/routes.json \
  --policy.default.config your_train_config_name \
  --policy.default.dir /path/to/default_checkpoint
```

查看该子命令专用参数：

```bash
uv run scripts/serve_policy.py policy:task-routed-ensemble --help
```

连接成功后，服务端会先发送 **metadata**（ensemble 模式下包含 `ensemble`、`routed_tasks`、各子 policy 的 metadata 等）。

### 5.3 客户端侧约定

- 在发给 `infer` 的 `obs` 顶层（或与现有代码一致、能被 `jax.tree.map(..., obs).get("task_name")` 读到的位置）放入 **`task_name`**。
- 路由命中时，由对应的 **`Policy.infer`** 完整执行（含该模型的 input transform、推理、output transform 以及 **`Policy.post_process`**，例如与 waist 相关的后处理）。

---

## 6. 代码改动说明（本 fork 相对「仅单模型 serve」的扩展）

### 6.1 `src/openpi/policies/policy.py`

| 新增/保留内容 | 作用 |
|---------------|------|
| **`task_name_from_obs(obs)`** | 从 observation 中读取 `task_name`，行为与 `post_process` 中 `jax.tree.map(lambda x: x, obs).get("task_name")` 一致，并处理 `bytes`、带 `.item()` 的 numpy 标量等，得到 `str \| None`。 |
| **`TaskRoutedEnsemblePolicy`** | 包装多个 `Policy` 实例；`infer` 内根据 `task_name_from_obs` 选择子 policy 或 `default_policy`；按路由表注入 **`policy_infer_tts`**，使同一 checkpoint 仅加载一次。构造参数 **`tts_by_task`** / **`default_tts`** 与 `policies_by_task` 对齐。提供 **`metadata`**（含 `per_task_tts`、`default_tts`）。可选 `log_routing`。 |
| **`POLICY_INFER_TTS_KEY` / `infer_tts_override_from_obs`** | 单次 `infer` 的 TTS 覆盖；ensemble 与可选客户端约定共用。 |

**`Policy.infer`** 支持顶层 **`policy_infer_tts`** 单次覆盖并在进 transform 前剔除；**`post_process`** 仍用调用方传入的 `obs`（ensemble 传入的副本上可能带该键，一般不影响 `task_name` 等字段）。**`metadata.policy_tts`** 仍为构造期默认值；ensemble 下子 policy 多为 `False`，真实每任务 TTS 见 **`TaskRoutedEnsemblePolicy.metadata`** 的 `per_task_tts` / `default_tts`。

### 6.2 `scripts/serve_policy.py`

| 改动 | 作用 |
|------|------|
| **`TaskRoute`（数据类）** | 描述 JSON 里单条路由的 `task_name`、`config`、`dir`（解析时用）；可选 `tts` 写入 **`tts_by_task`**。 |
| **`Checkpoint.tts`** | 用于 **`policy:checkpoint`** / **`default`**；ensemble 内子模型固定 **`tts=False`** 加载，真实 TTS 由路由表 + `policy_infer_tts` 注入。 |
| **`_ensemble_base_policy_from_checkpoint`** | ensemble 专用加载：`create_trained_policy(..., tts=False)`。 |
| **`_resolved_tts`** | 解析 default 与各 route 的布尔 TTS（route 缺省则用顶层 `--tts`）。 |
| **`TaskRoutedEnsemble`（数据类）** | Tyro 子命令 `policy:task-routed-ensemble`；字段 `default: Checkpoint` 与 `routes_json: pathlib.Path`。 |
| **`Args.policy` 类型扩展** | `Checkpoint \| Default \| TaskRoutedEnsemble`，默认仍为 `Default`。 |
| **`_policy_from_checkpoint`** | 抽取公共的「从 `Checkpoint` + 全局 args 创建 `Policy`」逻辑。 |
| **`_checkpoint_cache_key`** | 将 `(config, dir)` 规范成可哈希键，用于 ensemble 内 **按 checkpoint 去重**，避免相同 `dir` 被多个 task 重复加载。 |
| **`create_policy` 中 `match`** | 对 `TaskRoutedEnsemble`：读 JSON、校验，经缓存 **`_get_policy`** 为每个 **唯一** checkpoint 最多调用一次 `create_trained_policy`，再按 `task_name` 映射到共享的 `Policy`，构造 **`TaskRoutedEnsemblePolicy`**。 |

依赖：`json`、`pathlib`（读 `routes_json`）。

### 6.3 未修改的文件（供对照）

- **`openpi/serving/websocket_policy_server.py`**：仍为 `infer(obs)` + metadata 首包；ensemble 的 `metadata` 来自 `TaskRoutedEnsemblePolicy.metadata`。
- **`openpi/policies/policy_config.py`**：仍通过 `create_trained_policy` 加载单 checkpoint；ensemble 多次调用该函数即可。

---

## 7. 常见问题

**Q：`uv run scripts/serve_policy.py --env G2SIM` 会把所有环境的模型都加载到 GPU 吗？**  
A：不会。`policy:default` 只加载当前 `--env` 对应 **一个** checkpoint。

**Q：ensemble 下显存不够怎么办？**  
A：先确认 **不同 checkpoint 的个数**（看启动日志里的 `unique checkpoint(s)`）。若很多 task 共用两条 `dir`，应只占两份权重。若仍不够：减少 **互不相同** 的 `(config, dir)`、拆进程/多卡，或 CPU（JAX 配置与性能需自行权衡）。本实现未做「按需换模」或单卡时间复用卸载。

**Q：`task_name` 对不上会怎样？**  
A：使用 ensemble 的 **`default`** checkpoint 对应的子 `Policy`。

---

## 8. 程序内直接使用 ensemble（不经过 CLI）

```python
from openpi.policies.policy import Policy, TaskRoutedEnsemblePolicy

ensemble = TaskRoutedEnsemblePolicy(
    policies_by_task={"task_a": policy_a, "task_b": policy_b},
    default_policy=policy_default,
    tts_by_task={"task_a": True, "task_b": False},
    default_tts=False,
)
# 与 serve 一样：obs 中携带 task_name 即可切换
out = ensemble.infer(obs)
```

---

文档路径：`scripts/SERVE_POLICY_README.md`（与 `scripts/serve_policy.py` 同目录）。
