# Parallel Eval Direct Execute Task Workflow

本工作流定义 `parallel_eval.py` 为单个 case 启动的独立 worker 如何直接执行`execute_task` 阶段。`parallel_eval.py` 在调度端从当前评测集 JSON 中提取`prompt`，worker 直接使用调度器传入的任务指令完成任务。

本工作流适用于代码生成、代码修复、真实推理、真实评测和非训练型最小冒烟验证，不负责执行真实模型训练。任务要求生成训练代码、训练配置或训练启动入口时，可以生成并验证这些静态产物，但不得启动真实训练循环或提交真实训练作业。

---

## 1. 执行角色

调用本工作流的进程已经是 `parallel_eval.py` 为当前 case 创建的独立冷启动worker，因此该 worker 就是任务执行者，不是二次调度器。

- 必须在当前 worker 内直接执行 `prompt` 对应的任务。
- 禁止为了执行当前 case 再创建子智能体、协作代理或其他 case worker。
- 允许在当前 worker 内调用 `prompt` 明确要求或任务确实需要的技能和工具。
- 一个 worker 只能处理输入中指定的一个 case，不得执行或查看其他 case。
- 整个过程禁止向用户提问或请求确认；需要选择时按 `prompt`、stage 配置和已加载技能的确定性默认规则继续执行。

---

## 2. 输入

本工作流的三个必需输入由 `parallel_eval.py` 在 `Worker assignment` 中传入：

| 字段 | 必填 | 说明 |
|---|---|---|
| `prompt` | 是 | 调度器为当前 case 构造的有效任务指令；通常为原始任务指令，`use_oneskill=true` 时包含 OneScience 入口前缀 |
| `artifact_path` | 是 | 当前 case 的任务产物目录 |
| `evaluation_result_output_dir` | 是 | 当前 case 的评测过程与结果目录 |

调度器还会附带 `case_id`、`run_context`、`instance_id`、`use_oneskill`、可用 OneScience 技能类型清单和 `stage.params`。这些字段用于结果元数据、执行约束、技能编排和冒烟测试配置；worker 不得自行改写三个必需输入。`case_id` 已由调度器解析，worker 不得自行读取 case 文件补充或校验它。

冒烟测试环境字段固定为 `stage.params.conda_env`。`max_debug_attempts` 也只能从当前 `stage.params` 读取，缺省值为 `0`。

任务指令和输出路径由 `parallel_eval.py` 分配，必须直接使用：

- `prompt` 必须保持调度器传入的有效文本，不得从其他字段或文件重新构造。`use_oneskill=true` 时，其 OneScience 入口前缀由调度器添加，不属于 worker 篡改任务。
- 不得使用 case 中的 `expected_artifact_path` 改写 `artifact_path`。
- 不得将 profile 的 `eval_type`、case 的 `eval_type` 或 `case_id` 再拼接到已经分配好的目录后面。
- 路径不存在时，可以创建 `artifact_path`、`evaluation_result_output_dir` 及 `<evaluation_result_output_dir>/logs`；其他输入缺失或类型不正确时必须失败。

---

## 3. 执行前准备与资源边界

执行任务前，当前 worker 可以获取以下资源：

- `stages[*].workflow` 指定的当前启用阶段工作流文件
- `Worker assignment` 中已经提供的 `prompt`

除上述资源外，只允许在当前 `artifact_path` 和 `evaluation_result_output_dir` 内进行本工作流规定的读写。原始 case JSON 不是 worker 输入；worker 不得定位、读取或搜索它。任务输入可以位于 artifact 目录之外，但任务产物不得写到 artifact 目录之外。

资源访问必须满足以下限制：

- 不得尝试发现或读取原始 case JSON。调度器是 case 文件的唯一读取者，并且只把顶层 `prompt` 暴露给 worker；`checks`、期望输出和评分数据不会作为任务输入。
- 不得通过环境变量、进程参数、目录枚举、文本搜索或路径推导寻找 case 文件。
- 不得读取、枚举、搜索或引用未由 prompt 明确声明、且不是完成当前代码生成任务所必需的其他本地路径；不得把工作区根目录或其他 case 目录当作任务输入。
- 严禁访问 `benchmarks/standard/`、`benchmarks/evaluators/` 及其任何子目录或文件；不得通过索引、缓存、符号链接、复制副本或其他间接方式获取其中内容。
- 严禁读取其他 case、其他 agent、其他 base model、历史运行或下游阶段生成的 `eval_results/`、`submissions/`、评分结果、执行日志和任务产物。
- 当前 `artifact_path` 只允许读取本次任务已经创建的内容，或 `prompt` 明确声明预置在该目录中的输入文件；不得把历史遗留内容当作本次生成结果。
- 当前 `evaluation_result_output_dir` 只允许写入本工作流的日志和结果，并进行必要的存在性检查；不得读取其中的历史结果作为任务上下文。
- 若当前任务执行失败，必须直接进行真实的阻断并如实记录，不得生成 `BLOCKED.md`；冒烟测试的失败不得算作任务的失败。
- 禁止读取 evaluator 实现、标准答案、评分规则或当前 case 的期望产物来反推实现。

技能或工具读取了参考文档、技能说明、模板、配置或其他允许访问的文本时，必须按真实读取顺序保留实际送入当前 worker 上下文的原始内容，写入
`execute_task_read_content_records`。不得用摘要、文件介绍或事后补写内容代替。

---

## 4. 执行流程

### 步骤 1：校验输入并准备目录

1. 校验 `prompt` 是非空字符串。
2. 校验 `artifact_path` 和 `evaluation_result_output_dir` 是非空目录路径。
3. 创建 `artifact_path`、`evaluation_result_output_dir` 和 `<evaluation_result_output_dir>/logs`。
4. 从开始执行起，将真实过程按时间顺序追加到`<evaluation_result_output_dir>/logs/execute_task.log`。

输入校验失败时不得执行任务，但仍应在能够确定且可写的 `evaluation_result_output_dir` 中输出失败结果。

### 步骤 2：使用任务指令

直接使用 `Worker assignment.prompt`。该值已经由调度器从 case 顶层字段构造：`use_oneskill=false` 时使用 case 原始任务指令，`use_oneskill=true` 时在原始任务前增加 OneScience orchestrator 入口要求。worker 不需要也不得再次读取 case 文件。`prompt` 缺失、类型错误或去除首尾空白后为空时，记录输入失败并停止。

执行时必须保持 `prompt` 原文语义和约束：

- 不得修改、删减、扩写或用 case 中其他字段拼接调度器已经构造完成的有效 `prompt`。
- `prompt_prefix`、`prompt_suffix`、`checks`、`expected_artifact_path` 等字段都不是当前任务指令，不得加入 `prompt`。
- `prompt` 是当前 case 唯一的业务需求；本工作流仅补充执行边界、路径边界、
  冒烟测试和结果落盘要求。

### 步骤 3：在当前 worker 内直接执行任务

当前 worker 直接完成 `prompt`，不得再次调度其他智能体。任务执行期间必须遵守：

- `use_oneskill=true` 时，“当前 worker 直接完成”表示在同一个 OpenCode worker 内通过原生 `skill` 工具执行 OneScience 编排，不是绕过技能自行实现。技能调用和技能间 handoff 不属于创建其他 case worker 或子智能体。
- 此时第一个工具调用必须加载 `onescience-orchestrator`，第二个工具调用必须加载与任务匹配的 `type=resource` OneScience 技能；随后按 orchestrator 要求完成意图识别、匹配的 expert 规划、executor 清单核定和执行技能路由。
- 至少一个匹配的 `type=executor` OneScience 技能必须在创建、修改或运行任务产物之前加载并用于任务执行。宿主提供原生 `skill` 工具时，禁止以内联模拟、通用 `bash`/`read`/`apply_patch` 或其他普通工具替代应有的下游技能调用。
- 下游技能的选择必须依据 orchestrator 和各技能职责动态决定，不得仅因 case 属于 infer/train 等目录就硬编码为固定技能；没有匹配 expert 时应记录空召回结果，但 resource 和 executor 阶段不得省略。

- 所有代码、配置、脚本、manifest、模型定义、说明文件和任务产生的业务结果等任务产物，都必须写入 `artifact_path` 或其子目录。
- 相对路径一律以 `artifact_path` 为基准；worker 的工作目录已经设置为 `artifact_path`，必须保持该目录作为工作目录，避免文件落到仓库根目录或其他默认目录。
- `artifact_path` 是当前 case 的产物根目录，不是对 prompt 中相对输出目录的替换；例如 prompt 指定 `./test_case/` 时，必须写入 `<artifact_path>/test_case/`，不得把该目录下的文件扁平化到 `<artifact_path>` 根部。
- 禁止把任务产物写入 `evaluation_result_output_dir`；该目录只保存本工作流日志、冒烟验证日志和 `execute_task_result.json`。
- 禁止把产物写入项目根目录、其他 submission 或其他 case 目录。
- `prompt` 要求调用技能时，在当前 worker 内按技能说明执行；技能生成的所有任务文件仍必须落在 `artifact_path`。
- 每次创建、修改、删除文件以及每次工具或技能调用，都必须立即追加记录到`execute_task.log`，不得在任务结束后伪造或回顾性补写过程。

### 步骤 4：受控 debug

- 从 `stage.params.max_debug_attempts` 读取最大 debug 次数，缺省为 `0`。
- 一次“修改任务产物后重新执行验证”的循环计为一次 debug。
- 达到上限后必须停止继续修改，以当前真实状态完成失败结果。
- 不得通过创建新 worker、切换 case 或修改 evaluator 来规避 debug 次数限制。

### 步骤 5：执行最小冒烟测试

任务产物生成后，在不进入真实模型训练的前提下执行与任务类型匹配的最小验证。
详细环境和日志要求见第 5 节。

### 步骤 6：检查产物并输出结果

1. 检查 `prompt` 明确要求的文件是否都位于 `artifact_path`。
2. 递归生成本次任务实际产物的相对路径清单；不得把评测目录文件计入任务产物。
3. 记录冒烟测试状态、失败原因和最终产物清单。
4. 将结果写入
   `<evaluation_result_output_dir>/execute_task_result.json`。
5. 将同一结果对象写入当前 case 的 `stage_context.execute_task`，并在 worker 最终
   `case_result.stage_outputs.execute_task` 中返回。

`passed` 表示 `execute_task` 是否完成了任务执行并留下了任务产物，不表示冒烟
验证是否通过。代码生成类任务没有产生任何任务文件，或必需文件缺失时，`passed`
必须为 `false`，不得仅因为 worker 正常退出而报告成功。只要任务已经正常启动并
完成结果组装、任务产物存在，即使冒烟测试失败、运行环境不可用或静态/动态冒烟
检查未通过，`passed` 仍必须为 `true`；这些信息只记录在 `smoke_test` 及其日志中。

---

## 5. 冒烟测试要求

冒烟测试包括 import、CLI help、配置解析、shape/schema 检查、dry-run、
startup-check 或不会进入真实训练过程的 tiny smoke test。

### 5.1 环境来源

冒烟测试必须只使用 `stage.params.conda_env`：

- 不得从 case、`run_context`、其他 stage 或当前 shell 默认环境推导环境名。
- `conda_env` 缺失、为空或环境不可用时，记录真实阻断并停止冒烟测试。
- 不得回退到 `base` 或其他环境，不得在本阶段创建 Conda 环境或安装依赖。

### 5.2 初始化顺序

模块加载、环境激活和冒烟命令必须在同一个 shell 或同一个远程作业脚本中执行。当前
worker 可能由非交互式 shell 启动；如果 `module` 未定义，必须先加载可用的 Modules/Lmod
初始化脚本，再执行模块加载。`conda activate` 之前也必须加载 conda shell hook：

```bash
if ! command -v module >/dev/null 2>&1; then
  for init in /usr/share/Modules/init/bash /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
    if [ -r "$init" ]; then source "$init"; break; fi
  done
fi
if command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base)"
  if [ -r "$conda_base/etc/profile.d/conda.sh" ]; then
    source "$conda_base/etc/profile.d/conda.sh"
  fi
fi
module load sghpc-mpi-gcc/26.3
module load sghpcdas/25.6
conda activate <stage.params.conda_env>
cd <artifact_path>
<smoke-command>
```

必须把两个 `module load`、`conda activate` 和实际冒烟命令的命令文本、退出码、stdout/stderr 摘要写入 `execute_task.log` 和最终结果。任一环境初始化步骤失败时，不得继续执行冒烟命令，也不得伪造 `smoke_test.passed=true`；任务执行结果仍按第 6 节的任务完成标准单独判定。

### 5.3 执行边界

允许：

- 静态语法和配置检查
- import、CLI help、shape/schema 检查
- 不进入真实训练循环的 dry-run、startup-check 或 tiny smoke test
- 与任务要求一致且不依赖先做真实训练的真实推理或真实评测

禁止：

- 启动真实模型训练循环
- 提交以真实训练为目标的 SLURM 或其他远程作业
- 等待真实 epoch/step 完成
- 为冒烟测试伪造 `trainer.log`、`train.log`、训练 `metrics.json`、训练 `metrics.jsonl`、checkpoint、prediction 或 target 等正式运行证据

若 `prompt` 需要真实训练，本阶段仍应完成代码、配置和入口等静态前置产物，并在结果中明确记录：真实训练需求已识别，但本工作流只执行到非训练型冒烟验证。

冒烟测试的完整 stdout/stderr 写入
`<evaluation_result_output_dir>/logs/` 下具有 `smoke` 标识的文件，不得伪装成正式训练、推理或评测证据。

---

## 6. 输出文件契约

无论成功、失败或部分完成，只要 `evaluation_result_output_dir` 可写，就必须生成：

- `<evaluation_result_output_dir>/execute_task_result.json`
- `<evaluation_result_output_dir>/logs/execute_task.log`
- 实际执行冒烟测试时对应的 stdout/stderr 日志

`execute_task_result.json` 至少包含：

```json
{
  "passed": true,
  "case_id": "my_task_V1",
  "worker_instance_id": "worker instance id",
  "execute_task_infer_content": "按真实时间顺序记录的原始执行日志",
  "execute_task_read_content_records": [
    {
      "path": "实际读取的允许资源路径",
      "content": "实际送入上下文的原始内容"
    }
  ],
  "read_records_fidelity": "raw",
  "execute_task_log_path": "<evaluation_result_output_dir>/logs",
  "execute_task_artifact_path": "<artifact_path>",
  "artifact_manifest": [
    "relative/path/to/generated_file.py"
  ],
  "smoke_test": {
    "attempted": true,
    "passed": true,
    "command": "actual smoke command",
    "exit_code": 0,
    "stdout_log": "<evaluation_result_output_dir>/logs/smoke.out",
    "stderr_log": "<evaluation_result_output_dir>/logs/smoke.err",
    "blocking_reason": null
  },
  "failure_reason": null
}
```

字段要求：

- `execute_task_infer_content` 必须与 `execute_task.log` 的真实过程一致，按时间顺序保留对外消息、工具调用、技能调用、文件操作、报错和重试，不得改写成总结报告。
- `execute_task_read_content_records` 必须保存实际读取并进入上下文的原始内容；没有读取允许资源时为 `[]`。
- 调度器直接传入的 `prompt` 属于任务输入，不属于 worker 的文件读取记录，不得
  为它伪造 `execute_task_read_content_records` 条目。
- `read_records_fidelity` 只能为 `raw`、`mixed`、`summarized` 或 `missing`；只有所有记录都是原始内容时才能填写 `raw`。
- `execute_task_artifact_path` 必须与输入的 `artifact_path` 完全一致。
- `artifact_manifest` 只包含 `artifact_path` 下本次任务真实生成或修改的文件，使用相对于 `artifact_path` 的路径并按字典序排列。
- `smoke_test.attempted=false` 时，必须填写没有执行的真实原因。
- `smoke_test.passed=false` 时仍可将任务标记为 `passed=true`；smoke 的真实阻断原因必须保留在 `smoke_test.blocking_reason` 和对应日志中。
- 只有任务本身未完成、输入/执行被阻断或必需产物缺失时，才将 `passed=false`，并填写可操作、未经掩盖的 `failure_reason`。

当前 worker 最终还必须按 `parallel_eval.py` 要求返回 case 级 JSON；其中 `stage_outputs.execute_task` 必须与已落盘的 `execute_task_result.json` 内容一致。

---

## 7. 失败处理

- 输入无效、资源不可访问、技能失败、任务本身所需依赖缺失、debug 次数耗尽或必需产物缺失，都必须记录真实失败原因并令 `passed=false`。
- 冒烟测试失败或冒烟环境初始化失败不属于 `execute_task` 任务执行失败；必须保留
  `smoke_test.passed=false`、`blocking_reason` 和完整日志，但不得据此令顶层
  `passed=false`。顶层 `failure_reason` 仅用于任务本身未完成、产物缺失或输入/执行
  阻断；若只有 smoke 阻断，顶层 `failure_reason` 应为 `null`。
- 已完成的合法任务产物应保留在 `artifact_path`，不得因为后续失败而伪造、搬移或删除用户需要的产物。
- 失败结果也必须落盘，不能只在 worker 最终消息中返回。
- 禁止用后续 evaluator 的异常覆盖 `execute_task` 的原始失败原因。
- 临时辅助文件在任务结束前删除；不得询问用户是否清理。

---

## 8. 禁止行为

- 禁止二次启动 case worker 或把任务转交给其他智能体。
- 禁止读取或调用 `run.md`、`spawn_subagent.md`。
- 禁止修改、拼接或替换 case 的 `prompt`。
- 禁止定位、读取、枚举或搜索原始 case 文件；任务指令只能来自 `Worker assignment.prompt`。
- 禁止使用 `expected_artifact_path` 或自行推导的目录替代输入路径。
- 禁止把任何任务产物写到 `artifact_path` 之外。
- 禁止读取其他 case、历史结果、标准答案、evaluator 或评分规则。
- 禁止执行真实模型训练或伪造运行证据。
- 禁止只返回内存中的 stage output 而不写入 `execute_task_result.json`。
- 禁止在任务文件为空或必需产物缺失时报告成功。
