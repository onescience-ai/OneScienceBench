<p align="center">
  <img alt="智能体" src="https://img.shields.io/badge/agents-3-green.svg">
  <img alt="任务类型" src="https://img.shields.io/badge/task_types-5-green.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <img alt="许可证" src="https://img.shields.io/badge/license-MIT-lightgrey.svg">
</p>

# OnescienceBench

OnescienceBench 是一个面向科学任务的通用智能体评测基准与执行框架。它不限定智能体的实现形式或能力类型，既可评测科研代码生成智能体，也可评测承担推理、训练任务、论文复现、实验分析等其他科研任务的智能体。目前基准覆盖五类任务：推理、训练代码生成、论文复现、二次开发和端到端科学场景。

目前，执行框架支持以下三种智能体：

- [Codex CLI](https://developers.openai.com/codex/cli/)
- [OpenScience](https://www.npmjs.com/package/@synsci/openscience)
- [Biomni](https://pypi.org/project/biomni/)

每个用例都会在全新的智能体进程中启动。智能体将任务产物写入隔离的提交目录，随后评测框架统计 Token 用量、运行用例对应的规则检查，并生成机器可读的汇总结果。

## 评测内容

对于每个评测用例，OnescienceBench 会执行以下流程：

1. 从 JSON 数据集中读取用例提示词和执行设置。
2. 使用评测 Profile 中选择的智能体启动独立的冷启动 Worker。
3. 将生成的代码、配置和其他任务产物写入该用例专属的 `submissions/` 目录。
4. 执行配置的非训练型冒烟验证，并记录完整执行日志。
5. 使用与该用例关联的评测器执行确定性检查。
6. 当 Profile 启用评分阶段时计算分数。
7. 将通过/失败状态及 Token 用量汇总到 `summary.json`。

任务 Worker 无法读取评测器实现、预期产物或其他智能体的历史输出。对于训练类任务，评测对象是生成的训练实现及其最小冒烟验证；默认工作流不会启动真实训练。

## 评测任务

| 任务类型 | Profile | 默认数据集 |
|---|---|---|
| 推理 | `eval_profiles/onescience_infer.json` | `benchmarks/datasets/infer` |
| 训练 | `eval_profiles/onescience_train.json` | `benchmarks/datasets/train` |
| 论文复现 | `eval_profiles/onescience_paper_repro.json` | `benchmarks/datasets/paper_repro` |
| 二次开发 | `eval_profiles/onescience_sec_develop.json` | `benchmarks/datasets/sec_develop` |
| 科学场景 | `eval_profiles/onescience_scenario.json` | `benchmarks/datasets/scenario` |

Profile 用于定义智能体、基座模型、数据集、并发数和有序评测阶段。运行时可通过 `--dataset_path` 选择其他用例 JSON 文件或数据集目录。

## 快速开始

### 1. 准备 Python 环境

请使用以下命令创建 Python 3.11 Conda 环境。仓库内置的 Profile 默认使用 `onescience311` 环境执行冒烟测试；如需使用其他环境，请同步修改所选 Profile 中的 `stages[].params.conda_env`。

```bash
conda create -n onescience311 python=3.11 -y
conda activate onescience311

# DCU 平台安装
pip install onescience[all-dcu] -i http://mirrors.onescience.ai:3141/pypi/simple/ --trusted-host mirrors.onescience.ai
```

请在仓库根目录运行 `parallel_eval.py`。如果使用基于 Python 的智能体适配器，还必须在安装了对应适配器依赖的同一个 Python 环境中运行该脚本。

### 2. 安装并配置智能体

只需安装当前 Profile 的 `agent_id` 所指定的智能体。

#### Codex

全局安装 Codex CLI，然后按照 [Codex CLI 官方文档](https://developers.openai.com/codex/cli/)完成身份验证和配置：

```bash
npm install -g @openai/codex
codex --version
```

在 Profile 中设置 `"agent_id": "codex"`。评测框架通过 `codex exec --ephemeral --json` 启动 Codex，并使用 `-m` 参数传入 `base_model`。

#### OpenScience

全局安装 OpenScience，并按照 OpenScience 软件包文档配置模型提供商和基座模型：

```bash
npm install -g @synsci/openscience
openscience --version
```

在 Profile 中设置 `"agent_id": "openscience"`。当前适配器以 `<provider>/<base_model>` 的形式调用所选模型，其中 `<provider>` 必须填写 OpenScience 配置文件 `openscience.jsonc` 中已配置的服务提供商名称；开始评测前，请确认该 provider 和基座模型均已完成配置。

#### Biomni

在启动评测所用的同一个 Python 环境中安装 Biomni 及其兼容 OpenAI 的 LangChain 依赖：

```bash
python -m pip install biomni langchain-openai

export BASE_URL="https://your-openai-compatible-endpoint/v1"
export API_KEY="your-api-key"
```

在 Profile 中设置 `"agent_id": "biomni"`。启动评测前，请通过环境变量 `BASE_URL` 和 `API_KEY` 提供 OpenAI 兼容服务的地址与访问密钥，Biomni 适配器会直接读取这两个变量。

请勿将 API Key 或其他凭据提交到 Profile 或结果文件中。

### 3. 选择智能体和模型

选择目标评测任务对应的 Profile，并设置其中的 `agent_id` 和 `base_model` 字段。`agent_id` 当前支持 `codex`、`openscience` 和 `biomni`。

```json
{
  "agent_id": "codex",
  "base_model": "your-model-id"
}
```

Profile 中的其余字段用于定义默认数据集、最大并发数、工作流、规则评测和评分阶段。请保留对应任务 Profile 中的其他字段。

### 4. 运行评测

运行 Profile 默认数据集中的所有用例：

```bash
python scripts/parallel_eval.py \
  --profile eval_profiles/onescience_scenario.json
```

使用 `benchmarks/datasets/` 下的单个用例或其他目录覆盖默认数据集：

```bash
python scripts/parallel_eval.py \
  --profile eval_profiles/onescience_scenario.json \
  --dataset_path benchmarks/datasets/scenario/anthropogenic_precip_fingerprint.json
```

`run_config.max_concurrency` 用于控制并发运行的用例 Worker 数量。验证新的智能体或模型配置时，建议先设置为 `1`。

### 5. 查看结果

结果按照智能体版本、基座模型、任务类型和用例分层存储：

```text
eval-records/
`-- <agent_id>-<agent_version>/
    `-- <base_model>/
        `-- <task_type>/
            |-- submissions/
            |   `-- <case_id>/
            |       `-- ... 生成的任务产物
            `-- eval_results/
                |-- summary.json
                `-- <case_id>/
                    |-- agent_result.json
                    |-- execute_task_result.json
                    |-- rule_eval.json
                    |-- score.json
                    `-- logs/
```

如果某个评测任务的 Profile 未启用评分阶段，则不会生成 `score.json`。智能体的 stdout、stderr、执行轨迹和冒烟测试日志均保存在对应用例的 `logs/` 目录中。

## 智能体支持

| 智能体 ID | 运行时入口 | 基座模型选择方式 |
|---|---|---|
| `codex` | `codex exec --ephemeral --json` | `-m <base_model>` |
| `openscience` | `openscience run --auto-approve --workspace project --format json` | `--model <provider>/<base_model>` |
| `biomni` | `scripts/biomni_runner.py` | `--model <base_model>` |

评测框架在启动评测前调用各智能体启动器的 `--version` 命令，自动获取运行时版本，并将该版本用于结果目录命名和结果元数据记录。对于 `openscience`，命令中的 `<provider>` 必须与 `openscience.jsonc` 中配置的服务提供商名称一致。

### 添加其他智能体

其他智能体需要在 `scripts/parallel_eval.py` 中实现适配。请在 `_agent_command()` 中添加对应的 `agent_id` 分支，并确保启动器满足以下要求：

1. 支持 `--version` 命令，且输出中包含点分版本号。
2. 接受 Worker 提示词作为最后一个命令行参数，或使用一个小型 Runner 脚本实现该调用约定。
3. 输出 `parallel_eval.py` 能够解析的结构化最终用例结果。
4. 智能体调用失败时返回非零退出码。
5. 保留 stdout 和 stderr，以便评测框架记录完整轨迹并报告失败原因。

`scripts/biomni_runner.py` 展示了如何将 Python 智能体库封装为符合该调用约定的 CLI。

## 评测方法

- **冷启动执行。** 每个用例都会启动一个全新的智能体进程。
- **产物隔离。** 每个用例只能写入为其分配的提交目录和评测结果目录。
- **有序阶段。** Profile 按声明顺序执行已启用的工作流阶段和 Python 阶段；阻断性失败会被记录，并按照配置跳过后续阶段。
- **最小验证。** 对生成的产物执行静态检查、导入检查、CLI 帮助检查、Dry Run 或其他受限的冒烟测试。真实训练循环不属于默认评测工作流。
- **规则评测。** `scripts/rule_eval.py` 加载用例指定的评测器，并针对生成的产物执行用例声明的检查。
- **评分。** 启用评分阶段时，`scripts/scoring.py` 按检查类型汇总结果，并写入归一化的用例分数。
- **用量统计。** `scripts/collect_agent_usage.py` 从保留的事件流中提取 Token 用量和智能体身份信息。

## 仓库结构

```text
OneScienceBench/
|-- README.md                       项目说明
|-- LICENSE                         MIT 许可证
|-- benchmarks/
|   |-- datasets/                   按任务类型组织的用例 JSON 文件
|   |-- evaluators/                 用例专用评测器和基础评测器
|   `-- standard/                   基准参考资源
|-- eval_profiles/                  评测任务和运行时配置
|-- references/
|   `-- execute_task.md             Worker 执行约定
|-- scripts/
|   |-- parallel_eval.py            调度器和智能体分发入口
|   |-- biomni_runner.py            Biomni CLI 适配器
|   |-- collect_agent_usage.py      Token 和身份信息提取
|   |-- rule_eval.py                有序确定性检查
|   `-- scoring.py                  单用例分数汇总
`-- eval-records/                   生成的提交、日志和结果
```

## 引用

```bibtex
@misc{onesciencebench2026,
  title  = {OneScienceBench: Evaluating AI Agents on Scientific Tasks},
  author = {OneScience Contributors},
  year   = {2026},
  url    = {https://gitee.com/onescience-ai/onescience-bench}
}
```

## 许可证

OneScienceBench 基于 [MIT 许可证](LICENSE)发布。
