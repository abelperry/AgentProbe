# MTAC-IFBench: Benchmarking Instruction-Following in Multi-Turn Agentic Coding

## 🌟 Overview

**MTAC-IFBench** benchmarks **instruction following in multi-turn agentic coding**.

Existing agentic coding benchmarks (e.g., SWE-bench, Terminal-Bench) focus on **final functional correctness**, while current instruction-following benchmarks confine themselves to **single-turn** chat or code generation. Neither answers the question that matters in a real development session: **does the agent keep following the rules, turn after turn, as the requirements change?**

In real multi-turn software development, an agent must comply with:
- Repository policy files (`CLAUDE.md`, `AGENTS.md`) that govern the whole session
- Per-turn constraints on the response, the code, the environment, and its own workflow
- Constraints that persist across turns without decaying as context grows
- Constraints that are added, revised, or overridden by later instructions

Each instance in MTAC-IFBench is a complete development session: a repository policy file, 5–10 progressive user instructions, a constraint checklist for every turn, and a function checklist for the finished project. Each checklist item is verified by a verification script or a judge agent otherwise. MTAC-IFBench identifies significant deficiencies in existing code agents in multi-turn instruction-following, with performance degrading rapidly as the interaction session grows longer. See our paper for full results.

## 📊 Data statistics

| Metric | Value |
|---|---|
| Instances | 100 (`full`) / 20 (`lite`) |
| Constraints | 9,133 |
| Avg. turns per instance | 7.04 (sequences span 5 to 10 turns) |
| Avg. constraint checklist items | 91.33 per instance, 12.97 per turn |
| Avg. function checklist items | 16.07 per instance |

## 📦 Constraint taxonomy

Constraints in MTAC-IFBench span 6 primary and 18 secondary categories, covering the generated response and code as well as the agent's environment interactions and workflow.

| Primary Category | Secondary Categories | Description | Example |
|---|---|---|---|
| Content | Keyword, Persona, Format | Lexical elements, adopted persona, and structural templates in the generated code or responses | *Every response must end with the modal particle "meow~".* |
| Language | Response Language, Comment Language, File Encoding | The language used for responses and comments, and the character encoding of created files | *All comments in every code file you generate must be in Chinese.* |
| Quantity | Range, Exact Value, Complexity | Quantitative attributes of generated artifacts, as ranges, exact values, or complexity caps | *Each file you generate must contain between 200 and 500 lines.* |
| Style | Layout, Naming, Paradigm | Spatial organization, naming conventions, and the programming style used in implementation | *Every function name in the code files you generate must follow snake case.* |
| Environment | File Path, File Operation, Logging | How files are referenced, placed, and modified, and what record the agent leaves of its work | *Before modifying an existing file each time, you must create a backup of the original file in the same directory.* |
| Workflow | Tool Usage, Orchestration, Testing | Tool selection and sequencing of multi-step actions, and code verification practices | *You must execute multiple independent tool calls in parallel as much as possible to improve efficiency.* |


## ⚙️ Data format

Each line is one instance, a JSON object with the following fields:

```json
{
  "task_id": 1,
  "task_category": "Data Analysis",
  "repository_policy": "# 项目规范\n所有的回复内容都应以语气词\"喵～\"起头 ...",
  "repository_policy_checklist": [
    {
      "constraint": "所有的回复内容都应以语气词\"喵～\"起头",
      "validation_code": "def check_requirement(response: str, workspace_path: str) -> bool:\n    ...",
      "tags": ["Content", "Persona"]
    },
    ...
  ],
  "rounds": [
    {
      "round_id": 0,
      "instruction": "帮我写一个监控表格页面的基础框架。页面需要一个大标题 ...",
      "instruction_following_checklist": [
        {
          "constraint": "...", 
          "validation_code": "...", 
          "tags": ["Style", "Layout"]
        },
        ...
      ]
    },
    ...
  ],
  "function_checklist": [
    "查看页面整体结构，确认页面顶部包含大标题 ...",
    ...
  ]
}
```

| Field | Description |
|---|---|
| `task_id` | Instance id (1–100) |
| `task_category` | Development domain: Frontend Development, Data Analysis, Application Development, UI/UX Optimization, Build & Deployment, or Machine Learning |
| `repository_policy` | Repository policy file content (e.g. `CLAUDE.md` / `AGENTS.md`), imposing global constraints over the whole session |
| `repository_policy_checklist` | Constraint checklist for the repository policy file; applies to every turn |
| `rounds` | The multi-turn instruction sequence. Each entry carries a `round_id`, the user `instruction` for that turn, and an `instruction_following_checklist` holding **all** constraints in force at that turn, including constraints from the repository_policy file |
| `function_checklist` | Functional requirements for the **final** project |


## 🚀 Usage

Evaluation uses [AgentProbe](https://github.com/abelperry/AgentProbe), a sandbox framework for coding-agent assessment.

**1. Set up**

```bash
git clone https://github.com/abelperry/AgentProbe.git && cd AgentProbe
uv sync
./scripts/init.sh && source .agentprobe-env

uv pip install huggingface_hub
python scripts/pull_benchmarks.py mtacifbench            # 100 tasks
python scripts/pull_benchmarks.py mtacifbench --split lite   # or the 20-task subset
```

That leaves a runnable `benchmarks/mtacifbench/data/`: the chosen split placed as
`questions.jsonl`, which is what the adapter reads, alongside the two judge
configs tracked in this repo. They mirror the dataset's own `eval_config/` apart
from the agent version: that copy pins `2.1.14`, which cannot be installed
offline because npm has no `@anthropic-ai/claude-code-linux-x64@2.1.14` — the
platform builds start later. `scripts/init.sh` fetches `2.1.199`.

`judge.yaml` scores instruction-following only; point the dataset's
`judge_config_path` at `judge_if_function.yaml` to also build the final project
and check its function checklist.

**2. Configure the agent**

In `examples/exp-mtacifbench.yaml`, `models:` is the LLM to be evaluated and `agents:` is the harness driving it.

```yaml
models:
  your-model:
    base_url: "${GATEWAY_BASE_URL}"
    api_key: "${GATEWAY_API_KEY}"
    model_name: "your-model"
    format: "anthropic"

agents:
  claude_code:
    type: "agent_probe.agents.claude_code.ClaudeCodeAgent"
    version: "2.1.199"
    offline: true
    offline_package_dir: ${OFFLINE_PACKAGE_DIR}
  # opencode:
  #   type: "agent_probe.agents.opencode.OpenCodeAgent"
  #   version: "1.1.21"
  #   params: {output_format: "json"}
```

Every agent listed runs against every model listed, so you can uncomment `opencode` to compare one model across both harnesses.

**3. Start evaluation**

```bash
export GATEWAY_BASE_URL=... GATEWAY_API_KEY=...
uv run agentprobe -c examples/exp-mtacifbench.yaml -l info
```

Results land under `output/{experiment}/{dataset}/{agent}/{model}/`, with aggregated metrics in `metrics.jsonl`.

### `models.*.timeout` is per round, and gets multiplied

A task runs 5 to 10 rounds in one sandbox, so the sandbox has to outlive all of
them: `timeout_sec = timeout * rounds + build grace`, while `timeout` itself
bounds a single round. That product is checked against the OpenSandbox server's
`max_sandbox_timeout_seconds` (86400 by default), so a value that looks
reasonable per round fails at sandbox creation on the longest tasks:

| `timeout` | 10-round budget | Creates? |
|---|---|---|
| 37800 | 378900 | no |
| 10800 (the default) | 108900 | no |
| 7200 | 72900 | yes |

The failure is `Create sandbox failed: Sandbox timeout ... exceeds configured
maximum`, recorded as a rerunnable error rather than a score — so it shows up as
missing coverage, not as a bad model. Keep `timeout` at or below
`(max_sandbox_timeout_seconds - 900) / 10`.

## 👏 Citation

```bibtex
@article{wen2026mtacifbench,
  title   = {MTAC-IFBench: Benchmarking Instruction-Following in Multi-Turn Agentic Coding},
  author  = {Wen, Bosi and Wang, Cunxiang and Gui, Jiayi and Zhang, Haoke and
             Niu, Yilin and Ke, Pei and Yang, Dayong and Wang, Hongning and Huang, Minlie},
  journal = {arXiv preprint arXiv:2609.14992},
  year    = {2026}
}
```
Please kindly cite our paper if this paper and the codes are helpful.