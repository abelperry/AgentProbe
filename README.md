# AgentProbe

A general-purpose, extensible framework for evaluating AI agents in sandboxed environments.

## Features

- **Sandbox-native** -- Every inference and judge run executes inside an isolated [OpenSandbox](https://github.com/alibaba/OpenSandbox.git) container, ensuring reproducibility and security.
- **Lifecycle hooks** -- `SandboxSpec` exposes `on_setup`, `on_complete`, and `on_nextround` hooks, enabling agent-as-judge (run a separate agent to score results), multi-round interactions (iterative prompting until convergence), and custom environment preparation -- all without modifying the core engine.
- **Q-I-J paradigm** -- Each benchmark defines strongly-typed `Question`, `Inference`, and `Judgement` models. Types are automatically resolved from the task's generic parameters -- no redundant config.
- **Separation of concerns** -- Core abstractions (`BaseAdapter`, `BaseAgent`, `JudgeRepo`, `MetricsRepo`) are defined as interfaces. Each has pluggable implementations: swap file-based repos for a database, add new data adapters, or integrate new agents -- all by implementing the interface, with zero changes to the pipeline.
- **High concurrency & automatic resume** -- Evaluations run concurrently across the full cartesian product of models x datasets x agents, with parallelism at the individual question level and configurable concurrency limits. Results are persisted per-question; re-running an experiment skips completed items and retries only failures.

## Architecture

```
src/agent_probe/
  config.py              # YAML config models
  core/
    models.py            # Base Q-I-J types, JudgeResult, MetricsRecord
    task.py              # BaseTask[Q, I, J] generic interface
    executor.py          # BaseTaskExecutor, EvalUnit, EvalContext
    sandbox.py           # Sandbox engine (OpenSandbox SDK wrapper)
    adapter.py           # Data loading (LocalJsonlAdapter, registry)
    repo.py              # JudgeRepo / MetricsRepo abstract interfaces
    agent.py             # BaseAgent abstract interface
    factory.py           # ExperimentFactory (composition root)
  agents/
    claude_code.py       # ClaudeCodeAgent implementation
  executors/
    pipeline_executor.py # Concurrent inference -> judge pipeline
  repos/
    file_judge_repo.py   # File-based JudgeRepo
    file_metrics_repo.py # File-based MetricsRepo
  cli/
    main.py              # CLI entry point

docs/
  mtacifbench-design.md  # what MTACIFBench measures and how it is scored

benchmarks/              # Benchmark definitions (outside src/)
  mtacifbench/
    models.py            # MTACIFBenchQuestion, Inference, Judgement
    task.py              # MTACIFBenchTask (multi-turn, one conversation)
    validation.py        # dataset-supplied deterministic checkers
    README.md            # scoring, metrics, artifacts, dataset
  zbackendbench/
    models.py            # ZBackendBenchQuestion, Inference, Judgement
    task.py              # ZBackendBenchTask
    data/                # 2 sample tasks — the only question data in this repo
      questions.jsonl
      judge.yaml
      tasks/{challenge_003,dioxus_2}/   # environment + test scripts
  terminalbench_v2/
    models.py            # TerminalBenchV2Question, Inference, Judgement
    task.py              # TerminalBenchV2Task
  zfrontendbench/
    models.py            # ZFrontendBenchQuestion, Inference, Judgement
    task.py              # ZFrontendBenchTask
    data/
      questions.jsonl    # Question dataset
      judge.yaml         # Judge agent config
```

## Quick start

The repo ships one runnable example — two `zbackendbench` tasks whose question
data, judge config and test scripts are tracked here, and whose container images
are public on Docker Hub. Nothing else has to be downloaded first.

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for the OpenSandbox server)

### 1. Install dependencies

```bash
git clone <repo-url> && cd AgentProbe
uv sync
```

### 2. Start the OpenSandbox server

Every inference and judge run executes inside a sandbox, so this has to be up
before any evaluation:

```bash
uv pip install opensandbox-server
opensandbox-server init-config .sandbox.toml --example docker
opensandbox-server --config .sandbox.toml
```

### 3. Run the example

```bash
export ZHIPU_API_KEY="your-api-key"       # referenced by examples/experiment.yaml

uv run agentprobe -c examples/experiment.yaml -l info
```

`examples/experiment.yaml` evaluates `glm-5` on the two bundled tasks
(`challenge_003`, `dioxus_2`) with the Claude Code agent. Each task is scored by
running its test script inside the container, so a pass here means the whole
pipeline — sandbox, agent install, inference, verification, metrics — works.

If your OpenSandbox server requires authentication, add `api_key: "${SANDBOX_KEY}"`
under `sandbox:` and export `SANDBOX_KEY`; a local server started as above needs
neither.

Results land under `output/`:

```
output/{experiment}/{dataset}/{agent}/{model}/
  result/{qid}.json     # per-question JudgeResult
  infer/{qid}/traces/   # agent trace files
output/{experiment}/{dataset}/
  metrics.jsonl         # aggregated metrics
```

Re-running the same command skips questions that already have a valid result and
retries only the failures.

## Running a full benchmark

The quick start works out of the box because its data is small enough to track
in git. Every other benchmark needs three things supplied before you can run it,
plus a config that ties them together. Do them in this order.

Benchmark **code** lives here; benchmark **data** does not. Every
`benchmarks/<bench>/data/` directory is git-ignored (the two zbackendbench tasks
above are the one whitelisted exception):

```
benchmarks/<bench>/
  models.py, task.py, ...   # in this repo
  data/                     # NOT in this repo — you supply it
    questions.jsonl         # one JSON object per line
    judge.yaml              # agent-as-judge config, where the bench uses one
```

### 1. Question data

Datasets are published one Hugging Face dataset repo per benchmark, and each
repo maps 1:1 into `benchmarks/<bench>/data/`:

```bash
uv pip install huggingface_hub

# pull one benchmark
python scripts/pull_benchmarks.py --org AbelNexux mtacifbench

# or every benchmark hosted by an org/user
python scripts/pull_benchmarks.py --org your-org

# or point a single benchmark at any repo id
python scripts/pull_benchmarks.py --repo swebench=princeton-nlp/SWE-bench_Verified
```

| Benchmark | Dataset | Status |
|---|---|---|
| `mtacifbench` | [AbelNexux/mtacifbench](https://huggingface.co/datasets/AbelNexux/mtacifbench) | published |
| `zbackendbench` | 2 sample tasks tracked in this repo | full set not published |
| others | — | not published yet |

Pass `--revision <sha>` to pin a dataset when results need to be reproducible.
For a benchmark whose data is not published, point `datasets.<name>.data_dir` at
wherever you keep it, or drop a `questions.jsonl` into
`benchmarks/<bench>/data/`. The expected shape is that benchmark's `Question`
model in `models.py` — strict, so a malformed row fails loudly at load time
rather than mid-run. Where a converter exists (`scripts/build_*_dataset.py`), it
turns a looser export into that shape and validates it.

### 2. Container images

Both the inference image and, for benchmarks that use an agent-as-judge, the
judge image are read from the question data (`docker` / `judge_docker`) with
environment-variable fallbacks. There are deliberately **no built-in defaults** —
a hardcoded registry path would make the benchmark silently unusable outside the
network it was written in.

The published datasets reference public images, so `docker pull` needs no
credentials. If yours live in a private registry, make sure the OpenSandbox host
is logged in to it before starting a run.

### 3. Offline agent packages

Only needed for configs with `offline: true`. These install the Claude Code CLI
from a host directory instead of running `npm i -g` inside the sandbox.
MTACIFBench's judge config uses this: it starts one judge container per round, an
online install at that rate hits `ECONNRESET`, and the judge image has no npm to
fall back on.

Populate the directory with `npm pack`, which downloads a package's tarball
without installing it. The filenames it produces are exactly what the installer
looks for:

```bash
mkdir -p data/offline_package && cd data/offline_package

# glibc images (almost certainly what you want)
npm pack @anthropic-ai/claude-code-linux-x64@2.1.199

# musl images (Alpine) — only if your sandbox image is one
npm pack @anthropic-ai/claude-code-linux-x64-musl@2.1.199

export OFFLINE_PACKAGE_DIR=$PWD   # ~148 MB for both
```

The version must match `agent.version` in the config that will use it — the
shipped `benchmarks/mtacifbench/data/judge.yaml` pins `2.1.199`. AgentProbe
mounts the directory read-only at `/mnt/offline_package` in every sandbox, picks
`x64` or `x64-musl` by testing for `/lib/ld-musl-x86_64.so.1`, extracts
`package/claude` from the matching tarball and puts it on `PATH`. A missing
archive fails immediately with the exact path it looked for.

Nothing else in the directory is read: the CLI ships as a self-contained native
binary, so no Node runtime and no `@anthropic-ai/claude-code` wrapper package are
needed. If your sandboxes can reach the npm registry, drop `offline: true` and
skip all of this.

### 4. Write the experiment config

An experiment config is a YAML file describing what to run against what. The
shipped `examples/*.yaml` are working references; `examples/experiment.yaml` is
the smallest one.

```yaml
name: "demo-experiment"      # names the output directory
concurrency: 5               # questions evaluated in parallel
output_dir: "./output"

sandbox:
  host: "localhost:8080"     # the OpenSandbox server from step 2
  api_key: "${SANDBOX_KEY}"  # omit if your server does not require one
  request_timeout: 600

models:                      # key = model name sent to the API
  glm-5:
    base_url: "https://open.bigmodel.cn/api/anthropic/"
    api_key: "${ZHIPU_API_KEY}"
    format: "anthropic"      # "anthropic" | "openai" (default)
    timeout: 10800
    max_tokens: 32000

datasets:                    # key = dataset id, used in output paths
  zbackendbench:
    name: "zbackendbench"
    adapter_type: "local_jsonl"
    data_dir: "benchmarks/zbackendbench/data"
    task_type: "benchmarks.zbackendbench.task.ZBackendBenchTask"
    judge_config_path: "benchmarks/zbackendbench/data/judge.yaml"

agents:                      # key = agent id, used in output paths
  claude_code:
    type: "agent_probe.agents.claude_code.ClaudeCodeAgent"
```

Three things are worth knowing before you write your own:

**It runs the full cartesian product.** Every model × every dataset × every
agent becomes a run, and results are keyed by all three. Listing two models and
one dataset evaluates that dataset twice, with no extra plumbing.

**`task_type` is a dotted import path**, resolved at startup. It points at a
`BaseTask[Q, I, J]` subclass, and the Q/I/J types come from its generic
parameters — the config never repeats them.

**`${VAR}` is expanded from the environment** when the file loads, so no
credential is ever written into a config. A referenced variable that is not set
raises immediately, naming the variable — not halfway through a run.

For benchmarks that score with an agent-as-judge, `judge_config_path` points at a
second YAML holding the judge's own `model:` and `agent:` blocks (see
`benchmarks/mtacifbench/data/judge.yaml`). It can also be a mapping from eval
method to path when a benchmark scores several ways.

Variables the shipped configs expect:

| Variable | Used for |
|---|---|
| `GATEWAY_BASE_URL` | Base URL of your model gateway (`examples/*.yaml`, judge configs) |
| `OFFLINE_PACKAGE_DIR` | Host dir holding the `npm pack` tarballs from step 3 |
| `HTTP_PROXY_URL`, `NO_PROXY_HOSTS` | Egress proxy for sandboxes that need one |
| `MTACIF_JUDGE_IMAGE`, `MRCC_JUDGE_IMAGE`, `ZFRONT_JUDGE_IMAGE`, `DEFAULT_PLAYWRIGHT_IMAGE`, `DEFAULT_INFER_IMAGE`, `DEFAULT_EVAL_IMAGE` | Fallback images when the data omits them |
| `SWEBENCH_PRO_IMAGE_REPO` | Registry holding the SWE-bench Pro instance images |
| `EXTRACT_API_BASE_URL` | Score-extraction endpoint (mrccbench) |
| `SANDBOX_KEY` | OpenSandbox server API key |
| `ZHIPU_API_KEY`, `GATEWAY_API_KEY`, `GLM_API_KEY`, `DEEPSEEK_API_KEY` | Model credentials |

Keep these in a launcher script — `run.sh` and `run-*.sh` are git-ignored for
exactly this reason.

### 5. Run

```bash
uv run agentprobe -c examples/exp-mtacifbench.yaml -l info
```


## Creating a New Benchmark

### 1. Create the benchmark directory

```
benchmarks/
  my_bench/
    __init__.py
    models.py
    task.py
    data/
      questions.jsonl
      judge.yaml        # optional, for agent-as-judge
```

### 2. Define Q-I-J models

In `models.py`, define your strongly-typed data models:

```python
from agent_probe.core.models import BaseQuestion, BaseInference, BaseJudgement


class MyQuestion(BaseQuestion):
    prompt: str
    expected: str
    image: str            # sandbox container image


class MyInference(BaseInference):
    response: str


class MyJudgement(BaseJudgement):
    score: float
    passed: bool
    reason: str = ""
```

### 3. Implement the task

In `task.py`, subclass `BaseTask` with your Q-I-J types:

```python
from agent_probe.core.task import BaseTask
from agent_probe.core.sandbox import SandboxSpec
from .models import MyQuestion, MyInference, MyJudgement


class MyTask(BaseTask[MyQuestion, MyInference, MyJudgement]):

    async def inference(self, question, ctx):
        spec = SandboxSpec(
            image=question.image,
            prompt=question.prompt,
            agent_config=ctx.agent_config,
            model_cfg=ctx.model_config,
            output_dir=str(ctx.output_dir / "infer" / question.id),
        )
        result = await self.sandbox.run(spec)
        # Parse result into MyInference
        return MyInference(response=result.last.stdout if result.last else "")

    async def judge(self, question, inference_result, ctx):
        # Implement scoring logic
        passed = inference_result.response.strip() == question.expected.strip()
        return MyJudgement(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="exact match",
        )

    def collect_metrics(self, judgements):
        total = len(judgements) or 1
        passed = sum(1 for j in judgements if j.passed)
        return {"accuracy": passed / total}
```

### 4. Prepare question data

Create `data/questions.jsonl` with one JSON object per line. Each line **must** have an `id` field:

```jsonl
{"id": "q001", "prompt": "Write a hello world program", "expected": "Hello, World!", "image": "opensandbox/code-interpreter:v1.0.2"}
{"id": "q002", "prompt": "Calculate 2+2", "expected": "4", "image": "opensandbox/code-interpreter:v1.0.2"}
```

### 5. Add to experiment config

Add the dataset to a config file — see
[Write the experiment config](#4-write-the-experiment-config) for the full
anatomy. The dataset block is the only part specific to a new benchmark:

```yaml
datasets:
  my_bench:
    name: "my_bench"
    adapter_type: "local_jsonl"
    data_dir: "benchmarks/my_bench/data"
    task_type: "benchmarks.my_bench.task.MyTask"
    # judge_config_path: "benchmarks/my_bench/data/judge.yaml"   # agent-as-judge only
```

### 6. Run

```bash
export MY_API_KEY="your-api-key"
uv run agentprobe -c examples/my_experiment.yaml -l info
```

Results are written to:

```
output/{experiment}/{dataset}/{agent}/{model}/
  result/{qid}.json     # per-question JudgeResult
  infer/{qid}/traces/   # agent trace files
output/{experiment}/{dataset}/
  metrics.jsonl          # aggregated metrics
```

## License

Apache-2.0
