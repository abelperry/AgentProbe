# AGENTS.md

Orientation for coding agents working in this repository. Read this before
changing code; it records the contracts and the failure modes that are not
obvious from reading individual files.

## What this is

AgentProbe evaluates AI coding agents. Every inference and judge run happens
inside an isolated OpenSandbox container. One concurrent pipeline streams
inference into judging, and results are persisted per question so a rerun skips
work that already succeeded.

Python 3.12+, managed with `uv`.

## Layout

```
src/agent_probe/
  config.py                 # YAML config models, ${ENV_VAR} expansion
  core/
    models.py               # BaseQuestion / BaseInference / BaseJudgement, Error, ErrorCode
    task.py                 # BaseTask[Q, I, J] — the benchmark interface
    sandbox.py              # Sandbox + SandboxSpec (container lifecycle, hooks)
    agent.py                # BaseAgent interface
    adapter.py              # Question loading (local_jsonl)
    repo.py, factory.py     # persistence ports, composition root
  agents/                   # ClaudeCodeAgent, OpenClaw, mini-swe-agent
  executors/                # PipelineExecutor (inference -> queue -> judge)
  repos/                    # file-based JudgeRepo / MetricsRepo
benchmarks/<name>/          # one directory per benchmark (see below)
scripts/build_*_dataset.py  # upstream export -> strict questions.jsonl
tests/
examples/*.yaml             # experiment configs
```

`benchmarks/` is a **separate top-level package**, shipped alongside
`src/agent_probe` (see `[tool.hatch.build.targets.wheel]`). Benchmarks are
loaded by dotted path at runtime and the CLI prepends the working directory to
`sys.path`, so **always run from the repository root**.

## Commands

```bash
uv run agentprobe -c examples/experiment.yaml -l debug   # run an experiment
uv run pytest                                            # tests
uv run ruff check --fix . && uv run ruff format .         # lint / format
uv run mypy --strict src/                                 # type-check
```

`ruff` is configured with `line-length = 100` and `select = ["E","W","F","I","B","C4","UP"]`.
`mypy --strict` covers `src/` only, but new benchmark code is expected to pass it
too — run `uv run mypy --strict benchmarks/<name>` before submitting.

Some pre-existing failures in `tests/` and `ruff` findings predate current work.
**Establish the baseline before you change anything** (`git stash` your work, run
the gates, compare) and do not report someone else's failure as yours.

## The Q-I-J contract

Each benchmark is a `BaseTask[Q, I, J]` subclass parameterised by three pydantic
models. The concrete types are recovered at runtime from `__orig_bases__` by
`resolve_types()`, which is how the adapter knows how to deserialise
`questions.jsonl` with no extra configuration.

- **Declare concrete type parameters** (`BaseTask[MyQ, MyI, MyJ]`) or factory
  wiring fails.
- `Q.qid()` must return a unique, filesystem-safe identifier. It becomes a path
  component; validate it if it comes from data.
- **Task instances are shared across every question in their dataset.** Never
  store per-question state on `self` — keep it in closures over the `inference`
  call, or pass it through `EvalContext`.

Three methods:

```python
async def inference(self, question: Q, ctx: EvalContext) -> I
async def judge(self, question: Q, inference_result: I, ctx: EvalContext,
                prev_judgement: J | None = None) -> J
def collect_metrics(self, judgements: list[J]) -> tuple[dict[str, float], int]
```

`collect_metrics` returns `(scores, success_count)`; the denominator `total`
comes from the adapter's question count, not from the list you receive.

## Sandbox lifecycle

`Sandbox.run()` orchestrates: create container → `on_setup` → install agent →
loop(`run_prompt` → collect traces → `on_nextround`) → `on_complete` → kill.

- The client-side cap is `timeout_sec - 30`, so the orchestrator raises a clean
  `SANDBOX_TIMEOUT` before the server kills the container. `ClaudeCodeAgent`
  additionally reserves 600 s so `on_complete` still gets to export artifacts.
- `on_nextround` returns the next prompt, or `None` to stop. It is called after
  **every** round, including the last.
- Opt-in `SandboxSpec` fields worth knowing:
  - `keep_session` — do not rotate `session_id` between rounds, so the agent
    resumes one conversation. Required by any benchmark whose constraints span
    rounds. Off by default.
  - `append_system_prompt` — extra system-prompt text for the agent CLI. Use
    this instead of writing instructions into the workspace under evaluation.

## Error codes decide control flow

`ErrorCode` in `core/models.py`: **negative means transient/infra and
rerunnable**, positive is reserved for business-layer errors owned by
benchmarks.

`PipelineExecutor` resumes on these rules:

| Cached state | Behaviour |
|---|---|
| `judgement.error is None` | question skipped entirely |
| `inference.error is None`, judgement has an error | inference reused, **judge re-runs only** |
| otherwise | full re-inference |

Two consequences for benchmark authors:

- Validate your own inference output and set `inference.error` when artifacts are
  missing or incomplete. A half-finished run must never be scored — that is how
  a partial rollout silently becomes a low score.
- When scoring cannot be completed (judge crashed, output unparseable), set
  `judgement.error`. That keeps the expensive inference and re-runs only the
  cheap part. Use `prev_judgement` to reuse the parts that did succeed.

Only count rows without errors in metric denominators. Infrastructure failures
must show up as missing coverage (`success_count` < `total`), never as the model
getting an answer wrong.

## Adding a benchmark

```
benchmarks/my_bench/
  __init__.py
  models.py     # MyQuestion / MyInference / MyJudgement
  task.py       # MyTask(BaseTask[MyQuestion, MyInference, MyJudgement])
  README.md     # what is scored, metric definitions, artifact layout
  data/         # git-ignored: questions.jsonl, judge.yaml
```

1. **Keep the runtime models strict.** `questions.jsonl` should already be in
   the exact shape of your `Question` model, so a malformed row fails at load
   time with a clear message. Put every tolerance for upstream formats in
   `scripts/build_my_bench_dataset.py` and make that script fail hard rather
   than guess.
2. Register it in an experiment YAML under `datasets:` with `task_type` as a
   dotted path.
3. **No hardcoded container images or hostnames.** Images come from the question
   data with an environment-variable fallback; a baked-in private registry path
   makes the benchmark silently unusable elsewhere.
4. Judge artifacts belong under `eval/<qid>/`. Inference output stays a pure
   record of what the model did, so a judge bug can never corrupt it.
5. Write tests for the pure functions — parsers, metric aggregation, artifact
   validation. Container runs are not testable in CI; the logic around them is.

## Known failure modes

These were all found the hard way. Do not reintroduce them.

- **Prompts go on stdin, not argv.** Linux caps a single argument at 128 KiB
  (`MAX_ARG_STRLEN`); a judge prompt embedding a transcript exceeds it and dies
  with `Argument list too long` before the agent starts.
- **`pkill -f 'claude.*-p'` matches the shell that runs it**, because that
  shell's own command line contains the pattern text. Use `'[c]laude.*-p'`.
- **With `keep_session`, `SandboxResult.last_assistant` is the newest reply in
  the whole session, not the current round.** Derive a round's reply from that
  round's own trace slice; otherwise a round that produced nothing silently
  inherits the previous round's reply and is scored against it.
- **A dependency may collide with the `benchmarks` package name.** `pysbd`, for
  one, ships its own top-level `benchmarks` module, which shadows this repo's
  and breaks every benchmark import. Check before adding dependencies.
- **`tarfile` extraction must keep the `data` filter.** Passing a custom
  callable to `extractall(filter=...)` replaces it entirely, losing protection
  against absolute paths, `..` traversal and escaping symlinks — and the archive
  usually comes from a container whose contents the evaluated model controls.
  Chain to `tarfile.data_filter` and **drop** unsafe members; raising aborts the
  whole archive and discards every good file after the bad one.
- **Parse LLM-judge output fail-closed.** An unparseable verdict must mean "no
  verdict" (retry, then flag), never a pass. When a judge reproduces requirement
  text, compare it to the trusted checklist verbatim, so a requirement forged
  inside the evidence cannot be scored.
- **Untrusted evidence in a prompt needs a fence longer than any backtick run it
  contains**, or the model's own output closes the block and escapes the
  surrounding instructions.
- **A subprocess that changes cwd must be handed absolute paths.** Experiment
  configs default to a relative `output_dir` (`./output`), so every artifact path
  derived from it is relative too. Hand one of those to a helper process running
  in a scratch directory and it silently resolves to nothing — for
  dataset-supplied checkers that meant `os.path.exists(workspace_path)` was
  false every time, so an entire class of constraints scored 0 regardless of what
  the model wrote. It cost ~17 points of benchmark score and looked exactly like
  a model weakness: stable across runs, immune to model and prompt changes.
  **A metric that does not move when you change the model is a bug in the
  harness, not a property of the model.**
- **`ModelConfig` fields only take effect if an agent reads them.** Check that
  the value actually reaches the process (inspect `/proc/<pid>/environ` in the
  container) before concluding a setting had no effect.

## Data and images

No benchmark's questions are committed; `benchmarks/*/data/` is git-ignored.
Container images are deployment-specific and have no built-in defaults. See the
environment-variable table in `README.md`. Keep deployment values in a launcher
script — `run.sh` and `run-*.sh` are git-ignored for that reason.
