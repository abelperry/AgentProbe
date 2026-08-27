# MTACIFBench

Multi-turn agentic-coding **instruction following**. One sandbox, one agent
conversation, N rounds in the same workspace. Each round is scored against that
round's constraint checklist — not against the feature it built.

## What is scored

The score comes **entirely from instruction following**. There is no build step,
no HTTP server, no functional checklist. Whether the software works is a
different question, measured by other benchmarks; this one isolates whether the
agent obeyed what it was told while building it.

Each constraint is decided one of two ways:

| Path | Share of this dataset | How |
|---|---|---|
| `validation_code` | 892 / 1853 (48%) | Dataset-supplied Python checker, run in a short-lived subprocess with a hard timeout |
| LLM judge | 961 / 1853 (52%) | One judge sandbox per round, scoring all the round's undecided constraints in a single prompt |

A checker that times out, crashes, cannot be resolved to an entry point, or
returns a non-bool **falls back to the judge**. It never scores the constraint 0
— a broken checker is our problem, not the model's.

## Metrics

| Metric | Definition |
|---|---|
| `IFSSR` | Tasks where every round passed / valid tasks |
| `IFISR` | Rounds passed / total rounds |
| `IFCSR` | Constraints satisfied / total constraints |

Only judgements without an error enter the denominator: infrastructure failures
show up as missing coverage (`success_count` < `total`), never as constraint
violations.

## Baseline

`glm-5.3` driving Claude Code 2.1.199, judged by `deepseek-v4-pro`, 19 of 20
tasks valid (one hit an API quota limit mid-inference):

| Metric | Score |
|---|---|
| IFCSR | 86.89 |
| IFISR | 21.37 |
| IFSSR | 0.00 |

By verdict source: deterministic checkers 84.0%, LLM judge 89.4%. CLI budget 32k
output / 16k thinking tokens, reasoning effort `max`.

Both the agent scaffold and the judge model move these numbers, so report them
alongside any score. `docs/mtacifbench-design.md` explains what the benchmark
measures and how the data is built.

## Multi-turn semantics

Constraints span rounds — "every reply must start with 喵～", "keep the naming
you used last round", and a later round may *forbid* what an earlier round
required. The whole task therefore runs in **one** agent conversation: the task
sets `SandboxSpec.keep_session=True`, so `ClaudeCodeAgent` resumes the same
session (`--resume <sid>`) from round 2 on instead of starting a fresh one. Scoring a round
against constraints the agent could no longer see would measure the harness, not
the agent.

The dataset's `system_prompt` is injected through
`SandboxSpec.append_system_prompt` → `claude --append-system-prompt`, so
benchmark-only instructions never touch contestant-owned files in the workspace.

## Rounds are independent

Each round's `instruction_following_checklist` is self-contained and is never
merged with or inherited from another round. The dataset really does replace
constraints between rounds (`verified_1`: round 0 requires ESLint to pass,
round 1 forbids running ESLint at all, round 2 requires it again).

## Artifacts

```
output/{exp}/mtacifbench/{agent}/{model}/
  infer/{qid}/
    traces/{session}.jsonl              # one accumulating session across rounds
    round_records.json
    workspace.tar.gz
    instruction_following/round_{id}/
      workspace_snapshot/               # the workspace as that round left it
      context.json                      # that round's operation flow (sliced)
      last_response.txt                 # that round's final reply
  eval/{qid}/instruction_following/round_{id}/
      judge_prompt.txt
      round_results.json
      attempt_{n}/traces/               # judge agent traces
  result/{qid}.json
```

The judge only ever writes under `eval/`. Inference artifacts stay a pure record
of what the model did, so a judge bug cannot corrupt them.

Per-round operation flow is sliced out of the shared session trace positionally
(everything after the messages earlier rounds already consumed). If that offset
goes stale — the CLI compacted or rewrote the session file — it falls back to
slicing after the last occurrence of that round's prompt, which is still scoped
to the round.

## Dataset

The questions are published on the Hugging Face Hub as
[**AbelNexux/mtacifbench**](https://huggingface.co/datasets/AbelNexux/mtacifbench);
`benchmarks/*/data/` is git-ignored, so a fresh clone has this benchmark's code
but none of its 20 tasks. Pull them with:

```bash
uv pip install huggingface_hub
python scripts/pull_benchmarks.py --org AbelNexux mtacifbench
```

That drops `questions.jsonl` into `benchmarks/mtacifbench/data/` next to the
`judge.yaml` already tracked here, which is all a run needs. Pin a revision with
`--revision <sha>` when you need results to be reproducible, and add
`--force` to re-download.

The dataset card documents every field, the metric definitions and the
constraint categories. Two things worth knowing before reading the data:

- Each round's `instruction_following_checklist` is **self-contained**.
  Constraints may be replaced or even reversed between rounds (`verified_1`
  requires ESLint in round 0, forbids it in round 1, requires it again in round
  2), so a round is only ever scored against its own checklist.
- Roughly half the constraints carry a `validation_code` — a Python checker run
  instead of the LLM judge. `validation.py` executes each one in a subprocess
  with a timeout, and a checker that times out, raises or returns a non-bool
  degrades to the judge rather than scoring the constraint 0.

### Container images

`docker` (the agent's sandbox) and `judge_docker` (the judge's sandbox) come from
the question data. Both images the dataset ships are public on Docker Hub and
pull anonymously, so a clone plus a dataset pull is enough to run:

| Field | Image | Compressed | Role |
|---|---|---|---|
| `docker` | `alexgshaw/break-filter-js-from-html:20251031` | 365 MB | web-dev sandbox the agent works in |
| `judge_docker` | `dayong657/playwright-mcp-base:0.1.0` | 376 MB | sandbox the LLM judge runs in |

```bash
docker pull alexgshaw/break-filter-js-from-html:20251031
docker pull dayong657/playwright-mcp-base:0.1.0
```

Both are `linux/amd64` only. To substitute your own, either edit the data or set
`MTACIF_JUDGE_IMAGE`, which supplies the judge image for data that omits the
field — there is deliberately no built-in default, so a missing image fails at
load time rather than halfway through a run.

### Rebuilding from a private export

If you maintain your own copy of the source questions,
`scripts/build_mtacifbench_dataset.py` converts an export into the strict shape
this benchmark loads:

```bash
python scripts/build_mtacifbench_dataset.py \
  --src /path/to/export/questions.jsonl \
  --judge-docker dayong657/playwright-mcp-base:0.1.0
```

The converter holds every tolerance for loose input and fails hard rather than
guessing — mismatched checklist/validation-code lengths, duplicate round ids or
task ids, and empty instructions are all errors, not warnings. Useful flags:
`--limit N` and `--only id1,id2` for smoke runs.

## Run

```bash
export ZHIPU_API_KEY=... GATEWAY_API_KEY=... SANDBOX_KEY=...
export OFFLINE_PACKAGE_DIR=/path/to/offline_package
uv run agentprobe -c examples/exp-mtacifbench.yaml -l debug
```

`OFFLINE_PACKAGE_DIR` is needed because `judge.yaml` sets `offline: true` — this
benchmark starts one judge container per round, and an online `npm i -g` at that
rate hits `ECONNRESET` (the judge image has no npm to fall back on either). Fill
the directory with two `npm pack` tarballs:

```bash
mkdir -p data/offline_package && cd data/offline_package
npm pack @anthropic-ai/claude-code-linux-x64@2.1.199        # glibc images
npm pack @anthropic-ai/claude-code-linux-x64-musl@2.1.199   # musl images only
```

The version has to match `agent.version` in `judge.yaml`. See
[the root README](../../README.md#benchmark-datasets-and-images) for what the
sandbox does with these at install time.
