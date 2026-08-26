# MTACIFBench

Multi-turn agentic-coding **instruction following**. One sandbox, one agent
conversation, N rounds in the same workspace. Each round is scored against that
round's constraint checklist — not against the feature it built.

## What is scored

The score comes **entirely from instruction following**. There is no build step,
no HTTP server, no functional checklist. This matches the upstream chatglm-eval
`mtacifbench` behaviour under its shipped configuration, where
`function_checklist_eval_enabled` is `false` and the build / embedded-eval /
dependency stack is never reached.

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

## Multi-turn semantics

Constraints span rounds — "every reply must start with 喵～", "keep the naming
you used last round", and a later round may *forbid* what an earlier round
required. The whole task therefore runs in **one** agent conversation: the task
sets `SandboxSpec.keep_session=True`, so `ClaudeCodeAgent` resumes the same
session (`--resume <sid>`) from round 2 on instead of starting a fresh one. This
mirrors upstream, where axec appends `--continue` from the second round.

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

Not committed (`benchmarks/*/data/` is git-ignored). Build it from the upstream
chatglm-eval export:

```bash
python scripts/build_mtacifbench_dataset.py \
  --src /path/to/chatglm-eval/data/dataset/mtacifbench/questions.jsonl
```

The converter owns every tolerance for the upstream shape and fails hard rather
than guessing:

- `rounds[*].instruction` → `rounds[*].prompt`.
- Each constraint's `validation_code` is kept **on the constraint**. Upstream
  also ships a parallel `instruction_following_validation_codes[i][j]` array
  aligned by index; the converter asserts the two agree (they do, for all 1853
  constraints) and then drops the array, removing the whole class of
  index-misalignment bugs.
- `system_prompt_checklist` is dropped: it is a prefix of each round's own
  checklist and nothing scores it separately.
- Container images are absent upstream (hardcoded defaults) and are materialised
  into the output, so the dataset is self-describing. Override with
  `--infer-docker` / `--judge-docker`.

Useful flags: `--limit N` and `--only id1,id2` for smoke runs.

## Run

```bash
export ZHIPU_API_KEY=... GATEWAY_API_KEY=... SANDBOX_KEY=...
uv run agentprobe -c examples/exp-mtacifbench.yaml -l debug
```
