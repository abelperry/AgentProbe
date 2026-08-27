# MTACIFBench design

**M**ulti-**T**urn **A**gentic **C**oding **I**nstruction **F**ollowing.

This document explains what the benchmark measures, why the data is shaped the
way it is, and how a run is scored. For how to obtain the data and launch a run,
see [`benchmarks/mtacifbench/README.md`](../benchmarks/mtacifbench/README.md).

## What it measures

Most coding benchmarks ask whether the agent produced working software.
MTACIFBench asks a question that is orthogonal to correctness: **while building
the software, did the agent obey the constraints it was given — and did it keep
obeying them as the conversation grew?**

An agent receives a sequence of feature requests in one workspace and one
conversation. Alongside each request comes a checklist of constraints it must
satisfy *in that round*. The constraints are not about whether the feature works:

| Kind | Example |
|---|---|
| Reply format | every reply must begin with 喵～ and end with 汪～ |
| Reply language | always answer in Chinese |
| Code content | every code file's first line must be `-*- coding: utf-8 -*-` |
| Code style | indent with exactly 2 spaces; no spaces around `=` |
| Code annotation | every code file needs an `AI-Assisted:` comment on its own line |
| Prohibition | no `console.log`, no `TODO`, no `while` loops |
| Workflow | plan with `TodoWrite` before editing; prefer Read/Write/Edit over bash |
| Quantity | exactly 3 arrow functions per file; class names of even length |

Individually these are easy. The benchmark is hard because there are ~13 of them
per round, they accumulate across up to 10 rounds, a round scores only if **every
one** holds, and — crucially — a later round may **contradict** an earlier one.

## Why constraints are per-round, not cumulative

Each round's checklist is self-contained. It is never merged with, or inherited
from, another round. That is the central design decision, and it exists because
the data intentionally reverses instructions mid-conversation. From `verified_1`:

| Round | ESLint instruction |
|---|---|
| 0 | run ESLint over your code and fix every error |
| 1 | **do not** run any ESLint check |
| 2 | you must run ESLint again and resolve all errors |

An agent that treats round 0's instruction as permanent fails round 1. One that
forgets it fails round 2. Only an agent tracking *the current state of the
instructions* passes both. Merging checklists across rounds would make these
tasks unsatisfiable and would measure nothing.

The practical consequence for implementers: score round *N* against round *N*'s
checklist and nothing else.

## Why one conversation

Constraints reference conversational history — "keep the naming you used last
round", "every reply must start with …". If each round ran in a fresh agent
session, the agent could not see what it had been told, and the benchmark would
be measuring the harness rather than the agent.

So the whole task runs in a single agent conversation inside a single container.
In AgentProbe this is `SandboxSpec.keep_session=True`, which stops the engine
rotating the session id and makes `ClaudeCodeAgent` resume the same session
(`--resume`) from round 2 on.

The per-task `system_prompt` — constraints that apply to every round — is
injected out of band (`--append-system-prompt`), never by writing into the
workspace. The workspace is the artifact under evaluation; putting benchmark
instructions in it would contaminate the thing being measured.

## Data design

One JSON object per task. The full field reference lives in the
[dataset card](https://huggingface.co/datasets/AbelNexux/mtacifbench); the
design-relevant parts:

```
task_id, docker, judge_docker, workspace_dir, system_prompt, description
rounds[]
  round_id, prompt
  instruction_following_checklist[]
    constraint, validation_code, tags, main_id, type_id
```

**Scale.** 20 tasks, 141 rounds (5–10 per task, median 7), 1,853 constraints
(9–19 per round, median 13).

**`validation_code` lives on the constraint.** 892 of the 1,853 constraints
(48%) ship their own Python checker. Storing each checker next to the constraint
it checks — rather than in a parallel array aligned by index — removes an entire
class of failure: two independent lists that silently drift apart and shift every
verdict by one. `scripts/build_mtacifbench_dataset.py` accepts a source export
that uses the parallel-array form, verifies the two agree, and then drops the
array.

**Deterministic where possible.** Splitting the checklist roughly in half is
deliberate. Constraints with a mechanical criterion ("first line must be X",
"exactly 3 arrow functions") get code, so they are immune to a judge model's
mood. Constraints requiring reading comprehension ("prefer specialised tools over
bash", "comments must all be in English") go to an LLM judge. In the reference
run the two paths scored 84.0% and 89.4% respectively — close enough that
neither dominates the metric.

## Scoring pipeline

For each round, in order:

1. **Snapshot.** After the round finishes, export the workspace and store it with
   that round's final reply and its operation flow. Constraints are scored
   against the state *that round* left behind, not the end state.
2. **Deterministic checks.** For every constraint carrying `validation_code`, run
   it in a short-lived subprocess with a timeout, passing the round's reply and
   an absolute path to the snapshot.
3. **Judge.** Batch the remaining constraints into one judge prompt and run it in
   its own container, with only that round's evidence.
4. **Merge.** Reassemble verdicts in checklist order. The round passes only if
   every constraint passed.

### Round evidence for the judge

The judge sees three things: the round's workspace snapshot, the round's final
reply, and the round's operation flow. The flow is sliced out of the shared
session transcript positionally — everything after the messages earlier rounds
already consumed — with the round's prompt as a fallback boundary if those
offsets go stale. Tool-result payloads are replaced with a fixed placeholder, and
the user instruction itself is excluded: the judge is asked what the agent *did*,
not what it was told.

### Failure handling that keeps scores honest

Three rules, each of which exists because the alternative silently corrupts the
metric:

- **A checker that tells us nothing must not score 0.** Timeout, exception,
  unresolvable entry point, non-bool return — all fall back to the judge. A
  broken checker is the harness's problem, not the model's.
- **Unparseable judge output means "no verdict", never "pass".** Retry, then mark
  the round unresolved so the framework re-judges it. The judge must also
  reproduce each requirement verbatim; the parser compares that against the
  trusted checklist, so a requirement forged inside the model's own output cannot
  be scored.
- **Infrastructure failures belong in coverage, not in the score.** A dead
  container or an exhausted API quota makes the task invalid — reported as
  `success_count < total` — rather than a model that got things wrong.

### An absolute path is load-bearing

Checkers run in a subprocess whose working directory is a scratch dir, and nearly
every workspace-walking checker opens with:

```python
if not os.path.exists(workspace_path):
    return False
```

Hand that a path relative to a *different* working directory and every such
constraint fails, for every model, no matter what was written. The snapshot path
must be absolutised before it crosses the process boundary. This is worth
stating because the failure is invisible: scores stay plausible, stay stable
across runs, and do not move when you change the model — which is exactly what a
real model weakness would look like.

## Metrics

| Metric | Definition |
|---|---|
| **IFCSR** | constraints satisfied / total constraints |
| **IFISR** | rounds where every constraint held / total rounds |
| **IFSSR** | tasks where every round passed / total tasks |

Nested and increasingly strict. IFCSR is the smooth signal; IFISR is
hypersensitive because ~13 constraints must hold simultaneously (at 85%
per-constraint, independent failures would give ~12% clean rounds); IFSSR
requires a flawless 5-to-10-round session and is currently 0 for every model
measured.

Only tasks that produced a verdict enter the denominators.

## Reference results

`glm-5.3` driving Claude Code 2.1.199, judged by `deepseek-v4-pro`. 19 of 20
tasks valid — one hit an API quota limit mid-inference and is excluded from the
score rather than counted as a failure.

| Metric | Score |
|---|---|
| IFCSR | 86.89 |
| IFISR | 21.37 |
| IFSSR | 0.00 |

| Verdict source | Constraints | Pass rate |
|---|---|---|
| Deterministic checker | 800 | 84.0% |
| LLM judge | 908 | 89.4% |

Agent configuration: 32k output tokens, 16k thinking tokens, reasoning effort
`max`, one conversation across all rounds.

Two caveats on reading this. First, 20 tasks is enough to expose systematic
instruction-following weaknesses but too small for fine-grained ranking between
close models. Second, the number is a property of *model plus scaffold plus
judge*: thinking budget and session handling both move it materially, and half
the constraints are scored by a judge model that is itself part of the
measurement. Report the whole configuration alongside any score.

## What is deliberately not measured

Whether the software works. There is no build step, no HTTP server, no functional
checklist — those belong to other benchmarks. A task where the agent produced a
broken page but obeyed every constraint scores well here, and that is the
intended behaviour: this benchmark isolates instruction following so it can be
read independently of coding ability.
