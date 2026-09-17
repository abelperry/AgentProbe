# Implementing MTAC-IFBench

What the benchmark measures, its data shape and its constraint taxonomy are
documented by the dataset itself:
[thu-coai/MTAC-IFBench](https://huggingface.co/datasets/thu-coai/MTAC-IFBench),
and in the paper (arXiv:2609.14992). For how to run it here, see
[`benchmarks/mtacifbench/README.md`](../benchmarks/mtacifbench/README.md).

This file is the other thing: the handful of decisions a harness has to get
right, each of which fails *silently* when it is wrong. Every one cost us a run
to find.

## The workspace path handed to a checker must be absolute

3,586 of the constraints ship a Python checker, and nearly every one that looks
at the project opens with:

```python
if not os.path.exists(workspace_path):
    return False
```

Checkers run in a subprocess whose working directory is a scratch dir. Hand that
a path relative to a *different* directory and every such constraint fails, for
every model, no matter what was written.

This is the worst kind of bug because the result looks reasonable. Ours sat at
34–36% across four runs and did not move when we changed the model — which is
exactly what a real instruction-following weakness looks like. The tell was that
constraints checking the *reply text* passed 96% while constraints walking the
*workspace* passed 35%, and the only difference between the two groups was
whether they used `workspace_path`.

**A metric that does not move when you change the model is a bug in the harness,
not a property of the model.**

## Checkers run in a subprocess, with a timeout

Dataset-supplied code executes with whatever privileges the harness has. Running
it in-process is simpler and wrong twice over: one infinite loop in one checker
hangs the whole run, and the code can reach the results directory it is being
scored into. `validation.py` spawns a short-lived subprocess with a hard timeout.

## An unusable verdict is not a failing verdict

Three places where the easy thing corrupts the metric:

- **A checker that tells us nothing must not score 0.** Timeout, exception,
  unresolvable entry point, non-bool return — all fall back to the judge. A
  broken checker is the harness's problem, not the model's.
- **Unparseable judge output means "no verdict", never "pass".** Retry, then mark
  the round unresolved so the framework re-judges it.
- **Infrastructure failures belong in coverage, not in the score.** A dead
  container or an exhausted API quota makes the task invalid — reported as
  `success_count < total` — rather than a model that got things wrong.

The same rule governs the functional pass: a check that never ran scores `None`
and stays out of the denominator, so switching the pass off cannot read as the
product regressing.

## Score round N against round N's checklist and nothing else

Each round's checklist is self-contained, and the data deliberately reverses
instructions mid-session. From the task that used to be called `verified_1`:

| Round | ESLint instruction |
|---|---|
| 0 | run ESLint over your code and fix every error |
| 1 | **do not** run any ESLint check |
| 2 | you must run ESLint again and resolve all errors |

An agent that treats round 0's instruction as permanent fails round 1; one that
forgets it fails round 2. Merging checklists across rounds would make these
tasks unsatisfiable and would measure nothing.

Note that each round's checklist already *includes* the repository-policy
constraints in force at that turn — `repository_policy_checklist` is a prefix of
it, not an extra thing to score.

## One conversation, and the policy is a file

Constraints reference conversational history ("keep the naming you used last
round") so the whole task runs in one agent session: `keep_session=True`, which
makes Claude Code resume (`--resume`) and OpenCode continue (`--continue`) from
round 2 on.

`repository_policy` is a *file* — that is the dataset's definition of it. It is
written into the workspace, restored afterwards, and scrubbed from the snapshot
before scoring. Delivering it as a hidden system prompt instead changes what is
being measured: the agent is meant to be able to read it, and constraints refer
to it.

## The judge reads text the model wrote

Per-round evidence is the round's workspace snapshot, its final reply and its
operation flow — sliced out of the shared transcript positionally, with
tool-result payloads replaced by a placeholder and the user instruction excluded
(the judge is asked what the agent *did*, not what it was told).

All of that is model-controlled text arriving in a judge prompt, so the judge is
told out of band that it is evidence and not instructions, and the prompt names
the checklist as the only authority. Verdicts are keyed by block index, so a
judge that renumbers its blocks scores the wrong constraints; the requirement
text it echoes back is the only trace of that, and a mismatch is logged.
