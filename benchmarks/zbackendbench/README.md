# ZBackendBench

This is the AgentProbe/OpenSandbox port of the original `ccbench` task.

Place real data under this benchmark's `data_dir` with:

- `questions.jsonl`
- `tasks/{qid}/environment/`
- `tasks/{qid}/tests/test.sh`

Inference exports `changes.patch` and a deterministic verifier score. Judge
replays the patch and applies a code-quality rubric only when the deterministic
score is non-zero.
