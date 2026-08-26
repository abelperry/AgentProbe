# TerminalBench v2

Place real data under this benchmark's `data_dir` with:

- `questions.jsonl`
- `tasks/{qid}/environment/`
- `tasks/{qid}/tests/test.sh`

The task uploads `environment/` into the question workspace, runs the agent,
then uploads `tests/` to `/tests` and reads `/logs/verifier/reward.txt`.
