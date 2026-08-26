# MRCCBench

MRCCBench is a multi-round coding benchmark. A task contains ordered requirement
rounds, all executed in the same workspace. After each agent round, AgentProbe
runs a dependency check and may send a repair prompt before moving to the next
main round. Final scoring is checklist-based and uses a Playwright-capable judge.

## Data

AgentProbe consumes a clean `questions.jsonl` shape:

- `task_id`
- `docker`
- `workspace_dir`
- `description`
- `rounds`: list of `{round_id, prompt, scenario_tags, tags}`
- `checklist`: list of `{id, description, weight}`
- `dependencies`: list of `{round_id, critical_check}`
- runtime options such as `test_mode`, `judge_docker`, `http_port`, timeouts,
  `repair`, and `max_repair_attempts`

Convert chatglm-eval MRCCBench data with:

```bash
python scripts/build_mrccbench_dataset.py \
  --input-jsonl /path/to/chatglm-eval/data/dataset/<set>/questions.jsonl \
  --output-dir benchmarks/mrccbench/data
```

## Run

```bash
agentprobe run examples/exp-mrccbench.yaml
```

The judge config is `benchmarks/mrccbench/data/judge.yaml`.
