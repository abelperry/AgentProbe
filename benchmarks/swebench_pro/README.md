# SWE-bench Pro

AgentProbe port of ScaleAI SWE-bench Pro.

Expected `data_dir` layout:

- `questions.jsonl`

Each JSONL row must include the SWE-Pro instance fields plus prepared eval
assets:

- `dockerhub_tag`
- `base_commit`
- `prompt`
- `fail_to_pass`
- `pass_to_pass`
- `eval_cmd`
- `selected_test_files`
- `run_script`
- `parser_py`
- `env_exports`

The task runs the agent in `jefzda/sweap-images:{dockerhub_tag}` at `/app`,
resets the repo to `base_commit` before the agent starts, hides post-base git
refs, then replays the public SWE-Pro eval flow after inference.

Use `SWEBENCH_PRO_IMAGE_REPO` or dataset option `image_repo` to point at an
internal image mirror.

Generate the JSONL with:

```bash
python scripts/build_swebench_pro_dataset.py \
  --repo-dir /path/to/SWE-bench_Pro-os \
  --output-dir benchmarks/swebench_pro/data
```
