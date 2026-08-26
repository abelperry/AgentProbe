# ZFrontendBench

This is the AgentProbe/OpenSandbox port of the original `autoccbench` task.

Place real data under this benchmark's `data_dir` with:

- `questions.jsonl`
- optional task assets alongside the container image contents

Inference exports `workspace.tar.gz`. Judge extracts the workspace, detects
HTML/SVG/NPM projects, serves HTTP-mode projects, and evaluates each checklist
item with `ClaudeCodeAgent` plus the Playwright MCP config.
