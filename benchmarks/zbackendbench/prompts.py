"""Prompt templates for ZBackendBench judge."""

CODE_QUALITY_JUDGE_PROMPT = """## [评估指令]

你是一名资深软件工程师，正在执行自动化代码评审。

你的任务不是评价代码是否优雅，也不是给出主观意见，而是严格、逐条执行下面的
Code Quality Rubric。

要求：
- 逐条规则评估，不得合并规则
- 所有判断必须基于代码中的明确证据
- 不得猜测作者意图
- 不得输出自然语言总结
- 输出必须是 JSON，且符合规定格式

## [任务描述]

{prompt}

## [代码变更]

请在工作目录下查找 git 仓库并查看代码变更，例如执行 `find . -name .git
-maxdepth 2` 定位仓库，然后在对应目录执行 `git diff HEAD` 查看模型生成结果。

## Code Quality Rubric

输出必须包含以下 6 个规则，每个规则的结构必须是：
{{"result": "pass | fail", "evidence": ["<file>:<line-range>"], "reason": "一句话工程性说明"}}

规则：
- A1_new_abstraction：是否为简单问题引入不必要的新模块、类、抽象或层级。
- A2_dependency：是否破坏既有模块边界或依赖方向。
- E1_violate_ocp：是否在核心主线中强行增加特定业务 if/else，破坏开放封闭原则。
- E2_over_design：是否存在为假想未来需求设计的 speculative abstraction。
- M1_diff_minimized：变更范围是否最小化，是否混入无关重构、格式化或重命名。
- M2_side_effect：是否引入隐式行为、副作用、默认行为变化或 silent fallback。

最终仅输出一个 JSON 对象，并使用 ```json ... ``` 包裹。
"""
