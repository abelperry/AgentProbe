"""Tests for PairwiseJudgeStrategy._parse_pairwise_output — pure function, data-driven."""

from __future__ import annotations

import pytest

from benchmarks.zclawbench.eval_strategy.pairwise_judge import PairwiseJudgeStrategy

TARGET = "claude-sonnet-4-6"
BASELINE = "baseline-model"

_parse = PairwiseJudgeStrategy._parse_pairwise_output

# Real stdout tail from zcb_014 eval trace (double-backslash + \text wrapper).
_REAL_TRACE_ZCB_014 = (
    "claude-sonnet-4-6 在\"可访问链接\"这一用户核心诉求上做到了100%，"
    "且所有论文均为2025年发布；baseline-model 虽然时间分布更广、"
    "论文聚焦度更贴合\"综合能力\"，但有1篇没有具体可访问的arxiv链接，"
    "1篇是2024年论文，存在事实性瑕疵。\n\n"
    "$$\\\\boxed{[[\\\\text{claude-sonnet-4-6}]]}$$"
)
_REAL_TRACE_ZCB_014_1 = (
    """
    +GAmCmc0CDGI/oLPdXbxxbDy4i5/Gg7f6k5aRgB"},{"type":"text","text":"现在我已经有了足够的信息进行完整评估。\n\n---\n\n## 评估分析\n\n### claude-sonnet-4-6 的表现\n\n该模型尝试了大量工具调用：\n- 两次 `web_search` 调用均以 `\"This operation was aborted\"` 错误结束\n- 尝试 `web_fetch` arXiv，被阻断（私有 IP 错误）\n- 用 `curl` 访问 arXiv legacy API，已停用\n- 多次 arXiv API 查询，搜索返回的都是无关领域的论文\n- Semantic Scholar API 返回 429 限流错误\n- 最后一条消息（Line 55）**仅包含 thinking 内容，没有任何文本输出**\n\n**结论：模型在多次工具调用后，始终未能向用户给出最终答案，任务完全失败。**\n\n---\n\n### baseline-model 的表现\n\n该模型历经更多轮次的工具调用（94行记录），最终成功产出了一份完整答案：\n\n- 提供了 **7 篇代表性多语言综合能力 Benchmark 论文**\n- 经验证，6篇论文的 arXiv ID 和标题完全正确：\n  - `2502.04688` M-IFEval ✅\n  - `2502.07346` BenchMAX ✅\n  - `2503.10497` MMLU-ProX ✅\n  - `2507.11882` Marco-Bench-MIF ✅\n  - `2507.17476` MultiNRC ✅\n  - `2412.03304` Global MMLU ✅（但该论文投稿于 2024年12月，模型已如实标注）\n- 第7篇（AI Language Proficiency Monitor）未给出具体 arXiv ID，仅提供搜索链接，是轻微瑕疵\n- 回答结构清晰（含亮点说明、表格汇总），实用性强\n\n**结论：baseline-model 完成了任务，提供了结构化、可验证的高质量答案。**\n\n---\n\n### 总结对比\n\n| 维度 | claude-sonnet-4-6 | baseline-model |\n|------|-------------------|----------------|\n| 是否完成任务 | ❌ 未完成（无最终答案） | ✅ 完成 |\n| 链接准确性 | N/A | 6/7 可验证正确 |\n| 内容覆盖度 | N/A | 7篇，覆盖全年多月 |\n| 时间范围符合 | N/A | 基本符合2025年 |\n| 组织结构 | N/A | 清晰（亮点+表格） |\n\n$$\\boxed{[[\\text{baseline-model}]]}$$
    """
)


# ---------------------------------------------------------------------------
# Parametrized: _parse_pairwise_output
# ---------------------------------------------------------------------------

_CASES = [
    ("empty", "", None),
    ("no_match", "没有找到匹配的模式", None),

    ("single_bs_target", r"\boxed{[[claude-sonnet-4-6]]}", 1.0),
    ("single_bs_baseline", r"\boxed{[[baseline-model]]}", 0.0),
    ("single_bs_tie", r"\boxed{[[平局]]}", 0.5),

    ("double_bs_target", "\\\\boxed{[[claude-sonnet-4-6]]}", 1.0),
    ("double_bs_baseline", "\\\\boxed{[[baseline-model]]}", 0.0),
    ("double_bs_tie", "\\\\boxed{[[平局]]}", 0.5),

    ("text_wrapper_single_bs", r"\boxed{[[\text{claude-sonnet-4-6}]]}", 1.0),
    ("text_wrapper_double_bs", "\\\\boxed{[[\\\\text{claude-sonnet-4-6}]]}", 1.0),

    ("latex_math_single_bs", r"$$\boxed{[[\text{claude-sonnet-4-6}]]}$$", 1.0),
    ("latex_math_double_bs", "$$\\\\boxed{[[\\\\text{claude-sonnet-4-6}]]}$$", 1.0),

    ("embedded_in_text", "分析完毕，最终结果：\\\\boxed{[[baseline-model]]}，感谢阅读", 0.0),
    ("unrecognized_model", r"\boxed{[[unknown-model]]}", None),
    ("real_trace_zcb_014", _REAL_TRACE_ZCB_014, 1.0),
    ("real_trace_zcb_014_1", _REAL_TRACE_ZCB_014_1, 0.0),
]


@pytest.mark.parametrize(
    "label, stdout, expected",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_parse_pairwise_output(label: str, stdout: str, expected: float | None):
    assert _parse(stdout, TARGET, BASELINE) == expected
