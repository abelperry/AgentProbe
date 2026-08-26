"""PipelineExecutor — inference→Queue→judge streaming pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from agent_probe.config import EvalExperimentConfig
from agent_probe.core.adapter import BaseAdapter
from agent_probe.core.executor import BaseTaskExecutor, EvalContext, EvalUnit
from agent_probe.core.models import (
    BaseInference,
    BaseJudgement,
    BaseQuestion,
    Error,
    JudgeResult,
    MetricsRecord,
)
from agent_probe.core.repo import JudgeRepo, MetricsRepo
from agent_probe.core.task import BaseTask

# Sentinel type for queue termination
_QueueItem = tuple[EvalUnit, BaseQuestion, BaseInference, EvalContext, BaseJudgement | None] | None


class PipelineExecutor(BaseTaskExecutor):
    """Default executor: all units run in one fully-concurrent pipeline.

    Persistence is delegated to ``JudgeRepo`` and ``MetricsRepo``.
    """

    def __init__(
        self,
        config: EvalExperimentConfig,
        adapters: dict[str, BaseAdapter[Any]],
        tasks: dict[str, BaseTask[Any, Any, Any]],
        judge_repo: JudgeRepo,
        metrics_repo: MetricsRepo,
    ) -> None:
        self._config = config
        self._adapters = adapters
        self._tasks = tasks
        self._judge_repo = judge_repo
        self._metrics_repo = metrics_repo
        self._semaphore = asyncio.Semaphore(config.concurrency)
        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self._units: list[EvalUnit] = []
        self._build_units()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _build_units(self) -> None:
        for dataset_name, adapter in self._adapters.items():
            for qid in adapter.list_ids():
                for agent_name in self._config.agents:
                    for model_name in self._config.models:
                        self._units.append(
                            EvalUnit(
                                qid=qid,
                                agent_name=agent_name,
                                model_name=model_name,
                                dataset_name=dataset_name,
                            )
                        )

    def _build_context(self, unit: EvalUnit) -> EvalContext:
        output_dir = (
            Path(self._config.output_dir)
            / self._config.name
            / unit.dataset_name
            / unit.agent_name
            / unit.model_name
        )
        return EvalContext(
            unit=unit,
            model_config=self._config.models[unit.model_name],
            agent_config=self._config.agents[unit.agent_name],
            dataset_config=self._config.datasets[unit.dataset_name],
            sandbox_config=self._config.sandbox,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("Starting pipeline: {} units, concurrency={}", len(self._units), self._config.concurrency)
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._produce_all())
            tg.create_task(self._consume_all())
        self._finalize()
        logger.info("Pipeline complete")

    # ------------------------------------------------------------------
    # Producer: inference
    # ------------------------------------------------------------------

    async def _produce_all(self) -> None:
        await asyncio.gather(*[self._infer_one(u) for u in self._units])
        await self._queue.put(None)

    async def _infer_one(self, unit: EvalUnit) -> None:
        ctx = self._build_context(unit)
        cached = self._judge_repo.find(
            unit.dataset_name, unit.agent_name, unit.model_name, unit.qid,
        )

        # Fully done — skip entirely
        if cached and cached.valid_judgement:
            logger.debug("[{}] Cache hit (complete), skipping", ctx.log_tag())
            return

        # Inference is good — skip infer, re-judge
        if cached and cached.valid_inference:
            logger.debug("[{}] Cache hit (inference only), re-judging", ctx.log_tag())
            await self._queue.put((unit, cached.question, cached.inference, ctx, cached.judgement))
            return

        # Run inference
        adapter = self._adapters[unit.dataset_name]
        task = self._tasks[unit.dataset_name]
        question = adapter.load(unit.qid)

        async with self._semaphore:
            logger.info("[{}] Inferring", ctx.log_tag())
            try:
                inference = await task.inference(question, ctx)
            except Exception as exc:
                logger.exception("[{}] Inference failed", ctx.log_tag())
                inference = BaseInference(
                    error=Error(code=-1, message=str(exc)),
                )

        await self._queue.put((unit, question, inference, ctx, None))

    # ------------------------------------------------------------------
    # Consumer: judge
    # ------------------------------------------------------------------

    async def _consume_all(self) -> None:
        async with asyncio.TaskGroup() as tg:
            for _ in range(self._config.concurrency):
                tg.create_task(self._consume_worker())

    async def _consume_worker(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                await self._queue.put(None)
                return
            await self._judge_one(*item)

    async def _judge_one(
        self,
        unit: EvalUnit,
        question: BaseQuestion,
        inference: BaseInference,
        ctx: EvalContext,
        prev_judgement: BaseJudgement | None,
    ) -> None:
        # Inference failed — save an error judgement and skip judge.
        if inference.error:
            logger.warning("[{}] Skipping judge, inference error: {}", ctx.log_tag(), inference.error.message)
            self._judge_repo.save(
                unit.dataset_name, unit.agent_name, unit.model_name, unit.qid,
                JudgeResult(
                    question=question,
                    inference=inference,
                    judgement=BaseJudgement(
                        error=Error(code=-1, message="has error in inference")
                    ),
                ),
            )
            return

        # Run judge
        task = self._tasks[unit.dataset_name]
        logger.info("[{}] Judging", ctx.log_tag())
        try:
            judgement = await task.judge(question, inference, ctx, prev_judgement)
        except Exception as exc:
            logger.exception("[{}] Judge failed", ctx.log_tag())
            judgement = BaseJudgement(
                error=Error(code=-1, message=str(exc)),
            )

        self._judge_repo.save(
            unit.dataset_name, unit.agent_name, unit.model_name, unit.qid,
            JudgeResult(question=question, inference=inference, judgement=judgement),
        )

    # ------------------------------------------------------------------
    # Post-pipeline: metrics
    # ------------------------------------------------------------------

    def _finalize(self) -> None:
        logger.info("Computing metrics...")
        for dataset_name in self._config.datasets:
            for agent_name in self._config.agents:
                for model_name in self._config.models:
                    results = self._judge_repo.find_all(
                        dataset_name, agent_name, model_name,
                    )
                    if not results:
                        continue

                    task = self._tasks[dataset_name]
                    adapter = self._adapters[dataset_name]
                    total = len(adapter.list_ids())
                    judgements = [
                        r.judgement for r in results if r.judgement is not None
                    ]
                    scores, success_count = task.collect_metrics(judgements)
                    logger.info(
                        "Metrics [{}/{}/{}]: {} ({}/{})",
                        dataset_name, agent_name, model_name,
                        scores, success_count, total,
                    )

                    self._metrics_repo.save(
                        MetricsRecord(
                            dataset=dataset_name,
                            agent=agent_name,
                            model=model_name,
                            total=total,
                            success_count=success_count,
                            scores=scores,
                        )
                    )
