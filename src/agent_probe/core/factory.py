"""ExperimentFactory — assembles a fully-wired executor from config."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from agent_probe.config import EvalExperimentConfig, SandboxConfig
from agent_probe.core.adapter import BaseAdapter, create_adapter
from agent_probe.core.executor import BaseTaskExecutor
from agent_probe.core.models import resolve_types
from agent_probe.core.task import BaseTask
from agent_probe.utils.imports import import_class


class ExperimentFactory:
    """Creates a ready-to-run executor from an experiment config."""

    def create(self, config: EvalExperimentConfig) -> BaseTaskExecutor:
        from agent_probe.executors.pipeline_executor import PipelineExecutor

        # Fill model_name from YAML key when empty
        for key, model_cfg in config.models.items():
            if not model_cfg.model_name:
                model_cfg.model_name = key

        tasks = self._build_tasks(config)
        adapters = self._build_adapters(config, tasks)
        judge_repo, metrics_repo = self._build_repos(config)

        logger.info("Experiment: {}", config.name)

        return PipelineExecutor(
            config=config,
            adapters=adapters,
            tasks=tasks,
            judge_repo=judge_repo,
            metrics_repo=metrics_repo,
        )

    # ------------------------------------------------------------------
    # Component builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tasks(
        config: EvalExperimentConfig
    ) -> dict[str, BaseTask[Any, Any, Any]]:
        tasks: dict[str, BaseTask[Any, Any, Any]] = {}
        for dataset_name, dataset_cfg in config.datasets.items():
            task_cls = import_class(dataset_cfg.task_type)
            tasks[dataset_name] = task_cls()
        return tasks

    @staticmethod
    def _build_adapters(
        config: EvalExperimentConfig,
        tasks: dict[str, BaseTask[Any, Any, Any]],
    ) -> dict[str, BaseAdapter[Any]]:
        adapters: dict[str, BaseAdapter[Any]] = {}
        for dataset_name, dataset_cfg in config.datasets.items():
            question_type, _, _ = resolve_types(type(tasks[dataset_name]))
            adapters[dataset_name] = create_adapter(
                dataset_cfg.adapter_type,
                data_dir=dataset_cfg.data_dir,
                question_type=question_type,
            )
        return adapters

    @staticmethod
    def _build_repos(config: EvalExperimentConfig) -> tuple[Any, Any]:
        from agent_probe.repos.file_judge_repo import FileJudgeRepo
        from agent_probe.repos.file_metrics_repo import FileMetricsRepo

        exp_dir = Path(config.output_dir) / config.name
        return FileJudgeRepo(exp_dir, config.datasets), FileMetricsRepo(exp_dir, config.datasets)
