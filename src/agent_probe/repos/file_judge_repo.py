"""File-based JudgeRepo — one JSON file per question."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from agent_probe.config import DatasetConfig
from agent_probe.core.models import (
    BaseInference,
    BaseJudgement,
    BaseQuestion,
    JudgeResult,
    resolve_types,
)
from agent_probe.core.repo import JudgeRepo
from agent_probe.utils.imports import import_class


class FileJudgeRepo(JudgeRepo):
    """Stores JudgeResult as ``{exp_dir}/{dataset}/{agent}/{model}/result/{qid}.json``."""

    def __init__(self, exp_dir: Path, datasets: dict[str, DatasetConfig]) -> None:
        self._exp_dir = exp_dir
        self._type_cache: dict[str, tuple[type, type, type]] = {}
        for name, cfg in datasets.items():
            task_cls = import_class(cfg.task_type)
            self._type_cache[name] = resolve_types(task_cls)

    def _result_dir(self, dataset: str, agent: str, model: str) -> Path:
        return self._exp_dir / dataset / agent / model / "result"

    def _result_path(self, dataset: str, agent: str, model: str, qid: str) -> Path:
        return self._result_dir(dataset, agent, model) / f"{qid}.json"

    def _types_for(self, dataset: str) -> tuple[type, type, type]:
        return self._type_cache[dataset]

    def save(
        self,
        dataset: str,
        agent: str,
        model: str,
        qid: str,
        result: JudgeResult,
    ) -> None:
        path = self._result_path(dataset, agent, model, qid)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "question": result.question.model_dump(mode="json"),
            "inference": result.inference.model_dump(mode="json"),
            "judgement": result.judgement.model_dump(mode="json") if result.judgement else None,
        }
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def find(
        self,
        dataset: str,
        agent: str,
        model: str,
        qid: str,
    ) -> JudgeResult | None:
        path = self._result_path(dataset, agent, model, qid)
        if not path.exists():
            return None
        try:
            return self._load_file(path, dataset)
        except Exception:
            logger.warning("Failed to load {}", path)
            return None

    def find_all(
        self,
        dataset: str,
        agent: str,
        model: str,
    ) -> list[JudgeResult]:
        result_dir = self._result_dir(dataset, agent, model)
        if not result_dir.exists():
            return []
        results: list[JudgeResult] = []
        for path in sorted(result_dir.glob("*.json")):
            try:
                results.append(self._load_file(path, dataset))
            except Exception:
                logger.warning("Failed to load {}", path)
                continue
        return results

    def _load_file(self, path: Path, dataset: str) -> JudgeResult:
        data = json.loads(path.read_text(encoding="utf-8"))
        q_cls, i_cls, j_cls = self._types_for(dataset)

        question = q_cls.model_validate(data["question"])
        inference = i_cls.model_validate(data["inference"])
        judgement = j_cls.model_validate(data["judgement"]) if data.get("judgement") else None

        return JudgeResult(question=question, inference=inference, judgement=judgement)
