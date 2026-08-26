"""File-based MetricsRepo — one JSONL file per dataset."""

from __future__ import annotations

from pathlib import Path

from agent_probe.config import DatasetConfig
from agent_probe.core.models import MetricsRecord
from agent_probe.core.repo import MetricsRepo


class FileMetricsRepo(MetricsRepo):
    """Stores MetricsRecord as lines in ``{exp_dir}/{dataset}/metrics.jsonl``.

    Existing metrics files are deleted on init so each run starts fresh.
    """

    def __init__(self, exp_dir: Path, datasets: dict[str, DatasetConfig]) -> None:
        self._exp_dir = exp_dir
        for name in datasets:
            path = self._metrics_path(name)
            if path.exists():
                path.unlink()

    def _metrics_path(self, dataset: str) -> Path:
        return self._exp_dir / dataset / "metrics.jsonl"

    def save(self, record: MetricsRecord) -> None:
        path = self._metrics_path(record.dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
