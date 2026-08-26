"""Base Q-I-J data models and common types for the AgentProbe evaluation pipeline."""

from enum import IntEnum
from typing import Optional, TypeVar

from pydantic import BaseModel


class ErrorCode(IntEnum):
    """Framework-level error codes.

    Convention:
        code < 0 — transient/infra/agent runtime errors; user may rerun the task.
        code > 0 — reserved for business-layer errors (benchmarks decide).
    """

    SANDBOX_TIMEOUT = -1
    SANDBOX_UNKNOWN = -2
    AGENT_INSTALL = -3
    AGENT_EXIT_NONZERO = -4
    AGENT_STOP_ERROR = -5


class Error(BaseModel):
    """Generic error structure used across the framework."""

    code: int
    message: str


class LastAssistant(BaseModel):
    """Snapshot of the last assistant message from an agent trace."""

    stop_reason: Optional[str] = None
    error_message: Optional[str] = None
    content_text: str = ""


class BaseQuestion(BaseModel):
    """Base class for all Question types.

    Subclasses MUST override ``qid()`` to return a unique identifier.
    """

    def qid(self) -> str:
        """Return the unique question identifier. Must be overridden."""
        raise NotImplementedError("Subclasses must implement qid()")


class BaseInference(BaseModel):
    """Base class for all Inference result types."""

    error: Optional[Error] = None


class BaseJudgement(BaseModel):
    """Base class for all Judgement result types."""

    error: Optional[Error] = None


Q = TypeVar("Q", bound=BaseQuestion)
I = TypeVar("I", bound=BaseInference)
J = TypeVar("J", bound=BaseJudgement)


def resolve_types(
    task_cls: type,
) -> tuple[type[BaseQuestion], type[BaseInference], type[BaseJudgement]]:
    """Extract (Q, I, J) concrete types from a ``BaseTask[Q, I, J]`` subclass."""
    for base in getattr(task_cls, "__orig_bases__", ()):
        args = getattr(base, "__args__", None)
        if args and len(args) == 3:
            return args[0], args[1], args[2]
    raise TypeError(
        f"Cannot resolve Q/I/J types from {task_cls.__name__}. "
        f"Ensure it inherits BaseTask[Q, I, J] with concrete type arguments."
    )


# ---------------------------------------------------------------------------
# Aggregate models for persistence
# ---------------------------------------------------------------------------


class JudgeResult(BaseModel):
    """One fully-evaluated question: Q + I + J bundled together."""

    question: BaseQuestion
    inference: BaseInference
    judgement: Optional[BaseJudgement] = None

    @property
    def valid_inference(self) -> bool:
        return self.inference.error is None

    @property
    def valid_judgement(self) -> bool:
        return self.judgement is not None and self.judgement.error is None


class MetricsRecord(BaseModel):
    """Aggregated metrics for one (dataset, agent, model) combination."""

    dataset: str
    agent: str
    model: str
    total: int = 0
    success_count: int = 0
    scores: dict[str, float]
