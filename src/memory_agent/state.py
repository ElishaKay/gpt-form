"""Define the shared values."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated

@dataclass(kw_only=True)
class State:
    """Main graph state."""

    messages: Annotated[list[AnyMessage], add_messages]
    """The messages in the conversation."""


__all__ = [
    "State",
]

@dataclass(kw_only=True)
class FormState(State):
    """Form state tracking."""
    current_question_index: int = 0
    answered_questions: dict = field(default_factory=dict)
