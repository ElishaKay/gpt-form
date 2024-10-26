"""Define the configurable parameters for the agent."""

import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional, List

from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated

from memory_agent import prompts

@dataclass(kw_only=True)
class Configuration:
    """Main configuration class for the memory graph system."""

    user_id: str = "default"
    """The ID of the user to remember in the conversation."""
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )
    system_prompt: str = prompts.SYSTEM_PROMPT
    questions: list[str] = field(default_factory=lambda: [
        "What's your favorite color?",
        "How do you like to spend your weekends?",
        "What's your preferred way of learning?"
    ])

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }

        return cls(**{k: v for k, v in values.items() if v})

@dataclass
class FormConfiguration(Configuration):
    questions: List[str] = (
        "What is your name?",
        "What is your age?",
        "What are your hobbies?",
    )
    tone: str = "friendly and conversational"
    current_question_index: int = 0