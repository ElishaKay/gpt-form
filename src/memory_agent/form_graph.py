# form_graph.py
import asyncio
import logging
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.store.base import BaseStore

from memory_agent import configuration, tools, utils
from memory_agent.state import FormState

logger = logging.getLogger(__name__)

# Initialize the language model
llm = init_chat_model()

FORM_SYSTEM_PROMPT = """You are a {tone} assistant conducting a survey.
Current question: {current_question}
Previous answers: {user_info}
Current time: {time}

Your task:
1. If the user has answered the current question clearly, save it to memory
2. If the user hasn't answered clearly, ask for clarification
3. Once a question is answered, move to the next question

Remember to maintain a {tone} tone throughout the conversation."""

async def validate_answer(state: FormState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Check if the user answered the current question."""
    configurable = configuration.Configuration.from_runnable_config(config)
    current_question = config["configurable"]["questions"][state.current_question_index]
    
    sys = FORM_SYSTEM_PROMPT.format(
        tone=config["configurable"]["tone"],
        current_question=current_question,
        user_info=state.answered_questions,
        time=datetime.now().isoformat()
    )
    
    response = await llm.bind_tools([tools.upsert_memory]).ainvoke(
        [{"role": "system", "content": sys}, *state.messages],
        {"configurable": utils.split_model_and_provider(configurable.model)},
    )
    return {"messages": [response]}

def route_answer(state: FormState, config: RunnableConfig):
    """Determine next step based on answer validation."""
    msg = state.messages[-1]
    if msg.tool_calls:
        # Answer was valid and saved
        return "next_question"
    return "ask_again"

async def next_question(state: FormState, config: RunnableConfig) -> dict:
    """Move to next question or end form."""
    questions = config["configurable"]["questions"]
    state.current_question_index += 1
    
    if state.current_question_index >= len(questions):
        return END
    
    next_q = questions[state.current_question_index]
    return {"messages": [{"role": "assistant", "content": next_q}]}

# Create form graph
form_builder = StateGraph(FormState)
form_builder.add_node("validate_answer", validate_answer)
form_builder.add_node("next_question", next_question)
form_builder.add_conditional_edges(
    "validate_answer",
    route_answer,
    {
        "next_question": "next_question",
        "ask_again": "validate_answer"
    }
)
form_builder.add_edge("next_question", "validate_answer")
form_graph = form_builder.compile()
form_graph.name = "FormAgent"

# Example usage
if __name__ == "__main__":
    config = {
        "configurable": {
            "questions": [
                "What's your favorite color?",
                "How do you like to spend your weekends?",
                "What's your preferred way of learning?"
            ],
            "tone": "friendly and empathetic",
            "user_id": "user123",
            "model": "openai/gpt-4"  # Add model configuration
        }
    }

    # Start conversation
    async def main():
        await form_graph.ainvoke(
            {"messages": [{"role": "system", "content": "Let's begin the survey"}]},
            config
        )

    asyncio.run(main())

__all__ = ["form_graph"]