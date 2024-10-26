async def validate_answer(state: FormState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Check if the user answered the current question."""
    configurable = FormConfiguration.from_runnable_config(config)
    current_question = configurable.questions[state.current_question_index]
    
    sys = FORM_SYSTEM_PROMPT.format(
        tone=configurable.tone,
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
    configurable = FormConfiguration.from_runnable_config(config)
    state.current_question_index += 1
    
    if state.current_question_index >= len(configurable.questions):
        return END
    
    next_q = configurable.questions[state.current_question_index]
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