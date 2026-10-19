from typing import Literal
from langchain_core.messages import HumanMessage, RemoveMessage
from app.config import llm, MESSAGE_SUMMARY_THRESHOLD, RETAIN_RECENT_MESSAGES
from app.state import SupportAgentState

def should_summarize(state: SupportAgentState) -> Literal["summarize_conversation", "__end__"]:
    """Triggers conversation summarization when message count exceeds 12."""
    if len(state["messages"]) > MESSAGE_SUMMARY_THRESHOLD:
        return "summarize_conversation"
    return "__end__"

def summarize_conversation(state: SupportAgentState) -> dict:
    """Compresses older chat history when message count exceeds 12."""
    existing_summary = state.get("conversation_summary", "")
    messages_to_summarize = state["messages"][:-RETAIN_RECENT_MESSAGES]

    summary_prompt = (
        "Update the running conversation summary concisely. Preserve all key customer details: "
        "user ID, clinic/account setup status, recommended medical products (SKUs), clinical constraints, "
        "and scheduled call times.\n\n"
        f"Existing Summary: {existing_summary}\n\n"
        "New Messages to Incorporate:\n"
        + "\n".join(f"{m.type}: {m.content}" for m in messages_to_summarize)
    )

    new_summary = llm.invoke([HumanMessage(content=summary_prompt)]).content
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize if m.id]

    return {
        "conversation_summary": new_summary,
        "messages": delete_messages,
    }
