from langchain_core.messages import SystemMessage
from app.config import llm
from app.state import SupportAgentState

def supervisor_router(state: SupportAgentState) -> dict:
    """Classifies user intent to route to Account Setup, Call Scheduling, or Product RAG."""
    last_user_msg = state["messages"][-1].content
    summary_ctx = state.get("conversation_summary", "")

    router_prompt = (
        "You are a fast supervisor router for a medical customer support assistant. "
        "Classify the user's request into exactly ONE of these categories:\n"
        "- 'account_setup': account registration, onboarding, profile setup, credentials, KYC.\n"
        "- 'call_scheduling': booking a call, scheduling a demo/support consultation, checking calendar slots.\n"
        "- 'product_rag': questions about company information or any of our 47 medical products, specs, recommendations.\n\n"
        f"Conversation Summary: {summary_ctx}\n"
        f"User Message: {last_user_msg}\n\n"
        "Respond with ONLY the category name."
    )
    decision = llm.invoke([SystemMessage(content=router_prompt)]).content.strip().lower()
    if decision not in {"account_setup", "call_scheduling", "product_rag"}:
        decision = "product_rag"
    return {"active_agent": decision}
