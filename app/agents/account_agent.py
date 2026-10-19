from langchain_core.messages import SystemMessage
from app.config import llm
from app.state import SupportAgentState

def account_setup_agent(state: SupportAgentState) -> dict:
    """Handles automated account setup, verification, and profile onboarding."""
    summary = state.get("conversation_summary", "")
    sys_msg = SystemMessage(
        content=(
            "You are the Account Setup Specialist Agent. Help the customer set up their medical "
            "portal account, verify clinic credentials, and configure billing preferences concisely.\n"
            f"Prior Context Summary: {summary}"
        )
    )
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}
