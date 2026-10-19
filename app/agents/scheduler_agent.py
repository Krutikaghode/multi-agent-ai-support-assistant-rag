from langchain_core.messages import SystemMessage
from app.config import llm
from app.state import SupportAgentState

def call_scheduling_agent(state: SupportAgentState) -> dict:
    """Automates support and clinical consultation call scheduling."""
    summary = state.get("conversation_summary", "")
    sys_msg = SystemMessage(
        content=(
            "You are the Call Scheduling Specialist Agent. Assist the customer in booking, "
            "rescheduling, or confirming support and clinical specialist calls. Confirm date, "
            "time, timezone, and contact number.\n"
            f"Prior Context Summary: {summary}"
        )
    )
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}
