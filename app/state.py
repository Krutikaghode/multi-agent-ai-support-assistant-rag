from typing import Annotated, Literal, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class SupportAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    active_agent: Literal["account_setup", "call_scheduling", "product_rag"]
    conversation_summary: str
    account_status: Optional[dict]
    scheduled_call: Optional[dict]
