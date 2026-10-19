from langchain_core.messages import SystemMessage
from app.config import llm
from app.state import SupportAgentState
from app.rag.retriever import retrieve_medical_products

def product_rag_agent(state: SupportAgentState) -> dict:
    """Answers company queries and recommends suitable items from the 47 medical products catalog."""
    last_query = state["messages"][-1].content
    summary = state.get("conversation_summary", "")
    retrieved_context = retrieve_medical_products(last_query)

    sys_msg = SystemMessage(
        content=(
            "You are the Medical Product & Company Support RAG Specialist. Use ONLY the retrieved "
            "catalog information from our 47 medical products to answer the customer's question "
            "accurately and recommend suitable products.\n\n"
            f"Retrieved Catalog Context:\n{retrieved_context}\n\n"
            f"Prior Conversation Summary: {summary}"
        )
    )
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}
