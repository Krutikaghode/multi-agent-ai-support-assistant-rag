"""
Multi-Agent AI Support Assistant with RAG
Orchestrates Account Setup, Call Scheduling, and Medical Product RAG (47 products)
using LangGraph, Google Gemini, Pinecone, and MongoDB with >12 message summarization.
"""

import os
from typing import Annotated, Literal, Optional, TypedDict
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb import MongoDBSaver
from pinecone import Pinecone
from pymongo import MongoClient

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-products-47")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "support_assistant_db")
MESSAGE_SUMMARY_THRESHOLD = int(os.getenv("MESSAGE_SUMMARY_THRESHOLD", "12"))
RETAIN_RECENT_MESSAGES = 4

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

class SupportAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    active_agent: Literal["account_setup", "call_scheduling", "product_rag"]
    conversation_summary: str

def retrieve_medical_products(query: str, category_filter: Optional[str] = None, top_k: int = 4) -> str:
    if not PINECONE_API_KEY:
        return "Configure PINECONE_API_KEY to query the 47 medical products index."
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    query_vector = embeddings.embed_query(query)
    filter_dict = {"category": {"$eq": category_filter}} if category_filter else None
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True, filter=filter_dict)
    matches = results.get("matches", [])
    if not matches:
        return "No matching medical products found in the 47-product catalog."
    return "\n\n".join(
        f"- SKU: {m.get('metadata', {}).get('sku', 'N/A')} | Name: {m.get('metadata', {}).get('name', 'Unknown')}"
        for m in matches
    )

def supervisor_router(state: SupportAgentState) -> dict:
    last_user_msg = state["messages"][-1].content
    summary_ctx = state.get("conversation_summary", "")
    router_prompt = (
        "Classify the user's request into ONE category: 'account_setup', 'call_scheduling', or 'product_rag'.\n"
        f"Summary: {summary_ctx}\nUser Message: {last_user_msg}"
    )
    decision = llm.invoke([SystemMessage(content=router_prompt)]).content.strip().lower()
    if decision not in {"account_setup", "call_scheduling", "product_rag"}:
        decision = "product_rag"
    return {"active_agent": decision}

def account_setup_agent(state: SupportAgentState) -> dict:
    sys_msg = SystemMessage(content=f"Account Setup Specialist Agent. Prior Summary: {state.get('conversation_summary', '')}")
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}

def call_scheduling_agent(state: SupportAgentState) -> dict:
    sys_msg = SystemMessage(content=f"Call Scheduling Specialist Agent. Prior Summary: {state.get('conversation_summary', '')}")
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}

def product_rag_agent(state: SupportAgentState) -> dict:
    retrieved_context = retrieve_medical_products(state["messages"][-1].content)
    sys_msg = SystemMessage(content=f"Medical Product RAG Specialist (47 products).\nContext:\n{retrieved_context}\nSummary: {state.get('conversation_summary', '')}")
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}

def should_summarize(state: SupportAgentState) -> Literal["summarize_conversation", "__end__"]:
    return "summarize_conversation" if len(state["messages"]) > MESSAGE_SUMMARY_THRESHOLD else "__end__"

def summarize_conversation(state: SupportAgentState) -> dict:
    existing_summary = state.get("conversation_summary", "")
    messages_to_summarize = state["messages"][:-RETAIN_RECENT_MESSAGES]
    summary_prompt = f"Summarize concisely preserving key entities.\nExisting: {existing_summary}\n" + "\n".join(f"{m.type}: {m.content}" for m in messages_to_summarize)
    new_summary = llm.invoke([HumanMessage(content=summary_prompt)]).content
    return {"conversation_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages_to_summarize if m.id]}

def build_support_graph(checkpointer=None):
    builder = StateGraph(SupportAgentState)
    builder.add_node("supervisor", supervisor_router)
    builder.add_node("account_setup", account_setup_agent)
    builder.add_node("call_scheduling", call_scheduling_agent)
    builder.add_node("product_rag", product_rag_agent)
    builder.add_node("summarize_conversation", summarize_conversation)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", lambda s: s["active_agent"], {
        "account_setup": "account_setup", "call_scheduling": "call_scheduling", "product_rag": "product_rag"
    })
    for node in ["account_setup", "call_scheduling", "product_rag"]:
        builder.add_conditional_edges(node, should_summarize, {"summarize_conversation": "summarize_conversation", "__end__": END})
    builder.add_edge("summarize_conversation", END)
    return builder.compile(checkpointer=checkpointer)
