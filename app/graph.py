from langgraph.graph import END, START, StateGraph
from app.state import SupportAgentState
from app.agents.supervisor import supervisor_router
from app.agents.account_agent import account_setup_agent
from app.agents.scheduler_agent import call_scheduling_agent
from app.agents.rag_agent import product_rag_agent
from app.agents.summarizer import should_summarize, summarize_conversation

def build_support_graph(checkpointer=None):
    builder = StateGraph(SupportAgentState)
    builder.add_node("supervisor", supervisor_router)
    builder.add_node("account_setup", account_setup_agent)
    builder.add_node("call_scheduling", call_scheduling_agent)
    builder.add_node("product_rag", product_rag_agent)
    builder.add_node("summarize_conversation", summarize_conversation)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["active_agent"],
        {
            "account_setup": "account_setup",
            "call_scheduling": "call_scheduling",
            "product_rag": "product_rag",
        },
    )

    for agent_node in ["account_setup", "call_scheduling", "product_rag"]:
        builder.add_conditional_edges(
            agent_node,
            should_summarize,
            {
                "summarize_conversation": "summarize_conversation",
                "__end__": END,
            },
        )

    builder.add_edge("summarize_conversation", END)
    return builder.compile(checkpointer=checkpointer)
