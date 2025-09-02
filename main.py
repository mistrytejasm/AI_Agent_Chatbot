from core.agents.search_agent import SearchAgent
from core.agents.citation_manager import CitationManager
from core.agents.synthesis_agent import SynthesisAgent
from core.models.state import AgentState
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from dotenv import load_dotenv

load_dotenv()

# Initialize components
llm = ChatGroq(model="openai/gpt-oss-120b")
search_agent = SearchAgent()
synthesis_agent = SynthesisAgent()

# Database
sqlite_conn = sqlite3.connect("database/checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

def router_node(state: AgentState):
    """Routes queries for appropriate processing"""
    last_message = state["messages"][-1]
    return{
        **state,
        "current_query": last_message.content,
        "processing_stage": "routing_complete"
    }

def search_node(state: AgentState):
    """Handle search and source gathering"""
    search_results = search_agent.search_and_analyze(state)
    return {**state, **search_results}


def synthesis_node(state: AgentState):
    """Handle response synthesis"""
    synthesis_results = synthesis_agent.synthesize_response(state)

    # Create final AI message
    final_message = AIMessage(
        content=synthesis_results["final_response"],
        name="perplexity_search"
    )

    return {
        **state,
        **synthesis_results,
        "messages": state["messages"] + [final_message]
    }


# Build graph
graph = StateGraph(AgentState)
graph.add_node("router", router_node)
graph.add_node("search", search_node)
graph.add_node("synthesis", synthesis_node)

graph.set_entry_point("router")
graph.add_edge("router", "search")
graph.add_edge("search", "synthesis")
graph.add_edge("synthesis", END)

app = graph.compile(checkpointer=memory)

# CLI Interface
if __name__ == "__main__":
    print("🔍 Perplexity-like Search ChatBot is Ready!")
    print("=" * 60)

    config = {"configurable": {"thread_id": "search_session"}}

    while True:
        try:
            query = input("\nUser: ").strip()

            if query.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break

            if query.lower() == "reset":
                memory.delete_thread("search_session")
                print("🔄 Memory reset!")
                continue

            print("\n🔍 Searching and analyzing...")

            result = app.invoke({
                "messages": [HumanMessage(content=query)],
                "search_results": [],
                "sources": {},
                "synthesized_content": [],
                "current_query": "",
                "processing_stage": "initialized"
            }, config=config)

            final_message = result["messages"][-1]
            print(f"\n🤖 AI: {final_message.content}")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")