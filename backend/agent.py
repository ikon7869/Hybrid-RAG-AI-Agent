import os
import json, re
from typing import List, Literal, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig 

# Import API keys from config
from config import GROQ_API_KEY, TAVILY_API_KEY
from vectorstore import get_retriever

from redis_store import get_redis_client, store_cache, get_cached_answer
from cross_encoder import get_cross_encoder, re_rank_docs
# --- Tools ---
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
tavily = TavilySearch(max_results=3, topic="general")

@tool
def web_search_tool(query: str) -> str:
    """Up-to-date web info via Tavily"""
    try:
        result = tavily.invoke({"query": query})
        return result
    except Exception as e:
        return f"WEB_ERROR::{e}"

@tool
def rag_search_tool(query: str) -> str:
    """Top-K chunks from KB (empty string if none)"""
    try:
        retriever_instance = get_retriever()
        docs = retriever_instance.invoke(query, k=5) 
        return docs
    except Exception as e:
        return f"RAG_ERROR::{e}"

# --- Pydantic schemas for structured output ---
class RouteDecision(BaseModel):
    route: Literal["hybrid", "answer", "end"]
    reply: str | None = Field(None, description="Filled only when route == 'end'")

class RagJudge(BaseModel):
    sufficient: bool = Field(..., description="True if retrieved information is sufficient to answer the user's question, False otherwise.")

# --- LLM instances with structured output where needed ---
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

router_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0).with_structured_output(RouteDecision)
judge_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
answer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# --- Shared state type ---
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    rag_chunks: List[str]
    web: List[str]
    restricted_mode: bool
    route: Literal["hybrid", "answer", "end"]
    web_count: int
    rag_count: int
    cache: bool

def compute_confidence(state: AgentState, answer: str) -> int:
    rag_count = state.get("rag_count", 0)
    web_count = state.get("web_count", 0)
    score = 0
    if rag_count > 3:
        score += 50

    if web_count >= 2:
        score += 30

    if rag_count == 0 and web_count == 0:
        score = 20 

    return max(5, min(score, 95))


# --- Node 1: router (decision) ---
def router_node(state: AgentState, config: RunnableConfig) -> AgentState:

    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")

    client = get_redis_client()

    cached_ans = get_cached_answer(client, query)
    if cached_ans:
        print("cache hit")
        return {
            "messages": state["messages"] + [AIMessage(content=cached_ans)],  
            "route": "cache",
            "cache": True
        }

    restricted_mode = config.get("configurable", {}).get("restricted_mode", False)

    system_prompt = f"""
You are a routing agent for a Hybrid RAG system.

Your job is to decide WHETHER retrieval is needed — NOT which source.

Routes:
- "hybrid" → Use retrieval (documents + web if enabled)
- "answer" → No retrieval needed, LLM can answer directly
- "end" → Greeting or small talk. Must include a friendly reply.

Guidelines:
• Choose "hybrid" for factual, informational, or knowledge-based queries
• Choose "answer" for math, reasoning, coding help, or general LLM knowledge
• Choose "end" for greetings like "hi", "hello", "how are you"

Restricted Mode:{"ENABLED" if restricted_mode else "DISABLED"}
If restricted_mode is True, the "hybrid" route should rely solely on document retrieval and should not perform web searches. 


Examples:
User: "What is diabetes?" → hybrid
User: "Latest AI news" → hybrid
User: "2+2?" → answer
User: "Hello!" → end
"""

    messages = [
        ("system", system_prompt),
        ("user", query)
    ]

    result: RouteDecision = router_llm.invoke(messages)

    print(f"Router decision: {result.route}")

    out = {
        "messages": state["messages"],
        "route": result.route,
        "restricted_mode": restricted_mode,
        "cache": False
    }

    if result.route == "end":
        out["messages"] = state["messages"] + [AIMessage(content=result.reply or "Hey! How can I help you today?")]

    print("--- Exiting router_node ---")
    return out


# --- Node 2: RAG lookup ---
def rag_node(state: AgentState, config: RunnableConfig) -> AgentState:

    query = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    retriever = get_retriever()
    docs = retriever.invoke(query)
    reranked_docs = re_rank_docs(get_cross_encoder(), query, [d.page_content for d in docs])
    print(f"RAG retrieved {len(docs)} docs, reranked to top {len(reranked_docs)}")
    restricted_mode = config.get("configurable", {}).get("restricted_mode", False)

    if not docs:
        return {
            **state,
            "rag_chunks": [],
            "rag_count": 0,
            "restricted_mode": restricted_mode
        }

    unique_docs = []
    seen_contents = set()

    for d in docs:
        clean_text = " ".join(d.page_content.split()) 
        if clean_text not in seen_contents:
            seen_contents.add(clean_text)
            unique_docs.append((d, clean_text))

    formatted_chunks = []

    for i, (doc, clean_text) in enumerate(unique_docs, 1):

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")

        if page is not None:
            try:
                page = int(page)
            except:
                page = page
        else:
            page = "N/A"

        formatted_chunk = (
            f"[RAG-{i}] {source} (Page {page})\n"
            f"{clean_text}"
        )

        formatted_chunks.append(formatted_chunk)

    return {
        **state,
        "rag_chunks": formatted_chunks,
        "rag_count": len(formatted_chunks),
        "restricted_mode": restricted_mode
    }

# --- Node 3: web search ---
def web_node(state: AgentState,config:RunnableConfig) -> AgentState:
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    restricted_mode = state.get("restricted_mode", False)
    if restricted_mode:
        return {**state, "web": "Restricted Model Enabled", "route": "fusion_answer"}

    results = web_search_tool.invoke(query)

    snippets = [
        f"[WEB-{i}] {item['title']} - {item['content']} ({item['url']})"
        for i, item in enumerate(results.get("results", []), 1)
    ]
    return {**state, 
            "web": snippets, 
            "web_count": len(snippets),
            "route": "fusion_answer"
            }

# --- Node 4: final answer ---
def fusion_answer_node(state: AgentState) -> AgentState:
    user_q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    rag_context = "\n".join(state.get("rag_chunks", []))
    web_context = "\n".join(state.get("web", []))

    restricted_mode = state.get("restricted_mode", False)
    if restricted_mode:
        if not rag_context:
            refusal_msg = """
            The requested information is not available in the approved document knowledge base.
            Since restricted mode is enabled, external sources cannot be used.

            Please upload relevant documents or rephrase your query.
            """

            return {
                **state, 
                "messages": state["messages"] + [AIMessage(content=refusal_msg)]
            }

        context ="\n".join(rag_context)
        system_prompt = """
        You are operating in STRICT RESTRICTED MODE.

        Rules:
        - Use ONLY provided document context
        - Do NOT use outside knowledge
        - If answer not in context, explicitly say so
        - Cite sources like [RAG-1]
        """

        answer = answer_llm.invoke([
            ("system", system_prompt),
            ("user", f"Question: {user_q}\n\nDOCUMENT CONTEXT:\n{context}")
        ]).content

        confidence = compute_confidence(state, answer)

        final_answer = f"""{answer}
        Confidence: {confidence}/100 (Restricted Mode)
        """

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=final_answer)]
        }
    print(state.keys())
    if state['cache']:
        return {
        **state,
        "messages": state["messages"],
    }

    system_prompt = """
    You are a hybrid RAG system combining document knowledge and live web data.

    Use both document knowledge and web results with citations.

    Citing Rules:
    - Cite document sources like [RAG-1], [RAG-2]
    - Cite web sources like [WEB-1], [WEB-2]
    - Only cite sources actually used
    - If unsure, say so

    """

    prompt = f"""
    Question: {user_q}

    DOCUMENT KNOWLEDGE:
    {rag_context}

    LIVE WEB DATA:
    {web_context}
    """

    ans = answer_llm.invoke([
        ("system", system_prompt),
        ("user", prompt)
    ]).content

    confidence = compute_confidence(state, ans)
    ans += f"\n\nConfidence: {confidence}/100"

    store_cache(user_q, ans)

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=ans)],
    }

# --- Routing helpers ---
def from_router(st: AgentState) -> Literal["hybrid", "answer", "end", "cache"]:
    return st["route"]

# --- Build graph ---
def build_agent():
    """Builds and compiles the Hybrid RAG + Web LangGraph agent."""
    
    g = StateGraph(AgentState)

    g.add_node("router", router_node)
    g.add_node("rag_lookup", rag_node)
    g.add_node("web_search", web_node)
    g.add_node("fusion_answer", fusion_answer_node)  # NEW

    g.set_entry_point("router")

    g.add_conditional_edges(
    "router",
    lambda s: s["route"],
    {
        "hybrid": "rag_lookup", 
        "answer": "fusion_answer", 
        "cache": "fusion_answer",
        "end": END
    }
)

    # Hybrid path runs BOTH
    g.add_edge("rag_lookup", "web_search")
    g.add_edge("web_search", "fusion_answer")

    g.add_edge("fusion_answer", END)


    agent = g.compile(checkpointer=MemorySaver())
    return agent


rag_agent = build_agent()