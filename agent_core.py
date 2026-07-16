# agent_core.py
# All the agent logic from before, wrapped so a website can call it repeatedly.

import os
from typing import TypedDict, List
import streamlit as st

# Read the key from Streamlit's secrets manager once deployed.
# Locally, it falls back to an environment variable instead.
os.environ["OPENAI_API_KEY"] = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

# ---------- Sample policy data ----------
policies = [
    {"id": "policy_1", "text": "Auto Policy A: Covers collision damage up to $10,000. Water damage is excluded unless caused by a covered collision. Deductible is $500."},
    {"id": "policy_2", "text": "Auto Policy B: Covers water damage from flooding up to $5,000 if the policyholder has the Flood Endorsement add-on. Standard policy excludes flood damage entirely."},
    {"id": "policy_3", "text": "State Regulation - California: Insurers must respond to claims within 15 business days. Denying a valid claim without justification can result in bad-faith penalties."},
]

# ---------- Build the knowledge base once, when this file is imported ----------
def _build_retriever():
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = [Document(page_content=p["text"], metadata={"source": p["id"]}) for p in policies]
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="insurance_policies")
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = _build_retriever()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------- Agent state ----------
class ClaimState(TypedDict):
    claim: str
    query: str
    retrieved_docs: List[str]
    relevance_score: str
    retry_count: int
    decision: str
    reasoning: str
    grounded: bool

# ---------- Nodes ----------
def retrieve_node(state: ClaimState) -> ClaimState:
    query = state.get("query") or state["claim"]
    results = retriever.invoke(query)
    state["retrieved_docs"] = [doc.page_content for doc in results]
    return state

def grade_relevance_node(state: ClaimState) -> ClaimState:
    docs_text = "\n".join(state["retrieved_docs"])
    prompt = f"""Claim: {state['claim']}
Retrieved policy text: {docs_text}

Does the retrieved text contain enough information to address this claim?
Answer only "yes" or "no"."""
    response = llm.invoke(prompt)
    state["relevance_score"] = response.content.strip().lower()
    return state

def rewrite_query_node(state: ClaimState) -> ClaimState:
    prompt = f"""The search for this claim did not return relevant policy text.
Claim: {state['claim']}
Write one better, more specific search query to find the relevant policy clause."""
    response = llm.invoke(prompt)
    state["query"] = response.content.strip()
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state

def decide_node(state: ClaimState) -> ClaimState:
    docs_text = "\n".join(state["retrieved_docs"])
    prompt = f"""You are an insurance claims adjuster.
Claim: {state['claim']}
Policy text: {docs_text}

Decide: approve, deny, or escalate.
Line 1: your decision only.
Line 2: your reasoning, referencing the policy text."""
    response = llm.invoke(prompt).content.strip().split("\n", 1)
    state["decision"] = response[0].strip().lower()
    state["reasoning"] = response[1].strip() if len(response) > 1 else ""
    return state

def check_grounding_node(state: ClaimState) -> ClaimState:
    docs_text = "\n".join(state["retrieved_docs"])
    prompt = f"""Reasoning: {state['reasoning']}
Source policy text: {docs_text}

Does the reasoning rely only on the source text above, with no invented facts?
Answer only "yes" or "no"."""
    response = llm.invoke(prompt)
    state["grounded"] = "yes" in response.content.strip().lower()
    return state

def escalate_node(state: ClaimState) -> ClaimState:
    state["decision"] = "escalate"
    state["reasoning"] = "Insufficient or ungrounded evidence. Routed to a human adjuster."
    return state

def output_node(state: ClaimState) -> ClaimState:
    return state  # website reads the state directly, no need to print here

# ---------- Routing ----------
MAX_RETRIES = 2

def relevance_router(state: ClaimState) -> str:
    if "yes" in state["relevance_score"]:
        return "decide"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "escalate"
    return "rewrite"

def grounding_router(state: ClaimState) -> str:
    return "output" if state["grounded"] else "escalate"

# ---------- Build the graph once ----------
def _build_graph():
    graph = StateGraph(ClaimState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_relevance", grade_relevance_node)
    graph.add_node("rewrite", rewrite_query_node)
    graph.add_node("decide", decide_node)
    graph.add_node("check_grounding", check_grounding_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("output", output_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_relevance")
    graph.add_conditional_edges("grade_relevance", relevance_router, {
        "decide": "decide", "rewrite": "rewrite", "escalate": "escalate",
    })
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("decide", "check_grounding")
    graph.add_conditional_edges("check_grounding", grounding_router, {
        "output": "output", "escalate": "escalate",
    })
    graph.add_edge("escalate", END)
    graph.add_edge("output", END)
    return graph.compile()

app_graph = _build_graph()

# ---------- The function the website will call ----------
def run_claim(claim_text: str) -> dict:
    """Takes a claim as plain text, returns the agent's decision and reasoning."""
    result = app_graph.invoke({"claim": claim_text, "retry_count": 0})
    return {
        "decision": result.get("decision", "unknown"),
        "reasoning": result.get("reasoning", ""),
    }
