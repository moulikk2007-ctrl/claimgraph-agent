import os
import time
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Map the Streamlit secret to the environment variable for ChatOpenAI (if needed later)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
elif "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

def _build_retriever():
    # Sample Mock Policy Document representing standard coverages/exclusions
    raw_text = """
    SECTION 1: COMPREHENSIVE COVERAGE
    Comprehensive coverage protects against loss or damage to the insured vehicle caused by reasons other than collision. This includes acts of nature (hail, floods, hurricanes), theft, vandalism, and animal impacts.
    
    SECTION 2: PROPERTY EXCLUSIONS (WEAR AND TEAR)
    Standard homeowners and auto policies strictly exclude damage resulting from wear and tear, inherent vice, mechanical or electrical breakdown, gradual deterioration, or lack of routine maintenance by the property owner.
    
    SECTION 3: WATER DAMAGE VS FLOOD DEFINITION
    Standard homeowners insurance covers sudden and accidental internal water damage (e.g., burst pipes). However, external surface water, rising bodies of water, or mudslides entering from outside the structure are classified as 'Floods' and require a dedicated flood policy.
    """
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = [Document(page_content=raw_text)]
    chunks = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        chunks, 
        embeddings, 
        collection_name="insurance_policies"
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": 2}) # Pulls top 2 most relevant chunks

# Initialize the retriever when the module loads
retriever = _build_retriever()

def run_claim(user_input):
    """
    Retrieves policy context and runs adjudication logic.
    """
    # 1. Fetch matching documents from the vector database
    matched_docs = retriever.invoke(user_input)
    extracted_chunks = [doc.page_content.strip() for doc in matched_docs]
    
    # Simulate processing time for the agent's graph logic/LLM thought cycle
    time.sleep(2.5)
    
    # 2. Basic rule evaluation based on keywords for demonstration
    lower_input = user_input.lower()
    if "sump pump" in lower_input or "wear and tear" in lower_input or "old age" in lower_input:
        decision = "Excluded / Denied"
        reasoning = "The claim details specify the breakdown occurred due to 'old age' after 10 years of use. Section 2 of the policy explicitly excludes losses arising from mechanical breakdown, general wear and tear, and gradual deterioration."
    elif "hail" in lower_input or "storm" in lower_input:
        decision = "Approved"
        reasoning = "The damage was caused by a sudden hail storm, which falls directly under Section 1: Comprehensive Coverage (Acts of Nature)."
    else:
        decision = "Under Review"
        reasoning = "The incident requires manual adjuster review to verify external environmental conditions and precise policy boundary definitions."

    # Return structured data to the frontend UI
    return {
        "decision": decision,
        "reasoning": reasoning,
        "chunks": extracted_chunks
    }
