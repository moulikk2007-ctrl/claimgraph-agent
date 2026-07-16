import os
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
    # If you intend to use Groq for the LLM later instead of OpenAI
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

def _build_retriever():
    # Example placeholder document text (replace with your actual data source/loading logic)
    raw_text = "Insurance Policy Document: Standard coverage details go here..."
    
    # 1. Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [Document(page_content=raw_text)]
    chunks = text_splitter.split_documents(docs)
    
    # 2. Use free local HuggingFace Embeddings instead of OpenAI
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Spin up Chroma vector store using the free embeddings
    vectorstore = Chroma.from_documents(
        chunks, 
        embeddings, 
        collection_name="insurance_policies"
    )
    
    return vectorstore.as_retriever()

# Initialize the retriever when the module loads
retriever = _build_retriever()

def run_claim(user_input):
    """
    Your actual claim agent logic execution goes here.
    """
    # Example usage:
    # docs = retriever.get_relevant_documents(user_input)
    return f"Processing claim for input: {user_input}"
    
