# rag_agent_app/backend/vectorstore.py

import os
import uuid
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Import API keys from config (only Pinecone is needed here now)
from config import PINECONE_API_KEY

# Set environment variables for Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define Pinecone index name
INDEX_NAME = os.environ["INDEX_NAME"] 

#--- Function to get a retriever for the vector store ---
def get_retriever():
    """Initializes and returns the Pinecone vector store retriever."""

    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384, 
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1') 
        )
        print(f"Created new Pinecone index: {INDEX_NAME}")
    
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)

# --- Function to add documents to the vector store ---
def add_document_to_vectorstore(documents: list, filename: str):
    """
    Adds a list of documents to the Pinecone vector store.
    Splits each document's text into chunks before embedding and upserting.
    """
    if not documents:
        raise ValueError("Documents list cannot be empty.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    
    processed_docs = []

    for doc in documents:
        print(f"Processing document for vector store: {doc.metadata.get('source', 'unknown source')} (Page {doc.metadata.get('page', 'unknown page')})")
        page_number = doc.metadata.get("page", 0)

        chunks = text_splitter.split_text(doc.page_content)

        for chunk in chunks:
            processed_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "page": page_number
                    }
                )
            )

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    # Add unique IDs
    ids = [str(uuid.uuid4()) for _ in processed_docs]

    vectorstore.add_documents(processed_docs, ids=ids)

    print(f"Successfully indexed {len(processed_docs)} chunks into Pinecone.")