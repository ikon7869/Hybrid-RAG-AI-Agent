import json
import os
import time
from typing import List, Dict, Any
import tempfile

from fastapi import FastAPI, HTTPException, status, UploadFile, File
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader



from agent import rag_agent
from vectorstore import add_document_to_vectorstore

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph RAG Agent API",
    description="API for the LangGraph-powered RAG agent with Pinecone and Groq.",
    version="1.0.0",
)

# In-memory session manager for LangGraph checkpoints (for demonstration)
memory = MemorySaver()

# --- Pydantic Models for API ---
class TraceEvent(BaseModel):
    step: int
    node_name: str
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    event_type: str

class QueryRequest(BaseModel):
    session_id: str
    query: str
    restricted_mode: bool = False 

class AgentResponse(BaseModel):
    response: str
    trace_events: List[TraceEvent] = Field(default_factory=list)

class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    processed_chunks: int

# --- Document Upload Endpoint ---
@app.post("/upload-document/", response_model=DocumentUploadResponse, status_code=status.HTTP_200_OK)
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a PDF document, extracts text, and adds it to the RAG knowledge base.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        file_content = await file.read()
        tmp_file.write(file_content)
        temp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        total_chunks_added = 0
        if documents:
            add_document_to_vectorstore(documents, filename=file.filename)
            total_chunks_added = len(documents)
        
        return DocumentUploadResponse(
            message=f"PDF '{file.filename}' successfully uploaded and indexed.",
            filename=file.filename,
            processed_chunks=total_chunks_added
        )
    except Exception as e:
        print(f"Error processing PDF document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {e}"
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")



# --- Chat Endpoint ---
@app.post("/chat/", response_model=AgentResponse)
async def chat_with_agent(request: QueryRequest):
    trace_events_for_frontend: List[TraceEvent] = []
    print(json.dumps(request.dict(), indent=2))  
    try:

        config = {
            "configurable": {
                "thread_id": request.session_id,
                "restricted_mode": request.restricted_mode
            }
        }
        inputs = {"messages": [HumanMessage(content=request.query)]}

        final_message = ""
        
        print(f"--- Starting Agent Stream for session {request.session_id} ---")
        print(f"Restricted Mode Enabled: {request.restricted_mode}")

        for i, s in enumerate(rag_agent.stream(inputs, config=config)):
            current_node_name = None
            node_output_state = None

            if '__end__' in s:
                current_node_name = '__end__'
                node_output_state = s['__end__']
            else:
                current_node_name = list(s.keys())[0] 
                node_output_state = s[current_node_name]

            event_description = f"Executing node: {current_node_name}"
            event_details = {}
            event_type = "generic_node_execution"

            if current_node_name == "router":
                route_decision = node_output_state.get('route')
                initial_decision = node_output_state.get('initial_router_decision', route_decision)
                override_reason = node_output_state.get('router_override_reason', None)

                if override_reason:
                    event_description = f"Router initially decided: '{initial_decision}'. Overridden to: '{route_decision}' because {override_reason}."
                    event_details = {"initial_decision": initial_decision, "final_decision": route_decision, "override_reason": override_reason}
                else:
                    event_description = f"Router decided: '{route_decision}'"
                    event_details = {"decision": route_decision, "reason": "Based on initial query analysis."}
                event_type = "router_decision"

            elif current_node_name == "rag_lookup":

                rag_chunks = node_output_state.get("rag_chunks", [])
                num_chunks = len(rag_chunks)

                if num_chunks == 0:
                    event_description = "RAG lookup performed. No relevant document chunks found."
                    event_details = {"num_chunks": 0}

                else:

                    preview_chunks = []

                    for chunk in rag_chunks[:3]:
                        lines = chunk.split("\n")
                        header = lines[0] 
                        content_preview = lines[1][:350] + "..." if len(lines) > 1 else ""

                        preview_chunks.append(f"{header}\n{content_preview}")

                    formatted_sources = "\n\n".join(preview_chunks)

                    event_description = f"RAG lookup retrieved {num_chunks} document chunks."
                    event_details = {
                        "num_chunks": num_chunks,
                        "preview_sources": formatted_sources
                    }

                event_type = "rag_action"


            elif current_node_name == "web_search":

                restricted_mode = request.restricted_mode
                web_results = node_output_state.get("web", [])
                web_count = len(web_results)

                if restricted_mode:
                    event_description = "Web search skipped due to Restricted Mode."
                    event_details = {
                        "reason": "Restricted mode prohibits external data access."
                    }

                elif web_count == 0:
                    event_description = "Web search performed but no relevant results found."
                    event_details = {"num_results": 0}

                else:
                    preview_web = []

                    for web in web_results[:3]:
                        lines = web.split("\n")
                        header = lines[0] 
                        content_preview = lines[1][:300] + "..." if len(lines) > 1 else ""

                        preview_web.append(f"{header}\n{content_preview}")

                    formatted_web_sources = "\n\n".join(preview_web)

                    event_description = f"Web search retrieved {web_count} results."
                    event_details = {
                        "num_results": web_count,
                        "preview_sources": formatted_web_sources
                    }

                event_type = "web_action"

            
            elif current_node_name == "fusion_answer":

                event_description = "Generating final answer using hybrid context with source citations and confidence scoring."
                event_type = "answer_generation"

            elif current_node_name == "__end__":
                event_description = "Agent process completed."
                event_type = "process_end"

            trace_events_for_frontend.append(
                TraceEvent(
                    step=i + 1,
                    node_name=current_node_name,
                    description=event_description,
                    details=event_details,
                    event_type=event_type
                )
            )


        print("--- Stream loop finished. Fetching final state values ---")
        final_state = rag_agent.get_state(config)
        state_messages = final_state.values.get("messages", [])

        # Find the most recent AIMessage in the graph state history
        final_ai_msg = next((m for m in reversed(state_messages) if isinstance(m, AIMessage)), None)

        if final_ai_msg and final_ai_msg.content:
            final_message = final_ai_msg.content
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent did not return a valid response (final AI message not found)."
            )

        return AgentResponse(
            response=final_message,
            trace_events=trace_events_for_frontend
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error during agent invocation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    

@app.get("/health")
async def health_check():
    return {"status": "ok"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)