from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.utils.helpers import get_logger
from src.embeddings.embedder import TextEmbedder
from src.vectordb.vector_store import VectorStoreDB
from src.retrieval.retriever import SemanticRetriever
from src.prompts.prompt_templates import RAGPrompt
from src.llm.llm_client import LLMClient

logger = get_logger("API_Routes")
router = APIRouter()

# Initialize pipeline components once at startup
try:
    embedder = TextEmbedder()
    vector_db = VectorStoreDB(persist_directory="./chroma_data", collection_name="test_collection")
    retriever = SemanticRetriever(vector_db=vector_db, embedder=embedder)
    prompt_template = RAGPrompt.get_qa_prompt()
    llm_client = LLMClient(model_name="llama3")
except Exception as e:
    logger.error("Failed to initialize RAG components for API", extra={"extra_info": {"error": str(e)}})
    raise e

class QueryRequest(BaseModel):
    question: str
    top_k: int = 2

class QueryResponse(BaseModel):
    question: str
    answer: str
    context: str

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Receives user query, retrieves relevant context, and returns LLM generation.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    logger.info("Received query via API", extra={"extra_info": {"question": request.question}})
    
    try:
        # 1. Retrieve context
        context = retriever.retrieve_context(query=request.question, top_k=request.top_k)
        
        # 2. Format prompt
        formatted_prompt = prompt_template.format_messages(
            context=context,
            question=request.question
        )
        
        # 3. Generate answer
        answer = llm_client.generate_response(formatted_prompt)
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            context=context
        )
    except Exception as e:
        logger.error("Error during RAG execution", extra={"extra_info": {"error": str(e)}})
        raise HTTPException(status_code=500, detail=str(e))