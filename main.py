import sys
from src.utils.helpers import get_logger
from src.embeddings.embedder import TextEmbedder
from src.vectordb.vector_store import VectorStoreDB
from src.retrieval.retriever import SemanticRetriever
from src.prompts.prompt_templates import RAGPrompt
from src.llm.llm_client import LLMClient

# Initialize root logger
logger = get_logger("RAG_Application")

class RAGPipeline:
    def __init__(self):
        logger.info("Initializing complete RAG Pipeline components...")
        
        # 1. Initialize Vector DB and Embedder
        self.embedder = TextEmbedder()
        self.vector_db = VectorStoreDB(persist_directory="./chroma_data", collection_name="test_collection")
        
        # 2. Initialize Retriever
        self.retriever = SemanticRetriever(vector_db=self.vector_db, embedder=self.embedder)
        
        # 3. Initialize Prompt and Local LLM Client
        self.prompt_template = RAGPrompt.get_qa_prompt()
        self.llm_client = LLMClient(model_name="llama3")
        
        logger.info("RAG Pipeline ready for queries.")

    def ask(self, question: str) -> str:
        """
        Executes the full RAG query-to-response lifecycle.
        """
        logger.info("Processing user query", extra={"extra_info": {"question": question}})
        
        # Step A: Retrieve relevant context
        context = self.retriever.retrieve_context(query=question, top_k=2)
        
        # Step B: Format the prompt template with context + question
        formatted_prompt = self.prompt_template.format_messages(
            context=context,
            question=question
        )
        
        # Step C: Generate answer using local Llama 3
        response = self.llm_client.generate_response(formatted_prompt)
        return response

def interactive_session():
    print("\n=======================================================")
    print("🤖 LOCAL RAG PIPELINE (Llama 3 + ChromaDB + MiniLM)")
    print("Type 'exit' or 'quit' to stop.")
    print("=======================================================\n")
    
    rag = RAGPipeline()
    
    while True:
        try:
            query = input("\n🧑 Ask a Question: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("\nExiting session. Goodbye!")
                break
                
            print("\n⏳ Searching vector database and generating response via Llama 3...")
            answer = rag.ask(query)
            
            print("\n💡 Response:")
            print(answer)
            print("-" * 55)
            
        except KeyboardInterrupt:
            print("\nSession stopped.")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")

if __name__ == "__main__":
    interactive_session()