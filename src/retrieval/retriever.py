from src.utils.helpers import get_logger
from src.embeddings.embedder import TextEmbedder
from src.vectordb.vector_store import VectorStoreDB

# Initialize our structured enterprise logger
logger = get_logger(__name__)

class SemanticRetriever:
    def __init__(self, vector_db: VectorStoreDB, embedder: TextEmbedder):
        """
        Initializes the retriever with a connected vector database and an embedder.
        """
        self.vector_db = vector_db
        self.embedder = embedder
        logger.info("Initialized Semantic Retriever")

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """
        Embeds the user query, searches ChromaDB, and returns the most relevant text chunks.
        """
        logger.info("Retrieving context for query", extra={"extra_info": {"query": query, "top_k": top_k}})
        
        try:
            # 1. Embed the user's query into a vector
            query_embedding = self.embedder.embed_query(query)
            
            # 2. Query the vector database for the closest matching vectors
            results = self.vector_db.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # 3. Extract the raw text documents from the results
            retrieved_chunks = results['documents'][0]
            
            # 4. Combine the chunks into a single readable string separated by dashes
            context = "\n\n---\n\n".join(retrieved_chunks)
            
            logger.info("Successfully retrieved context", extra={
                "extra_info": {"retrieved_chunks_count": len(retrieved_chunks)}
            })
            return context
            
        except Exception as e:
            logger.error("Failed to retrieve context", extra={"extra_info": {"error": str(e)}})
            raise e