from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.helpers import get_logger

# Initialize structured enterprise logger
logger = get_logger(__name__)

class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the embedding model.
        Default model: 'sentence-transformers/all-MiniLM-L6-v2' (384 dimensions).
        """
        self.model_name = model_name
        logger.info("Initializing HuggingFace embedding model",extra={
    "extra_info": {"model_name": self.model_name}})
        
        try:
            self.embedder = HuggingFaceEmbeddings(model_name=self.model_name)
            logger.info("Embedding model initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize embedding model", extra={
    "extra_info": {"error": str(e)}})
            raise e

    def embed_documents(self, texts: list) -> list:
        """
        Generates vector embeddings for a list of text strings (chunked documents).
        """
        logger.info("Generating embeddings for document chunks", extra={
    "extra_info": {"chunk_count": len(texts)}})
        try:
            embeddings = self.embedder.embed_documents(texts)
            logger.info("Successfully generated document embeddings")
            return embeddings
        except Exception as e:
            logger.error("Failed to generate document embeddings", extra={
    "extra_info": {"error": str(e)}})
            raise e

    def embed_query(self, text: str) -> list:
        """
        Generates a single vector embedding for a user query.
        """
        logger.info("Generating embedding for search query", extra={
    "extra_info": {"query": text}})
        try:
            embedding = self.embedder.embed_query(text)
            logger.info("Successfully generated query embedding")
            return embedding
        except Exception as e:
            logger.error("Failed to generate query embedding", extra={
    "extra_info": {"error": str(e)}})
            raise e