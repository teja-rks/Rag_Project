import chromadb
from src.utils.helpers import get_logger

# Initialize the enterprise logger
logger = get_logger(__name__)

class VectorStoreDB:
    def __init__(self, persist_directory: str = "./chroma_data", collection_name: str = "rag_collection"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_db()

    def _initialize_db(self):
        """
        Connects to ChromaDB and gets or creates the target collection.
        """
        logger.info("Initializing Vector Database client",extra={
            "extra_info": {"persist_directory": self.persist_directory}
        })
        try:
            # Create a persistent local database
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Create a new collection or load it if it already exists
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            
            logger.info("Vector Database initialized successfully", extra={
                "extra_info": {"collection_name": self.collection_name}
            })
            
        except Exception as e:
            logger.error("Failed to initialize Vector Database", extra_info={"error": str(e)})
            raise e
            
    def add_data(self, chunks: list, embeddings: list, ids: list, metadata: list = None):
        """
        Inserts chunks and their embeddings into the database.
        """
        logger.info("Inserting chunks into Vector Database", extra={
            "extra_info": {"chunk_count": len(chunks)}
        })
        try:
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadata,
                ids=ids
            )
            logger.info("Successfully inserted chunks into Vector Database")
        except Exception as e:
            logger.error("Failed to insert chunks", extra={
                "extra_info": {"error": str(e)}
            })
            raise e