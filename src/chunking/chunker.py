from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.helpers import get_logger

# Initialize our structured enterprise logger
logger = get_logger(__name__)

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the chunker. 
        - chunk_size: The maximum number of characters in each chunk.
        - chunk_overlap: How many characters to overlap between chunks to maintain context.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize the LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split_documents(self, documents: list) -> list:
        """
        Takes a list of raw documents and splits them into smaller overlapping chunks.
        """
        logger.info("Starting document chunking", extra={
            "extra_info": {
                "chunk_size": self.chunk_size, 
                "chunk_overlap": self.chunk_overlap,
                "document_count": len(documents)
            }
        })
        
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            logger.info("Successfully chunked documents", extra={
                "extra_info": {
                    "total_chunks_created": len(chunks)
                }
            })
            return chunks
            
        except Exception as e:
            logger.error("Failed to chunk documents", extra={
                "extra_info": {"error": str(e)}
            })
            raise e