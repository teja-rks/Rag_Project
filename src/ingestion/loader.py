import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from src.utils.helpers import get_logger

logger = get_logger(__name__)

class DataIngestor:
    def __init__(self, data_directory: str):
        self.data_directory = data_directory

    def load_documents(self) -> list:
        """
        Loads all text (.txt) and PDF (.pdf) documents from the specified directory.
        """
        logger.info("Starting document ingestion", extra={"extra_info": {"directory": self.data_directory}})
        
        if not os.path.exists(self.data_directory):
            logger.error("Data directory not found", extra={"extra_info": {"directory": self.data_directory}})
            raise FileNotFoundError(f"Directory {self.data_directory} does not exist.")

        documents = []

        try:
            # 1. Load all .txt files
            txt_loader = DirectoryLoader(
                self.data_directory, 
                glob="**/*.txt", 
                loader_cls=TextLoader
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)

            # 2. Load all .pdf files
            pdf_loader = DirectoryLoader(
                self.data_directory, 
                glob="**/*.pdf", 
                loader_cls=PyPDFLoader
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            
            logger.info(
                "Successfully loaded documents", 
                extra={
                    "extra_info": {
                        "total_documents": len(documents),
                        "txt_count": len(txt_docs),
                        "pdf_count": len(pdf_docs)
                    }
                }
            )
            return documents
            
        except Exception as e:
            logger.error("Failed to load documents", extra={"extra_info": {"error": str(e)}})
            raise e