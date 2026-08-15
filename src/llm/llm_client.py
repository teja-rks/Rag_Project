from langchain_ollama import ChatOllama
from src.utils.helpers import get_logger

# Initialize structured enterprise logger
logger = get_logger(__name__)

class LLMClient:
    def __init__(self, model_name: str = "llama3", temperature: float = 0.0):
        """
        Initializes the connection to local Ollama.
        """
        self.model_name = model_name
        logger.info("Initializing Local LLM Client", extra={"extra_info": {"model_name": self.model_name}})
        
        try:
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=temperature
            )
            logger.info("Local LLM Client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Local LLM Client", extra={"extra_info": {"error": str(e)}})
            raise e
            
    def generate_response(self, prompt_messages) -> str:
        """
        Sends the formatted prompt to local Llama 3 and returns the string response.
        """
        logger.info("Sending request to Local LLM")
        try:
            response = self.llm.invoke(prompt_messages)
            logger.info("Successfully received response from Local LLM", extra={
                "extra_info": {"response_length": len(response.content)}
            })
            return response.content
        except Exception as e:
            logger.error("Local LLM Generation failed", extra={"extra_info": {"error": str(e)}})
            raise e