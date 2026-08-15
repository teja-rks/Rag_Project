from langchain_core.prompts import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
from src.utils.helpers import get_logger

# Initialize structured enterprise logger
logger = get_logger(__name__)

class RAGPrompt:
    @staticmethod
    def get_qa_prompt() -> ChatPromptTemplate:
        """
        Returns a ChatPromptTemplate that binds the retrieved context 
        and the user's question into a strict instruction set.
        """
        logger.info("Initializing QA Prompt Template")
        
        # The System prompt strictly limits the LLM's knowledge to our retrieved context
        system_template = """You are a highly technical, precise AI assistant.
Your goal is to answer the user's question accurately using ONLY the context provided below.

If the answer is not contained within the context, simply state: "I don't know based on the provided context." 
Do not hallucinate or pull information from your general training data.

Context:
{context}
"""
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
        
        # Combine into a final chat prompt structure
        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])