import os
from src.utils.helpers import get_logger
from src.ingestion.loader import DataIngestor
from src.chunking.chunker import TextChunker
from src.embeddings.embedder import TextEmbedder
from src.vectordb.vector_store import VectorStoreDB

# Initialize logger for test execution
logger = get_logger("IntegrationTest")

def run_test():
    print("\n==================================================")
    print("🚀 STARTING END-TO-END RAG PIPELINE TEST")
    print("==================================================\n")
    
    # --- 1. Test Ingestion ---
    logger.info("Step 1: Testing Document Ingestion")
    data_dir = "data"
    ingestor = DataIngestor(data_directory=data_dir)
    raw_docs = ingestor.load_documents()
    print(f"[1/4] ✅ Ingestion Success: Loaded {len(raw_docs)} document(s) from '{data_dir}/'")

    # --- 2. Test Chunking ---
    logger.info("Step 2: Testing Document Chunking")
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    chunked_docs = chunker.split_documents(raw_docs)
    print(f"[2/4] ✅ Chunking Success: Created {len(chunked_docs)} chunk(s)")

    # --- 3. Test Embeddings ---
    logger.info("Step 3: Testing Embedding Generation")
    embedder = TextEmbedder()
    chunk_texts = [doc.page_content for doc in chunked_docs]
    embeddings = embedder.embed_documents(chunk_texts)
    print(f"[3/4] ✅ Embedding Success: Generated {len(embeddings)} vector(s) of dimension {len(embeddings[0])}")

    # --- 4. Test Vector DB Storage ---
    logger.info("Step 4: Testing Vector DB Storage")
    vector_db = VectorStoreDB(persist_directory="./chroma_data", collection_name="test_collection")
    
    chunk_ids = [f"chunk_id_{i}" for i in range(len(chunked_docs))]
    chunk_metadata = [doc.metadata for doc in chunked_docs]

    vector_db.add_data(
        chunks=chunk_texts,
        embeddings=embeddings,
        ids=chunk_ids,
        metadata=chunk_metadata
    )
    print(f"[4/4] ✅ Vector Store Success: Stored {len(chunk_texts)} chunk vectors in ChromaDB")

    print("\n==================================================")
    print("🎉 PIPELINE INTEGRATION TEST PASSED SUCCESSFULLY!")
    print("==================================================\n")

if __name__ == "__main__":
    run_test()