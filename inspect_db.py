import chromadb

# 1. Connect to the ChromaDB directory created by the test
client = chromadb.PersistentClient(path="./chroma_data")

# 2. Get the collection we wrote to
collection = client.get_collection(name="test_collection")

# 3. Retrieve all stored items (including embeddings, documents, and metadatas)
results = collection.get(include=["embeddings", "documents", "metadatas"])

print(f"\n📊 Total Items in ChromaDB: {len(results['ids'])}\n")
print("=" * 60)

# for i in range(len(results['ids'])):
#     chunk_id = results['ids'][i]
#     document_text = results['documents'][i]
#     embedding_vector = results['embeddings'][i]
#     metadata = results['metadatas'][i]
    
#     print(f"🔹 Chunk ID: {chunk_id}")
#     print(f"📄 Metadata: {metadata}")
#     print(f"📝 Document Snippet: {document_text[:120]}...")
#     print(f"📐 Vector Dimensions: {len(embedding_vector)}")
#     print(f"🔢 Vector Preview (First 5 floats): {embedding_vector[:5]}")
#     print("-" * 60)