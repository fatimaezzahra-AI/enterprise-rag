from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 1. Initialize local Qdrant client and load the embedding model
client = QdrantClient(path="./qdrant_db")
encoder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

collection_name = "enterprise_knowledge"

def search_knowledge(user_query: str, user_role: str):
    print(f"\n User Query: '{user_query}' |  User Role: '{user_role}'")
    print("-" * 60)
    
    # 2. Convert the user prompt into a numerical vector embedding
    query_vector = list(encoder.embed([user_query]))[0]

    # 3. Query Qdrant with Cosine Similarity and RBAC payload filter
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="role",
                    match=models.MatchValue(value=user_role)
                )
            ]
        ),
        limit=2  # Retrieve top 2 most relevant matches
    )

    # 4. Process and display retrieved search results
    if not results:
        print(" Access Denied or No relevant information found.")
        return

    for hit in results:
        print(f" Score (Similarity): {hit.score:.4f}")
        print(f" Role Allowed: {hit.payload['role']}")
        print(f" Text: {hit.payload['text']}")
        print("-" * 60)

# --- Execute RBAC access verification tests ---

# Test 1: HR role querying engineering data (Should fail/return empty if access restricted)
search_knowledge(
    user_query="What are the tech stack and engineering updates?", 
    user_role="hr"
)

# Test 2: Engineer role querying engineering data (Should match and return context)
search_knowledge(
    user_query="What are the tech stack and engineering updates?", 
    user_role="engineer"
)