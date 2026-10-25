import os
from groq import Groq
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models


# 1. GROQ CLIENT INITIALIZATION

GROQ_API_KEY = "GROQ_API_KEY"

# Initializing Groq client with custom timeout and retries
client_groq = Groq(
    api_key=GROQ_API_KEY,
    timeout=15.0,
    max_retries=3
)

def call_llm(prompt: str) -> str:
    response = client_groq.chat.completions.create(
        model="qwen/qwen3.8-27b",  # Active production model
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

# 2. SECURE RBAC RAG PIPELINE CLASS

class SecureRAGPipeline:
    def __init__(self, db_path: str = "./qdrant_db", collection_name: str = "enterprise_knowledge"):
        self.client = QdrantClient(path=db_path)
        self.encoder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.collection_name = collection_name

    def retrieve_context(self, user_query: str, user_role: str, limit: int = 2) -> list[str]:
        query_vector = list(self.encoder.embed([user_query]))[0]

# Qdrant query with payload filtering for Role-Based Access Control (RBAC)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="role",
                        match=models.MatchValue(value=user_role)
                    )
                ]
            ),
            limit=limit
        )

        return [hit.payload["text"] for hit in results.points]

    def generate_response(self, user_query: str, user_role: str) -> str:
        retrieved_contexts = self.retrieve_context(user_query, user_role)

        if not retrieved_contexts:
            return " Access Denied: No relevant documents found matching your clearance role."

        context_str = "\n---\n".join(retrieved_contexts)

        system_prompt = f"""You are a secure corporate assistant. Answer the user's question STRICTLY using only the provided context below.
If the answer cannot be derived from the context, state that you do not have permission or context to answer.

[RETRIEVED CONTEXT]:
{context_str}

[USER QUESTION]:
{user_query}

[ANSWER]:"""

        return call_llm(system_prompt)


# PIPELINE EXECUTION & TESTING

if __name__ == "__main__":
    rag = SecureRAGPipeline()

    query = "What are the details about employee salaries?"
    
    print("=== Test 1: Query with 'PUBLIC' role ===")
    print(rag.generate_response(user_query=query, user_role="PUBLIC"))

    print("\n=== Test 2: Query with 'HR' role ===")
    print(rag.generate_response(user_query=query, user_role="HR"))

    # Gracefully close the client connection
    rag.client.close()