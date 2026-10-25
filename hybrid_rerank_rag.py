import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# -------------------------------------------------------------------
# 1. INITIALIZING MODELS & CLIENTS
# -------------------------------------------------------------------
print(" Loading models...")
# Dense Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Reranking Model (Cross-Encoder for precision scoring)
reranker_model = CrossEncoder("BAAI/bge-reranker-base")

# Qdrant Client (In-Memory execution)
qdrant_client = QdrantClient(":memory:")
COLLECTION_NAME = "hybrid_rag_collection"

# LLM Client (Groq API)
groq_client = Groq(api_key="GROQ_API_KEY")

# -------------------------------------------------------------------
# 2. DATA LOADING & PREPARATION FROM LOCAL FOLDER
# -------------------------------------------------------------------
DATA_FOLDER = "data" 

documents = []
doc_id = 1

# Load all .txt files from the local directory
for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".txt"):
        filepath = os.path.join(DATA_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:  # Ensure the file is not empty
                documents.append({"id": doc_id, "text": text})
                doc_id += 1

print(f" Loaded {len(documents)} document(s) from '{DATA_FOLDER}' directory.")

# A. Qdrant Vector Indexing (Clean Collection Setup)
if qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.delete_collection(COLLECTION_NAME)

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

points = []
for doc in documents:
    vec = embedding_model.encode(doc["text"]).tolist()
    points.append(PointStruct(id=doc["id"], vector=vec, payload={"text": doc["text"]}))

qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

# B. BM25 Keyword Indexing
corpus = [doc["text"] for doc in documents]
tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

print(" Data & Indexing complete!")

# -------------------------------------------------------------------
# 3. HYBRID SEARCH (VECTOR + BM25)
# -------------------------------------------------------------------
def hybrid_search(query: str, top_k: int = 4):
    # 1. Vector Search (Qdrant)
    query_vector = embedding_model.encode(query).tolist()
    qdrant_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    vector_docs = [hit.payload["text"] for hit in qdrant_results.points]
    
    # 2. BM25 Search (Keywords)
    tokenized_query = query.lower().split(" ")
    bm25_docs = bm25.get_top_n(tokenized_query, corpus, n=top_k)

    # 3. Candidate Fusion & Deduplication
    combined_docs = list(set(vector_docs + bm25_docs))
    return combined_docs

# -------------------------------------------------------------------
# 4. RERANKING FUNCTION
# -------------------------------------------------------------------
def rerank(query: str, candidate_docs: list[str], top_n: int = 2):
    if not candidate_docs:
        return []
        
    # Pair creation: (query, candidate_text)
    pairs = [[query, doc] for doc in candidate_docs]
    
    # Predict relevance scores using Cross-Encoder
    scores = reranker_model.predict(pairs)
    
    # Sort candidates by score descending
    doc_score_pairs = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
    
    # Extract top N documents
    top_docs = [doc for doc, score in doc_score_pairs[:top_n]]
    return top_docs

# -------------------------------------------------------------------
# 5. END-TO-END RAG PIPELINE
# -------------------------------------------------------------------
def run_hybrid_rag(user_query: str):
    print(f"\n Query: {user_query}")
    
    # Step 1: Hybrid Search (Retrieve top candidates)
    candidates = hybrid_search(user_query, top_k=3)
    print(f" Candidates retrieved (Vector + BM25): {len(candidates)}")
    
    # Step 2: Reranking (Select best context chunks)
    final_context_chunks = rerank(user_query, candidates, top_n=2)
    print(f" Top {len(final_context_chunks)} Reranked Context Chunks:")
    for i, chunk in enumerate(final_context_chunks, 1):
        print(f"   [{i}] {chunk}")
        
    # Step 3: LLM Generation (Groq)
    context_text = "\n".join(final_context_chunks)
    prompt = f"""You are an AI assistant. Answer the question in English based ONLY on the context provided below.
Do NOT use LaTeX or backslashes (like \\times) for mathematical calculations. Use plain text formatting.

Context:
{context_text}

Question: {user_query}
Answer:"""

    response = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    print("\n LLM Response:")
    print(response.choices[0].message.content)

# PIPELINE EXECUTION & TESTING
if __name__ == "__main__":
    run_hybrid_rag("What is the annual salary scale for an AI Engineer in 2026?")
    run_hybrid_rag("How much profit will be allocated to AI infrastructure?")