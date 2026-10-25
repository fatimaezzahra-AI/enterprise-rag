import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding

def do_ingest():
    client = QdrantClient(path="./qdrant_db")
    encoder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    COLLECTION_NAME = "enterprise_knowledge"

    # Re-create collection to make sure it's fresh
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    file_role_mapping = {
        "public_policy.txt": "PUBLIC",
        "hr_salaries.txt": "HR",
        "finance_reports.txt": "FINANCE",
    }

    points = []
    point_id = 1
    data_dir = "./data"

    for filename, role in file_role_mapping.items():
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = [c.strip() for c in text.split("\n") if c.strip()]
            embeddings = list(encoder.embed(chunks))

            for chunk, vector in zip(chunks, embeddings):
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={
                            "text": chunk,
                            "role": role,
                            "source": filename,
                        },
                    )
                )
                point_id += 1

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Successfully ingested {len(points)} chunks into Qdrant!")
    client.close()

if __name__ == "__main__":
    do_ingest()