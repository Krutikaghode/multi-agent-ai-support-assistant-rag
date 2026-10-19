import json
from pinecone import Pinecone, ServerlessSpec
from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, embeddings

def ingest_catalog(catalog_path: str = "data/medical_products_47.json"):
    with open(catalog_path, "r") as f:
        products = json.load(f)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(PINECONE_INDEX_NAME)
    vectors = []
    for item in products:
        text = f"{item['name']} - {item['category']}. {item['indications']} {item['specifications']}"
        vec = embeddings.embed_query(text)
        vectors.append({"id": item["sku"], "values": vec, "metadata": item})

    index.upsert(vectors=vectors)
    print(f"Successfully indexed {len(vectors)} medical products into Pinecone.")

if __name__ == "__main__":
    ingest_catalog()
