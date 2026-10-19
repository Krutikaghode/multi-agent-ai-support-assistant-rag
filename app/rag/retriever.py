from typing import Optional
from pinecone import Pinecone
from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, embeddings

def retrieve_medical_products(query: str, category_filter: Optional[str] = None, top_k: int = 4) -> str:
    """Retrieves relevant company & product info from the 47-item medical catalog in Pinecone."""
    if not PINECONE_API_KEY:
        return "Catalog preview mode: Configure PINECONE_API_KEY to query the 47 medical products index."

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    query_vector = embeddings.embed_query(query)

    filter_dict = {"category": {"$eq": category_filter}} if category_filter else None
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict,
    )

    matches = results.get("matches", [])
    if not matches:
        return "No matching medical products found in the 47-product catalog."

    formatted_chunks = []
    for match in matches:
        meta = match.get("metadata", {})
        formatted_chunks.append(
            f"- Product SKU: {meta.get('sku', 'N/A')} | Name: {meta.get('name', 'Unknown')}\n"
            f"  Category: {meta.get('category', 'General')} | Price: {meta.get('price', 'Contact Sales')}\n"
            f"  Clinical Indications: {meta.get('indications', 'N/A')}\n"
            f"  Specifications: {meta.get('specifications', 'N/A')}"
        )
    return "\n\n".join(formatted_chunks)
