import numpy as np

from app.embeddings import get_embeddings
from app.config import TOP_K, SIMILARITY_THRESHOLD


def search(query, index, chunks, k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    query_embedding = get_embeddings([query])[0]

    distances, indices = index.search(
        np.array([query_embedding]).astype("float32"),
        k
    )

    results = []

    for dist, idx in zip(distances[0], indices[0]):
        similarity = 1 / (1 + dist)

        if similarity > threshold:
            chunk, source = chunks[idx]

            results.append({
                "text": chunk,
                "source": source,
                "score": similarity
            })

    return results