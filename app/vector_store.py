import numpy as np
import faiss


def build_faiss_index(embeddings):
    dimension = len(embeddings[0])

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    return index