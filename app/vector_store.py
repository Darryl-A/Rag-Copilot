import numpy as np
import faiss
import os
import pickle


def build_faiss_index(embeddings):
    dimension = len(embeddings[0])

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    return index

def save_faiss_index(index, path="storage/index.faiss"):
    faiss.write_index(index, path)

def load_faiss_index(path="storage/index.faiss"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index file not found at {path}. Build the index first.")

    return faiss.read_index(path)

def save_chunks(chunks, path="storage/chunks.pkl"):
    with open(path, 'wb') as f:
        pickle.dump(chunks, f)

def load_chunks(path="storage/chunks.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunks file not found at {path}. Build the chunks first.")

    with open(path, 'rb') as f:
        return pickle.load(f)