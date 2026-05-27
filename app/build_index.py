from app.document_loader import load_documents
from app.embeddings import get_embeddings
from app.chunker import chunk_text
from app.vector_store import build_faiss_index, save_faiss_index, save_chunks



def main():
    print("Loading documents...")

    docs = load_documents()

    all_chunks = []

    # Chunk + metadata
    for source, text in docs.items():
        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append((chunk, source))

    print(f"Total chunks created: {len(all_chunks)}")

    # Create embeddings
    print("Creating embeddings...")

    embeddings = get_embeddings(
        [c[0] for c in all_chunks]
    )

    # Build FAISS
    print("Building FAISS index...")

    index = build_faiss_index(embeddings)

    # Save index + metadata
    print("Saving index and chunks...")

    save_faiss_index(index)
    save_chunks(all_chunks)

    print("Index build complete!")


if __name__ == "__main__":
    main()