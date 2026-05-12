from app.document_loader import load_documents
from app.chunker import chunk_text
from app.embeddings import get_embeddings
from app.vector_store import build_faiss_index
from app.retriever import search
from app.generator import generate_answer


def main():
    # 1. Load documents
    docs = load_documents()

    # 2. Chunk + metadata
    all_chunks = []

    for source, text in docs.items():
        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append((chunk, source))

    print(f"Total chunks: {len(all_chunks)}")

    # 3. Embeddings
    embeddings = get_embeddings(
        [c[0] for c in all_chunks]
    )

    print("Embeddings created")

    # 4. Build FAISS index
    index = build_faiss_index(embeddings)

    print("FAISS index built")

    # 5. Query
    query = "What is overfitting?"

    retrieved_chunks = search(
        query,
        index,
        all_chunks
    )

    print("\nRetrieved Chunks:\n")

    for i, r in enumerate(retrieved_chunks):
        print(f"\nResult {i+1}")
        print("Source:", r["source"])
        print("Score:", round(r["score"], 2))
        print("Text:", r["text"][:300])
        print("-" * 50)

    # 6. Generate answer
    answer = generate_answer(
        query,
        retrieved_chunks
    )

    print("\nFinal Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()