from app.vector_store import (load_chunks, load_faiss_index)
from app.retriever import search
from app.generator import generate_answer

def main():

    print("Loading FAISS index and chunks...")

    index = load_faiss_index()
    chunks = load_chunks()

    print("Ready to answer questions")

    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        retrieved_chunks = search(query, index, chunks)
        print("DEBUG retrieved chunks:", retrieved_chunks)
        for i, chunk in enumerate(retrieved_chunks):
            print(f"\nResult {i+1})")
            print("Source:" + chunk['source'])
            print("Score:", round(chunk['score'], 2))
            print("Text:", chunk['text'][:300])
            print("-" * 50)

        answer = generate_answer(query, retrieved_chunks)
        print("\nGenerated Answer:")
        print(answer)

if __name__ == "__main__":
    main()

