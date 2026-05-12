import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_documents(folder = "data"):
    # Load documents from the specified folder
    documents = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r") as f:
                documents[filename] = f.read()
    return documents


def chunk_text(text, chunk_size = 300):
    sentences = text.split(". ")
    chunks = []

    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def get_embeddings(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]


def build_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index


def search(query, index, chunks, k=5, threshold=0.5):
    query_embedding = get_embeddings([query])[0]

    D, I = index.search(
        np.array([query_embedding]).astype("float32"),
        k
    )

    results = []

    for dist, idx in zip(D[0], I[0]):
        similarity = 1 / (1 + dist)

        if similarity > threshold:
            chunk, source = chunks[idx]
            results.append({
                "text": chunk,
                "source": source,
                "score": similarity
            })

    return results


def generate_answer(query, retrieved_chunks):
    print("\nDEBUG - retrieved_chunks passed to LLM:")
    print(retrieved_chunks)
    
    if not retrieved_chunks:
        return "I don't know based on the provided documents."

    context = "\n\n".join(
        [f"Source: {r['source']}\nText: {r['text']}" for r in retrieved_chunks]
    )

    prompt = f"""
            You are an AI assistant. Answer the question using only the context below.
            If the answer is not in the context, say: "I don't know based on the provided documents."

    Context:
    {context}

    Question:
    {query}

    Answer clearly and concisely.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    docs = load_documents()

    all_chunks = []

    # 1. chunk + metadata
    for source, text in docs.items():
        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append((chunk, source))

    print(f"Total chunks: {len(all_chunks)}")

    # 2. embeddings (ONLY text)
    embeddings = get_embeddings([c[0] for c in all_chunks])

    # 3. build FAISS
    index = build_faiss_index(embeddings)

    print("Index built successfully!")

    # 4. test query
    query = "What is overfitting?"
    results = search(query, index, all_chunks)

    print("\nQuery:", query)

    print("\nRetrieved chunks:\n")
    for i, r in enumerate(results):
        print(f"\nResult {i+1}")
        print("Source:", r["source"])
        print("Score:", round(r["score"], 2))
        print("Text:", r["text"][:300])
        print("-" * 50)

    answer = generate_answer(query, results)

    print("\nFinal Answer:\n")
    print(answer)