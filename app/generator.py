from app.config import client, CHAT_MODEL


def generate_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return "I don't know based on the provided documents."

    context = "\n\n".join(
        [
            f"Source: {r['source']}\nText: {r['text']}"
            for r in retrieved_chunks
        ]
    )

    prompt = f"""
You are an AI assistant.

Answer the question using ONLY the context below.
If the answer is not present in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content