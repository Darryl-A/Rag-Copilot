from app.config import client, EMBEDDING_MODEL


def get_embeddings(texts):
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )

    return [item.embedding for item in response.data]