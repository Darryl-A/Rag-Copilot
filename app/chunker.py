from app.config import CHUNK_SIZE


def chunk_text(text, chunk_size=CHUNK_SIZE):
    sentences = text.split(". ")
    chunks = []

    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks