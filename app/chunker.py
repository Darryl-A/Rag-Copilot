from app.config import CHUNK_SIZE


def chunk_text(text, chunk_size=300, overlap_limit=60):
    sentences = text.split(". ")
    chunks = []

    current_chunk = []

    for sentence in sentences:
        sentence = sentence.strip()

        candidate_chunk = ". ".join(current_chunk + [sentence]) + "."

        if len(candidate_chunk) <= chunk_size:
            current_chunk.append(sentence)

        else:
            #Save the current chunk first
            completed_chunk = ". ".join(current_chunk) + "."
            chunks.append(completed_chunk)

            # To keep more context we will add some sentence from previous chunk to next chunk
            overlap_sentences = []
            overlap_length = 0

            for previous_sentence in reversed(current_chunk):

                sentence_length = len(previous_sentence)

                if overlap_sentences:
                    sentence_length += 2

                if overlap_length + sentence_length > overlap_limit:
                    break

                overlap_sentences.insert(0, previous_sentence)
                overlap_length += sentence_length

            # Start next chunk
            current_chunk = overlap_sentences + [sentence]

    if current_chunk:
        completed_chunk = ". ".join(current_chunk) + "."
        chunks.append(completed_chunk)
    return chunks