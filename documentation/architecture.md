# Architecture

## Current Pipeline
Documents → Chunking → Embeddings → FAISS → Retrieval → LLM

## Key Engineering Decisions
- Sentence-aware chunking (Initially word based chunking caused loss of context)
- Metadata-aware retrieval
- Similarity threshold filtering (Added to remove accuracy of retrieval)

## Planned Improvements
- Persistent vector storage
- Chunk overlap
- Reranking
- PDF ingestion