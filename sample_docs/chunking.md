# Chunking documents for RAG

Documents are too long to embed or send to an LLM in one piece, so they are split into chunks before indexing.

## Chunk size

Small chunks (200 to 500 characters) give precise matches but can lose context. Large chunks (1,000 to 2,000 characters) keep more context but dilute the embedding, so a chunk about many topics matches none of them well. A size around 500 characters with a 50 character overlap is a reasonable starting point for question answering over articles and notes.

## Recursive character splitting

LangChain's RecursiveCharacterTextSplitter tries to split on paragraph breaks first, then on line breaks, then on sentences and finally on single characters. This keeps paragraphs together whenever they fit, which produces chunks that read naturally.

## Overlap

Overlap repeats the end of one chunk at the start of the next one. It stops an answer from being cut in half at a chunk boundary. Too much overlap wastes storage and returns near duplicate candidates.

## Ingestion runs once

Chunking, embedding and indexing happen once, when documents are added. Only the query path runs on every question, which is why ingestion can afford slower but better processing.
