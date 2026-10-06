# Hybrid search: dense plus keyword

Retrieval-augmented generation (RAG) starts by finding passages related to the user's question. There are two classic ways to do this.

## Dense retrieval

An embedding encoder turns every chunk into a vector. At query time, the query is embedded with the same encoder and the nearest vectors are returned. Dense retrieval understands meaning: "car" matches "automobile". Common vector indexes are FAISS, Chroma, Qdrant and pgvector. Small embedding models such as all-MiniLM-L6-v2 run on a laptop CPU.

## Keyword retrieval with BM25

BM25 scores documents by how often the query words appear in them, adjusted for document length and for how rare each word is across the collection. It is great at exact matches: product codes, error messages, names and acronyms that an embedding model may blur together. BM25 needs no model at all, only the raw text.

## Merging with reciprocal rank fusion

Dense and keyword search return two ranked lists with scores on different scales, so you cannot just add the scores. Reciprocal rank fusion (RRF) ignores the raw scores and uses only the positions. Each passage gets 1 / (k + rank) from every list it appears in, with k usually set to 60, and the totals are sorted. A passage that ranks well in both lists rises to the top.

## How many candidates to keep

Hybrid search should return more candidates than the LLM needs, typically 20 to 50, and let a reranker choose the best few. Recall matters more than precision at this stage, because the reranker can only reorder what it is given.
