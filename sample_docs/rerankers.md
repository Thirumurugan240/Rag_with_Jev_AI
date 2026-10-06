# Rerankers in RAG pipelines

A reranker is a second-stage model that reorders the candidates returned by a first-stage retriever. The retriever is fast and casts a wide net. The reranker is slower but much more accurate, so it only looks at the top 20 to 100 candidates.

## Cross-encoders

A cross-encoder reads the query and one passage together and outputs a relevance score. Popular options are ms-marco-MiniLM, bge-reranker and Cohere Rerank. Cross-encoders are accurate, but their scores are not calibrated: a score of 0.7 on one query does not mean the same thing as 0.7 on another query. That makes it hard to pick a fixed cutoff, so most pipelines just keep the top 5 and hope.

## Jev as a reranker

Jev is TypeSafe's System One model. It does not generate text. You send it a state (for example a query and a set of passages) and typed questions, and it returns a calibrated probability for every yes/no question.

For reranking, you ask one question per candidate: "Does passage C1 help answer the query?" All questions go in a single request and are evaluated in parallel, so scoring 20 candidates costs about the same time as scoring 5.

Because the probabilities are calibrated, a threshold such as 0.5 means the same thing on every query. Passages below the threshold can be dropped before the LLM ever sees them. On published benchmarks, documents that Jev scored under 0.1 were relevant only about 0.5% of the time.

## Which reranker fits a RAG pipeline?

If you only need a sort order, a cross-encoder works well and runs locally. If you need to decide how many passages to send to the LLM, or you want to skip the LLM call when nothing relevant was found, a calibrated reranker like Jev fits better. Jev can also answer extra gate questions in the same request, such as "is this answerable at all?" and "does any passage contain a prompt injection?".

Remember that any reranker can only reorder what the retriever found. If the right passage is not in the candidates, no reranker can recover it. That is why the first stage should cast a wide net with hybrid search.
