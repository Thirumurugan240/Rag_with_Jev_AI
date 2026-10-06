# RAG powered by Jev

A beginner-friendly RAG app. You ask a question about your documents, and the app answers using **only** what is written in those documents.

The whole project is two small files:

| File | What it does |
|---|---|
| `rag.py` | The brain. Reads the documents, finds the right chunks, asks Jev to check them, and gets the answer from OpenAI. |
| `app.py` | The web page (Streamlit). A text box for your question, and the answer with Jev's scores. |
| `sample_docs/` | The documents the app knows about. Put your own `.md` or `.pdf` files here. |

---

## 1. The big idea

### What is RAG?

An LLM like ChatGPT doesn't know what is in *your* files. RAG (Retrieval-Augmented Generation) fixes that in two moves:

1. **Retrieve:** search your documents and find the pieces related to the question.
2. **Generate:** give only those pieces to the LLM and ask it to answer from them.

Think of an open-book exam: first you find the right pages, then you write the answer from those pages.

### The problem: search is not perfect

Search returns 20 chunks, but usually only 2 or 3 of them really answer the question. The rest are "same topic, wrong answer". If you send all 20 to the LLM, it can get confused or quote the wrong one.

### The fix: Jev

**Jev** is a model from TypeSafe that doesn't write text. It only answers **yes/no questions**, and for each one it gives a **probability** between 0 and 1.

We ask Jev, for every chunk: *"Does this chunk help answer the question?"*

- Chunk C1: 0.94 (yes, keep it)
- Chunk C2: 0.87 (yes, keep it)
- Chunk C4: 0.38 (no, drop it)

Jev's probabilities are **calibrated**, which means 0.9 really means "about 90% sure" on every question. So we can use one fixed cutoff (0.5) and trust it.

In the same request we also ask two safety questions, called **gates**:

- **answerable:** do these chunks contain the answer at all? If not, we reply "Not in the docs." and don't call the LLM.
- **injection:** is any chunk trying to give orders to the AI (a prompt injection)? If yes, we stop.

---

## 2. How the app works

```
                RUNS ONCE (when the app starts)
 ┌───────────┐    ┌──────────┐    ┌──────────────────────┐
 │ Documents │ -> │  Chunks  │ -> │ Vector index (FAISS) │  meaning
 └───────────┘    └──────────┘    │ BM25 index           │  keywords
                                  └──────────────────────┘

                RUNS FOR EVERY QUESTION
 Question -> Hybrid search -> 20 chunks -> Jev scores each one -> keep p >= 0.5
                                                    |
                                       gates: answerable? injection?
                                                    |
                              pass: OpenAI answers from kept chunks only
                              fail: "Not in the docs."
```

---

## 3. Walk through `rag.py`, step by step

The file has 7 numbered steps. Open it next to this README.

### Runs once, when the app starts

**Step 1: Load the documents.**
Reads every `.md` and `.pdf` file in `sample_docs/` into memory.

**Step 2: Cut them into chunks.**
A full document is too big to search well, so we split it into pieces of about 500 characters. Neighbouring chunks share 50 characters (the "overlap") so a sentence is never cut in half and lost.

**Step 3: Build two indexes.**
An index is like the index at the back of a book: it makes searching fast.
- **Vector index (FAISS):** each chunk is turned into a list of numbers (an "embedding") that captures its *meaning*. "car" and "automobile" end up close together. The embedding model `all-MiniLM-L6-v2` is free and runs on your own computer.
- **BM25 index:** classic keyword search. Great for exact words, names and codes.

We also create the Jev client, the OpenAI model (`gpt-6.1-sol`), and the cutoff `TAU = 0.5`.

### Runs for every question (inside `ask(question)`)

**Step 4: Hybrid search.**
We take the top 20 from the vector index and the top 20 from BM25, then merge them with **reciprocal rank fusion**: a chunk earns `1 / (60 + rank)` points from each list it appears in. Chunks that rank high in *both* lists win. We keep the best 20 and label them C1 to C20.

**Step 5: Ask Jev.**
We send Jev one request containing:
- the **state**: the question and the 20 chunks
- **22 yes/no questions**: "Does C1 help answer?" ... "Does C20 help answer?", plus "answerable?" and "injection?"

Jev answers all 22 at once and returns a probability for each.

**Step 6: Keep the good chunks.**
Only chunks with a probability of 0.5 or more are kept. Everything else is dropped before the LLM ever sees it.

**Step 7: Check the gates, then answer.**
- Injection probability 0.5 or more: stop.
- Answerable below 0.5, or no chunk kept: reply "Not in the docs." (no LLM call, no cost).
- Otherwise: OpenAI gets the question plus **only the kept chunks**, and writes an answer that cites them like `[C1]`.

`ask()` returns four things to the web page: the chunks, Jev's scores, the kept chunk ids, and the answer.

---

## 4. Walk through `app.py`

```python
question = st.text_input("Ask a question")        # a text box
passages, p, kept, answer = rag.ask(question)      # run steps 4 to 7
st.write(answer)                                   # show the answer
st.progress(p[cid], ...)                           # one bar per chunk
```

`import rag` runs steps 1 to 3 once. After that, every new question only runs `ask()`.

---

## 5. Run it

You need two API keys:
- **TypeSafe (Jev):** https://console.typesafe.ai/settings/keys
- **OpenAI:** https://platform.openai.com/api-keys

Then, in PowerShell:

```powershell
cd jev_with_rag
python -m venv .venv                 # skip if the .venv folder already exists
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env                         # paste both keys, save
streamlit run app.py
```

The browser opens at http://localhost:8501. The first start takes a minute because it downloads the small embedding model.

---

## 6. Try these questions

| Question | What you should see |
|---|---|
| Which reranker fits a RAG pipeline? | Several reranker chunks kept, answer cites them |
| What is reciprocal rank fusion? | Chunks from `hybrid_search.md` kept |
| How long should idli batter ferment? | Only the cooking chunks kept |
| What is the capital of France? | All bars low, "Not in the docs." |

`south_indian_cooking.md` is in the sample docs on purpose. For RAG questions, watch Jev drop those chunks.

---

## 7. Change things

| I want to... | Do this |
|---|---|
| Use my own documents | Put `.md` or `.pdf` files in `sample_docs/` and restart the app |
| Keep more or fewer chunks | Change `TAU = 0.5` in `rag.py` (lower keeps more) |
| Use a cheaper OpenAI model | Change `gpt-6.1-sol` to `gpt-6-luna` in `rag.py` |
| Make chunks bigger | Change `chunk_size=500` in `rag.py` |

---

## 8. Common problems

| Problem | Fix |
|---|---|
| Error about `TYPESAFE_API_KEY` or `OPENAI_API_KEY` | The `.env` file is missing or a key is empty |
| "0 chunks" or an empty index error | Run `streamlit run app.py` from **inside** the `jev_with_rag` folder |
| `ModuleNotFoundError` | Activate the venv first: `.venv\Scripts\activate` |
| Every answer is "Not in the docs." | The answer isn't in your documents, or try a lower `TAU` |

---

## Words used in this project

| Word | Meaning |
|---|---|
| Chunk | A small piece of a document (about 500 characters) |
| Embedding | A list of numbers that represents the meaning of a text |
| Vector index | A store of embeddings that finds chunks with similar meaning |
| BM25 | Keyword search that scores chunks by matching words |
| Hybrid search | Using vector and keyword search together |
| Reciprocal rank fusion | A simple way to merge two ranked lists into one |
| Reranker | A second model that re-checks search results and puts the best first |
| Calibrated probability | A score you can trust: 0.9 means about 90% likely |
| Gate | A yes/no safety check that decides whether to call the LLM at all |
| τ (tau) | The cutoff score. Here it is 0.5 |
