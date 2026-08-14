# 📄 RAG Chat — Document Q&A with Hybrid Search & Reranking

A Retrieval-Augmented Generation (RAG) web app that lets you upload any PDF and ask questions about it. Answers are grounded in the document, cite their source chunks, and the retrieval pipeline is evaluated against a custom test set.

**🔗 Live demo:** [rag-chat-app-kq3hdcglfesmegtkqqsyx6.streamlit.app](https://rag-chat-app-kq3hdcglfesmegtkqqsyx6.streamlit.app)

## What it does

- Upload any PDF and chat with it in natural language
- Answers are grounded only in the uploaded document (no hallucinated facts)
- Every answer shows the exact source chunks and page numbers it was based on
- Retrieval quality is measured with a custom evaluation script, not just eyeballed

## How it works

1. **Ingestion** — the PDF is loaded and split into overlapping ~1000-character chunks
2. **Indexing** — each chunk is embedded (`sentence-transformers/all-MiniLM-L6-v2`, runs locally, no API cost) and stored in a FAISS vector index. A parallel BM25 keyword index is built over the same chunks.
3. **Hybrid retrieval** — a query is run against both the vector index (semantic search) and the BM25 index (exact keyword search), and results are merged. This catches both meaning-based matches and exact terms (IDs, names, acronyms) that pure vector search often misses.
4. **Reranking** — a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescoring pass reorders the merged candidates by actual relevance to the question before the top few are kept.
5. **Generation** — the top chunks are passed to Gemini (`gemini-3.6-flash`) with a prompt that instructs it to answer only from the given context, and to say "I don't know" rather than guess.

## Why hybrid search + reranking

Plain vector search is good at *meaning* but weak on exact terms — form field names, IDs, acronyms, specific numbers. Adding BM25 keyword search alongside it, then reranking the combined results with a cross-encoder, meaningfully improves precision on documents with structured or technical content (tested here on both a research PDF and a real-world administrative form).

## Evaluation

`evaluate.py` runs a small set of test questions with expected keywords against the retrieval pipeline and reports a hit rate per question plus an overall score — currently **83.3% retrieval accuracy** on a 4-question test set for the sample document. Run it yourself:

```bash
python evaluate.py "your_document.pdf"
```

## Tech stack

- **LangChain** — orchestration (loaders, splitters, retrievers, prompt chains)
- **FAISS** — vector similarity search
- **rank_bm25** — keyword-based retrieval
- **sentence-transformers** — local embeddings + cross-encoder reranker (no cost)
- **Google Gemini API** — answer generation (free tier)
- **Streamlit** — web UI, deployed on Streamlit Community Cloud

## Running locally

```bash
git clone https://github.com/RiteshMande45/rag-chat-app.git
cd rag-chat-app
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file with:
```
GOOGLE_API_KEY=your_gemini_api_key
```

Then run:
```bash
streamlit run app.py
```

## Possible extensions

- Multi-document upload (query across several PDFs at once)
- Persistent vector store (currently rebuilt per session)
- Larger, curated evaluation set with precision/recall metrics