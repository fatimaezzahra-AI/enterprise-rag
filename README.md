# Enterprise RAG: Hybrid Retrieval, Re-ranking & Automated Evaluation
 
A Retrieval-Augmented Generation (RAG) pipeline that answers questions over internal company documents. It combines **keyword search (BM25)** and **dense vector search (Qdrant)**, **re-ranks** the candidates with a cross-encoder, generates the answer with an LLM on **Groq**, and is evaluated automatically with **RAGAS**.
 
## How it works
 
1. **Ingestion**: documents in `data/` are split into chunks and indexed in Qdrant (dense embeddings) and BM25 (sparse).
2. **Hybrid search**: a query retrieves top candidates from both retrievers.
3. **Re-ranking**: a cross-encoder scores each (query, chunk) pair and keeps the best chunks.
4. **Generation**: the LLM answers using **only** the retrieved context.
5. **Evaluation**: RAGAS measures Faithfulness, Answer Relevancy and Context Precision on a test set.
 
## Tech stack
 
Python · Qdrant · BM25 (`rank_bm25`) · Sentence-Transformers (embeddings + cross-encoder) · Groq API · RAGAS · [Streamlit]
 
## Project structure
 
```
enterprise-rag/
├── data/                  # [sample documents: finance, HR salaries, public policy]
├── ingest.py              # [indexes documents]
├── hybrid_rerank_rag.py   # hybrid search, re-ranking, answer generation
├── query.py               # [CLI to ask questions]
├── app.py                 # [Streamlit UI]
├── eval.py                # RAGAS evaluation script
├── check_models.py        # [model check]
├── requirements.txt
└── .env.example           # template for your API key
```
 
## Getting started
 
### 1. Install
 
```bash
git clone https://github.com/fatimaezzahra-AI/<repo-name>.git
cd <repo-name>
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```
 
### 2. Get a Groq API key
 
1. Create a free account at [console.groq.com](https://console.groq.com).
2. Go to **API Keys** → **Create API Key** and copy it.
3. Copy the template and add your key:
 
```bash
   cp .env.example .env          # Windows PowerShell: Copy-Item .env.example .env
```
 
   Then edit `.env`:
 
```
   GROQ_API_KEY=your_key_here
```
 
Never commit your key. `.env` is listed in `.gitignore`.
 
### 3. Run
 
```bash
python ingest.py                # index the documents
python query.py                 # [ask a question]
streamlit run app.py            # [web UI]
python eval.py                  # run the RAGAS evaluation
```
 
## Evaluation
 
`eval.py` runs the pipeline on a set of questions with reference answers and scores it with RAGAS:
 
| Metric | What it measures |
|---|---|
| Faithfulness | Is the answer supported by the retrieved context? |
| Answer Relevancy | Does the answer address the question? |
| Context Precision | Is the retrieved context relevant? |
 
Results on a [12]-question test set over the sample documents: `[fill in after your final run]`.
 
> **Note:** the test set is small and the sample documents are limited, so these scores are a sanity check, not a benchmark.
 
## Roadmap
 
- [ ] Larger evaluation set on real-world documents (e.g. annual reports)
- [ ] [Self-correction loop / agentic workflow]
- [ ] [Your next step]
 
## Author
 
**Fatima Ezzahra Boukhorssa**: Master's student in Applied Mathematics & AI
[LinkedIn](https://www.linkedin.com/in/fatima-ezzahra-boukhorssa-09aa9a259) · [GitHub](https://github.com/fatimaezzahra-AI)