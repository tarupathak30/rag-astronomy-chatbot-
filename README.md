🌌 Astronomy RAG Chatbot

Grounded Question-Answering over Exoplanet Data using Retrieval-Augmented Generation

General-purpose LLMs hallucinate when asked precise scientific questions.
This project tackles that failure mode by grounding responses in a curated exoplanet knowledge base using a Retrieval-Augmented Generation (RAG) pipeline.

🚀 Motivation

Astronomy queries demand precision, not plausibility.

Instead of relying on an LLM’s parametric memory, this system:

retrieves factual context from real exoplanet datasets

injects that context into the generation step

produces answers constrained by retrieved evidence

The result: answers that are explainable, verifiable, and domain-grounded.

🧠 Architecture Overview
User Query
   ↓
Dense Query Embedding
   ↓
FAISS Vector Similarity Search
   ↓
Top-K Context Retrieval
   ↓
LLM Generation (Context-Injected)
   ↓
Grounded Final Answer


This clean separation between retrieval and generation was intentional to reduce hallucinations and improve traceability.

🧩 Core Capabilities

Semantic retrieval using sentence-level embeddings

Domain-specific knowledge base built from exoplanet datasets

Hallucination reduction via strict context injection

Low-latency inference with a precomputed vector index

Interactive Gradio UI for exploratory querying

🗂️ Dataset

Format: JSON

Domain: Exoplanets

Attributes include:

Planet name

Physical properties (mass, radius, orbital parameters)

Discovery metadata

Location: data/exoplanets/

🏗️ Repository Structure
.
├── app/                    # Application & query pipeline
├── data/exoplanets/        # Raw domain data
├── index/                  # FAISS vector index
├── exoplanets_GITHUB.ipynb # Data prep & indexing workflow
├── requirements.txt
└── README.md

🛠️ Tech Stack

Python

Sentence Transformers (embeddings)

FAISS (vector search)

LLM (API-based)

Gradio (UI)

📌 Indexing Design

Data is chunked into semantically meaningful units

Embeddings are generated once and persisted locally

Index rebuilds are triggered when:

dataset content changes

new files are added

Incremental indexing is intentionally deferred to prioritize correctness and reproducibility in this MVP.

▶️ Running the Application
pip install -r requirements.txt
python app/ui.py


Launches a local Gradio interface for interactive querying.

🧪 Example Queries

Which exoplanet has the largest known radius?

Tell me about WASP-17 b

Compare hot Jupiters discovered by Kepler

All responses are generated strictly from retrieved dataset context.

⚠️ Known Limitations

Coverage limited to exoplanet data

No live astronomy API integration

Evaluation is currently qualitative

These trade-offs were chosen to keep the system focused and interpretable.

🔮 Planned Extensions

Expand knowledge base to stars, galaxies, and missions

Hybrid retrieval (BM25 + dense)

Incremental index updates

Automated evaluation benchmarks

Public deployment

📎 Why This Project Exists

This project demonstrates:

System-level understanding of RAG pipelines

Explicit control over hallucination failure modes

Separation of concerns between retrieval and generation

A shift from “LLM demo” to grounded AI system design

👤 Author
Taru Pathak
