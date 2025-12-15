🌌 Astronomy RAG Chatbot

Domain-Specific Retrieval-Augmented Generation System

A lightweight but production-minded Retrieval-Augmented Generation (RAG) chatbot designed to answer factual astronomy questions using curated exoplanet datasets, rather than relying on hallucination-prone generic LLM knowledge.

🚀 Overview

Large Language Models are fluent but unreliable when queried on specialized scientific domains.
This project addresses that gap by building a RAG pipeline that grounds responses in real exoplanet data.

The chatbot retrieves relevant information from a vector index built over structured astronomy datasets and uses it to generate accurate, context-aware answers.

🧠 System Architecture

High-level flow:

User Query
   ↓
Query Embedding
   ↓
Vector Similarity Search (FAISS)
   ↓
Relevant Context Retrieval
   ↓
LLM Answer Generation
   ↓
Final Grounded Response

🧩 Key Features

🔍 Semantic Retrieval using dense embeddings

📚 Domain-specific knowledge base (exoplanet data)

🧠 Context-grounded generation to reduce hallucinations

⚡ Fast inference with pre-built vector index

🖥️ Interactive UI using Gradio

🗂️ Dataset

Source: Curated exoplanet datasets (JSON format)

Content includes:

Planet name

Physical characteristics (mass, radius, orbital parameters)

Discovery metadata

Stored under: data/exoplanets/

🏗️ Project Structure
.
├── app/                    # Chatbot application logic
├── data/exoplanets/        # Raw exoplanet datasets
├── index/                  # FAISS vector index
├── exoplanets_GITHUB.ipynb # Data exploration & indexing notebook
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

🛠️ Tech Stack

Python

Sentence Transformers (Embeddings)

FAISS (Vector similarity search)

LLM (API-based) for response generation

Gradio for UI

📌 Indexing Strategy

Data is chunked into semantically meaningful text units

Embeddings are generated once and stored locally

Index is rebuilt when:

Dataset changes

New files are added

(Current version uses full index rebuild for reliability; incremental updates are a planned improvement.)

▶️ Running the Project
pip install -r requirements.txt
python app/ui.py


The Gradio interface will launch locally in your browser.

🧪 Example Queries

“Which exoplanet has the largest radius?”

“Tell me about WASP-17 b”

“Compare hot Jupiters discovered by Kepler”

Responses are generated only from retrieved dataset context.

⚠️ Limitations

Dataset coverage is currently limited to exoplanets

No real-time astronomy API integration yet

Evaluation is qualitative (manual inspection)

These are intentional trade-offs for a focused MVP.

🔮 Future Work

Expand dataset to stars, galaxies, and missions

Add hybrid retrieval (BM25 + dense)

Implement incremental index updates

Introduce automated evaluation metrics

Deploy as a public web service

📎 Why This Project Matters

This project demonstrates:

Practical understanding of RAG systems

Clear separation of retrieval vs generation

Focus on grounded, reliable AI outputs

Ability to move beyond toy LLM demos into system design

👤 Author tarupathak30

Taru Pathak
GitHub: tarupathak30
