import os, re
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json 
from .utils import planet_to_text
from .llm import GroqLLM
from app.query_interpreter import QueryInterpreter

class AstroRAG:
    def __init__(self, index_path="index/astro.index", meta_path="index/metadata.pkl"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.llm = GroqLLM()
        self.index = None
        self.metadata = None


        self.planet_data = self.load_planet_data()

        if self.planet_data is None:
            raise ValueError("Must provide planet_data to QueryInterpreter")
        
        self.query_interpreter = QueryInterpreter(self.planet_data)

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)

            index_dim = self.index.d
            model_dim = self.model.get_sentence_embedding_dimension()

            if index_dim != model_dim:
                print(f"FAISS index dim={index_dim}, model dim={model_dim}. Rebuild index needed.")
                self.build_index("data")
            else:
                print("Loaded existing FAISS index.")

        else:
            print("No index found. Build index first using .build_index()")


    def load_planet_data(self, path="/workspaces/rag-astronomy-chatbot-/data/exoplanets/exoplanets.json"):
            with open(path, "r") as f:
                data = json.load(f)
            # data should be a list of planet dicts
            return data
    

    def load_texts(self, folder="data"):
        docs = []
        ids = []

        for root, _, files in os.walk(folder):
            for f in files:
                path = os.path.join(root, f)
                if f.endswith(".txt"):
                    with open(path, "r", encoding="utf-8") as file:
                        content = file.read()
                    chunks = self.chunk_text(content)
                    for chunk in chunks:
                        docs.append(chunk)
                        ids.append(path)

                elif f.endswith(".json"):
                    with open(path, "r", encoding="utf-8") as jf:
                        data = json.load(jf)

                    if isinstance(data, list):
                        for item in data:
                            text = planet_to_text(item)
                            if text.strip():
                                chunks = self.chunk_text(text)
                                for chunk in chunks:
                                    docs.append(chunk)
                                    ids.append(path)

                    elif isinstance(data, dict):
                        text = planet_to_text(data)
                        if text.strip():
                            chunks = self.chunk_text(text)
                            for chunk in chunks:
                                docs.append(chunk)
                                ids.append(path)

        # Debug: show a snippet of what’s loaded for indexing
        if docs:
            print(f"Loaded sample text for indexing (first 500 chars):\n{docs[0][:500]}...\n")

        return docs, ids


    def chunk_text(self, text, max_len=350):
        sentences = text.split(".")
        chunks = []
        current = ""

        for s in sentences:
            if len(current) + len(s) < max_len:
                current += s.strip() + ". "
            else:
                chunks.append(current.strip())
                current = s.strip() + ". "

        if current:
            chunks.append(current.strip())

        return chunks

    def build_index(self, folder="data"):
        docs, ids = self.load_texts(folder)
        embeddings = self.model.encode(docs, show_progress_bar=True)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))

        os.makedirs("index", exist_ok=True)
        faiss.write_index(index, self.index_path)

        with open(self.meta_path, "wb") as f:
            pickle.dump({"docs": docs, "ids": ids}, f)

        print("🚀 Index built successfully!")
        self.index = index
        self.metadata = {"docs": docs, "ids": ids}



    def query(self, user_query):
        parsed = self.query_interpreter.parse_intent(user_query)

        # Structured query route (no embeddings)
        if "order_by" in parsed or any(k in parsed for k in ["radius", "mass", "orbital_period", "distance"]):
            print(" Structured DB mode (no RAG fallback)")
            df = self.planet_db.df.copy()

            # Apply constraints if present
            for key, val in parsed.items():
                if key in ["radius", "mass", "orbital_period", "distance"] and isinstance(val, tuple):
                    df = df[(df[key] >= val[0]) & (df[key] <= val[1])]

            # Sort
            if "order_by" in parsed:
                df = df.sort_values(parsed["order_by"], ascending=not parsed.get("desc", True))

            # Limit
            limit = parsed.get("limit", 10)
            df = df.head(limit)

            return df.to_dict(orient="records")  # structured data, not summarization

        # Default → embeddings
        print("📎 Fallback to RAG search mode")
        results = self.search(user_query)
        return self.llm.summarize(results)


    def summarize_chunks(self, chunks, q):
        combined = "\n\n".join([c["chunk"] for c in chunks])
        print(combined)
        prompt = f"""
        You are a strict astronomy RAG assistant.

        User question: {q}

        Context:
        {combined}

        Rules:
        - If the question uses terms like "largest", "longest", "highest", "biggest", "maximum":
            → extract ALL candidates mentioned in context
            → compare them by the correct attribute (radius for largest planet)
            → output ONLY the top ranking planet
        - Do not ignore planets if context lists multiple.
        - No summaries unless specifically asked.
        - Answer in one sentence.
        """

        return self.llm.generate(prompt, max_tokens=400)

    def answer_directly(self, chunks, q):
        combined_text = "\n\n".join([c["chunk"] for c in chunks])
        print(combined_text)
        prompt = f"""
        You are an astronomy expert.
        Answer strictly based on the given context.

        Question: {q}

        Context:
        {combined_text}

        Answer in one clear sentence.
        Do not summarize multiple planets.
        Do not add explanation.
        Give only the fact asked.

        Rules:
        - If the question uses terms like "largest", "longest", "highest", "biggest", "maximum":
            → extract ALL candidates mentioned in context
            → compare them by the correct attribute (radius for largest planet)
            → output ONLY the top ranking planet
        - Do not ignore planets if context lists multiple.
        - No summaries unless specifically asked.
        - Answer in one sentence.
        """
        return self.llm.generate(prompt, max_tokens=500)
