import os, re
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json 
from .utils import planet_to_text
from .llm import GroqLLM
from app.query_interpreter import QueryInterpreter
from app.planet_db import PlanetDB


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
        self.planet_db = PlanetDB(self.planet_data)
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

    def search(self, query, top_k=10):
        if self.index is None:
            raise ValueError("FAISS index not loaded. Build index first using .build_index()")

        if self.metadata is None or "docs" not in self.metadata:
            raise ValueError("Metadata missing or corrupted. Rebuild index.")

        # Encode the query
        q_vec = self.model.encode([query])

        # Search FAISS
        distances, indices = self.index.search(np.array(q_vec), top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.metadata["docs"]):
                results.append({
                    "chunk": self.metadata["docs"][idx],
                    "source": self.metadata["ids"][idx]
                })

        return results

    def rank(self, intent):
        df = self.df.copy()

        attr = intent.get("attribute", "radius")

        # apply filters
        if "year_filter" in intent:
            df = df[df["discovery_year"] >= intent["year_filter"]]

        if "mass_gt" in intent:
            df = df[df["mass"] > intent["mass_gt"]]

        if "mass_lt" in intent:
            df = df[df["mass"] < intent["mass_lt"]]

        # sort
        if intent.get("agg") == "max":
            df = df.sort_values(attr, ascending=False)
        elif intent.get("agg") == "min":
            df = df.sort_values(attr, ascending=True)

        # limit
        limit = intent.get("limit", 10)
        return df.head(limit).to_dict(orient="records")

    def query(self, user_query):
        parsed = self.query_interpreter.parse_intent(user_query)
        print("Parsed intent:", parsed)

        if parsed.get("agg") == "max" and parsed.get("plural"):
            planets = self.planet_db.rank(parsed)  # your pandas output list of dicts
            # convert ranked planet dicts to chunks expected by summarize_chunks
            chunks = [{"chunk": f"{p['planet_name']} — Radius: {p.get(parsed['attribute'] + '_earth_radii', 'N/A')} Earth radii"} for p in planets[:parsed["limit"]]]
            return self.summarize_chunks(chunks, user_query)

        if parsed.get("agg") == "max":
            return self.query_interpreter.answer(user_query)

        if parsed.get("agg") in ["min", "avg", "count"] or parsed.get("attribute"):
            return self.query_interpreter.answer(user_query)

        chunks = self.search(user_query)
        return self.summarize_chunks(chunks, user_query)





    def summarize_chunks(self, chunks, q):
        combined = "\n\n".join([c["chunk"] for c in chunks])
        print(combined)
        prompt = f"""
        You are a strict astronomy RAG assistant.

        User question: {q}

        Context:
        {combined}

        Rules:
            - If the user requests "top", "largest", "biggest", "max", etc:
                → return ALL matching planets up to limit (default 10 unless specified)
            - Format as a ranked list, not a single summary.
            - Do NOT collapse into one planet.
            - Do NOT summarize into prose. Show list format.

            Format:
            1. Planet Name — Radius X
            2. Planet Name — Radius Y
        """

        return self.llm.generate(prompt, max_tokens=700)

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
            - If the user requests "top", "largest", "biggest", "max", etc:
                → return ALL matching planets up to limit (default 10 unless specified)
            - Format as a ranked list, not a single summary.
            - Do NOT collapse into one planet.
            - Do NOT summarize into prose. Show list format.

            Format:
            1. Planet Name — Radius X
            2. Planet Name — Radius Y
        """
        return self.llm.generate(prompt, max_tokens=700)
