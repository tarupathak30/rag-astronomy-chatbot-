
from .logic.exoplanets_comparator import Comparator
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .utils import load_json_folder, planet_text, chunk_text

class RAGPipeline:
    def __init__(self, folder="/workspaces/rag-astronomy-chatbot-/data/exoplanets", embedding_model="all-MiniLM-L6-v2"):
        # Load planet data
        self.data = load_json_folder(folder)
        if not self.data:
            raise ValueError(f"No JSON data found in folder: {folder}")

        # Convert planets to text and chunk
        self.corpus_texts = []
        self.corpus_metadata = []
        for obj in self.data:
            text = planet_text(obj)
            chunks = chunk_text(text, max_len=300)
            for c in chunks:
                self.corpus_texts.append(c)
                self.corpus_metadata.append(obj)

        self.cmp = Comparator(self.data)

        # Initialize embeddings
        self.model = SentenceTransformer(embedding_model)
        self.embeddings = np.array(self.model.encode(self.corpus_texts, show_progress_bar=True))

        # Build FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)
        print(f"✅ Index built with {len(self.corpus_texts)} chunks, dim={dim}")

    def structured(self, query: str):
        """Run numeric/structured lookup."""
        res = self.cmp.lookup(query)
        if res: 
            print("bro, there is a response for you ")
            print(res)
            return res
    

    def retrieve(self, query: str, top_k=5):
        if not query:
            return []

        # 1. Force-inject any planets explicitly named
        forced = self.force_load_planets(query)

        # 2. Semantic retrieval (as before)
        q_emb = np.array(self.model.encode([query]))
        D, I = self.index.search(q_emb, top_k)

        semantic = []
        for idx in I[0]:
            semantic.append({
                "text": self.corpus_texts[idx],
                "metadata": self.corpus_metadata[idx]
            })

        # 3. Merge → forced first (ensures planet availability)
        return forced + semantic


    def force_load_planets(self, query: str):
        """Hard-match planet names and inject their full JSON rows."""
        found = []

        for obj in self.data:
            name = obj.get("planet_name", "")
            if name.lower() in query.lower():
                found.append({
                    "text": planet_text(obj),   # full cleaned text
                    "metadata": obj
                })

        return found
