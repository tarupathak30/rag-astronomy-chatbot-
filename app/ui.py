import gradio as gr
from .rag_pipeline import RAGPipeline  # relative import optional if running as script
from .llm import GroqLLM

# Init RAG + LLM
rag = RAGPipeline("data")
llm = GroqLLM()

def chat(query: str):
    # Retrieve relevant chunks
    retrieved = rag.retrieve(query, top_k=5)
    context = "\n\n".join([r["text"] for r in retrieved])

    # Prepare prompt
    prompt = f"""
You are answering a question about exoplanets.
Use ONLY the following context:

{context}

Question: {query}
Answer in one short paragraph.
"""
    # Call LLM
    return llm.generate(prompt)


iface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask about exoplanets..."),
    outputs=gr.Markdown(),
    title="AstroRAG Chatbot",
    description="Query exoplanets using RAG + Groq LLM"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=4482)
