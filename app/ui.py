import gradio as gr
from .rag_pipeline import RAGPipeline  # relative import optional if running as script
from .llm import GroqLLM

# Init RAG + LLM
rag = RAGPipeline("data")
llm = GroqLLM()


def chat(query: str):
    # Try structured numeric comparison first
    structured = rag.structured(query)

    if structured is not None and "planet" in structured and structured["planet"]:
        p = structured["planet"]
        name = p.get("planet_name", "Unknown Planet")
        val = structured.get("value", "N/A")
        attr = structured.get("attribute", "Unknown Attribute")

        return f"✅ Structured Answer → **{name}** has **{attr.replace('_',' ')}** = **{val}**"

    # Fallback: retrieve relevant chunks from RAG
    retrieved = rag.retrieve(query, top_k=5)
    if not retrieved:
        return "⚠️ Sorry, no relevant information found in the context."

    context = "\n\n".join([r["text"] for r in retrieved])

    # Prepare prompt for LLM
    prompt = f"""
You are answering a question about exoplanets.
Use ONLY the following context:

{context}

Question: {query}
Answer in one short paragraph.
"""

    # Call LLM
    try:
        answer = llm.generate(prompt)
    except Exception as e:
        answer = f"⚠️ LLM call failed: {str(e)}"

    return answer



iface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask about exoplanets..."),
    outputs=gr.Markdown(),
    title="AstroRAG Chatbot",
    description="Query exoplanets using RAG + Groq LLM"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=4482)
