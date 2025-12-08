import gradio as gr
from .rag_pipeline import AstroRAG
rag = AstroRAG()
rag.build_index() 


def chat(query):
    answer = rag.query(query)  # returns LLM answer now
    return f"{answer}"


iface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Ask anything about the universe"),
    outputs=gr.Markdown(),
    title="AstroRAG Chatbot",
    description="Ask about exoplanets, black holes, dark matter, movies, phenomena, etc."
)


if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=4482)

