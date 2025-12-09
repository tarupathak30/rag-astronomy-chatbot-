import os
from groq import Groq

class GroqLLM:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment!")
        self.client = Groq(api_key=api_key)

    def generate(self, prompt: str, max_tokens: int = 900) -> str:
        resp = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.9,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
        return resp.choices[0].message.content
