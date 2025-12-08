import os
from groq import Groq

class GroqLLM:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))


    def generate(self, prompt, max_tokens=900):
        resp = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
        return resp.choices[0].message.content

