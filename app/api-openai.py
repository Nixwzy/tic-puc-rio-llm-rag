import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLAMA_BASE_URL"),
    api_key=os.getenv("LLAMA_API_KEY")
)

print("Cliente OpenAI iniciado com sucesso.") # msg ok
completion = client.chat.completions.create(
    model="llama3",
    messages=[
        {"role": "system", "content": "Você é um assistente útil."},
        {
            "role": "user",
            "content": "Olá, tudo bem?"
        }
    ],
    temperature=1.0,
    stream=False # está bugando
)

print(completion.choices[0].message.content)