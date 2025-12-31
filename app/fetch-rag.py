# conteudo relacionado à aula 2
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

from config import llm, get_retriever 

retriever = get_retriever()

qa_prompt = ChatPromptTemplate.from_template(
    "Você é um assistente factual. Use EXCLUSIVAMENTE o contexto para responder.\n"
    "Se não houver informação suficiente, diga isso explicitamente.\n\n"
    "Pergunta: {pergunta}\n\n"
    "Contexto:\n{contexto}\n\n"
    "Responda de forma concisa e cite as fontes no final no formato [fonte: título]."
)

def format_docs(docs):
    blocos = []
    for d in docs:
        titulo = d.metadata.get("title", "Wikipedia")
        blocos.append(f"[{titulo}] {d.page_content}")
    return "\n\n---\n\n".join(blocos)
        
def answer_linear(pergunta: str):
    #atualizado para retriever.invoke
    docs = retriever.invoke(pergunta) 
    contexto = format_docs(docs)
    msg = qa_prompt.format_messages(pergunta=pergunta, contexto=contexto)
    out = llm.invoke(msg)
    return out.content

#teste rapido
print(answer_linear("Quais são os princípios da LGPD?"))

# resposta gerada: Os princípios da LGPD não são explicitamente mencionados no contexto fornecido. No entanto, a lei estabelece normas para a proteção e tratamento de dados pessoais em território nacional, bem como a proteção de direitos fundamentais, particularmente a privacidade, a honra e a intimidade.
# [Fonte: Lei Geral de Proteção de Dados Pessoais]