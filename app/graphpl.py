# conteudo relacionado à aula 2
import os
from typing import List, TypedDict
from dotenv import load_dotenv

# Imports do Grafo e LangChain
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

from config import llm, get_retriever 

retriever = get_retriever()

qa_prompt = ChatPromptTemplate.from_template(
    "Você é um assistente factual. Use EXCLUSIVAMENTE o contexto para responder.\n"
    "Pergunta: {pergunta}\n\nContexto:\n{contexto}"
)

def format_docs(docs):
    return "\n\n".join([f"[{d.metadata.get('title', 'Wiki')}] {d.page_content}" for d in docs])

class RAGState(TypedDict):
    pergunta: str
    docs: List
    contexto: str
    resposta: str

def node_retrieve(state: RAGState) -> RAGState:
    #get_relevant_documents não funciona
    docs = retriever.invoke(state["pergunta"])
    return {"docs": docs, "contexto": format_docs(docs)}

def node_augment(state: RAGState) -> RAGState:
    return {"contexto": state["contexto"]}

def node_generate(state: RAGState) -> RAGState:
    msg = qa_prompt.format_messages(pergunta=state["pergunta"], contexto=state["contexto"])
    out = llm.invoke(msg).content
    return {"resposta": out}

# montagem do grafo
graph = StateGraph(RAGState)
graph.add_node("Retrieve", node_retrieve)
graph.add_node("AugmentPrompt", node_augment)
graph.add_node("Generate", node_generate)

graph.set_entry_point("Retrieve")
graph.add_edge("Retrieve", "AugmentPrompt")
graph.add_edge("AugmentPrompt", "Generate")
graph.add_edge("Generate", END)

app = graph.compile()

# exemplo de uso
result = app.invoke({"pergunta": "O que é LangChain?"})
print("resposta", result["resposta"])
print("fonte:", result["docs"][0].metadata["source"])