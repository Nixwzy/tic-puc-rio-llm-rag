# Passo a passo
#     Escolha dois temas da Wikipedia (por exemplo: “Inteligência Artificial” e “História da Internet”).
#     Use o WikipediaLoader do LangChain para carregar os artigos correspondentes e criar os documentos.
#     Divida os textos em chunks para tornar as buscas mais específicas.
#     Instale a biblioteca de embeddings sentence-transformers com o comando pip install sentence-transformers
#     Gere embeddings com o modelo HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1").
#     Crie um banco vetorial local com ChromaDB e adicione os embeddings.
#     Teste uma consulta simples ao banco para verificar os resultados de similaridade.
#     Monte o grafo no LangGraph, incluindo:
#         Nó de busca no banco vetorial;
#         Nó de inclusão de contexto no prompt;
#         Nó de geração de resposta com LLM.
#     Execute uma pergunta de teste (por exemplo: “Quando surgiu o termo Inteligência Artificial?”) e observe a resposta gerada com citações.

import os
from typing import List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import llm, embeddings, get_retriever

def minirag():
    print("Iniciando coleta de dados...")
    topics = ["Inteligência Artificial", "História da Internet"]
    docs = []
    for t in topics:
        loader = WikipediaLoader(query=t, lang="pt", load_max_docs=1)
        docs.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    splits = splitter.split_documents(docs)
    
    from langchain_community.vectorstores import Chroma
    # separando em chunks
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_wiki",
        collection_name="wikipedia_pt_rag"
    )

class RAGState(TypedDict):
    pergunta: str
    docs: List
    contexto: str
    resposta: str

def node_retrieve(state: RAGState) -> RAGState:
    retriever = get_retriever()
    documents = retriever.invoke(state["pergunta"])
    return {"docs": documents}

def node_augment(state: RAGState) -> RAGState:
    conteudo = "\n\n".join([f"[Fonte: {d.metadata.get('title')}] {d.page_content}" for d in state["docs"]])
    return {"contexto": conteudo}

def node_generate(state: RAGState) -> RAGState:
    prompt = ChatPromptTemplate.from_template(
        "Você é um assistente factual. Use o contexto para responder.\n\nContexto:\n{contexto}\n\nPergunta: {pergunta}"
    )
    msg = prompt.format_messages(pergunta=state["pergunta"], contexto=state["contexto"])
    response = llm.invoke(msg)
    return {"resposta": response.content}

workflow = StateGraph(RAGState)
workflow.add_node("Retrieve", node_retrieve)
workflow.add_node("Augment", node_augment)
workflow.add_node("Generate", node_generate)

workflow.set_entry_point("Retrieve")
workflow.add_edge("Retrieve", "Augment")
workflow.add_edge("Augment", "Generate")
workflow.add_edge("Generate", END)

app = workflow.compile()

if __name__ == "__main__":
    minirag()
    
    pergunta_teste = "Quando surgiu o termo Inteligência Artificial?"
    print(f"\nProcessando pergunta: {pergunta_teste}")
    
    resultado = app.invoke({"pergunta": pergunta_teste})
    
    print("\nResposta:")
    print(resultado["resposta"])
    
# Resposta:
# De acordo com o contexto, a inteligência artificial (IA) genericamente é a inteligência, o raciocínio e o aprendizado exibida por máquinas semelhante ao raciocino humano; busca desenvolver máquinas autônomas ou sistemas especialistas capazes de simular o pensamento humano e realizar várias tarefas complexas de forma independente.