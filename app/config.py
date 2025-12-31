import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# configuração do LLM Local
llm = ChatOllama(model="llama3", temperature=0)

# configuração de Embeddings Local
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def get_retriever():
    vectordb = Chroma(
        persist_directory="./chroma_wiki", 
        embedding_function=embeddings,
        collection_name="wikipedia_pt_rag"
    )
    return vectordb.as_retriever(search_kwargs={"k": 4})