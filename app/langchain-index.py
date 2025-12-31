import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  # Mudança aqui
from langchain_community.vectorstores import Chroma

load_dotenv()

# 2.2 Coleta de páginas 
topics = ["Lei Geral de Proteção de Dados", "Transformers (NLP)", "LangChain", "Wikipedia"]
docs = []
for t in topics:
    loader = WikipediaLoader(query=t, lang="pt", load_max_docs=1)
    docs.extend(loader.load())

# 2.3 Split em chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=120, 
    add_start_index=True
)
splits = splitter.split_documents(docs)

# 2.4 Embeddings + VectorStore
# precisei adaptar para o ollamaembeddings

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="wikipedia_pt_rag",
    persist_directory="./chroma_wiki"
)

# 2.5 Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

print(f"Sucesso! {len(splits)} chunks armazenados no ChromaDB local.")