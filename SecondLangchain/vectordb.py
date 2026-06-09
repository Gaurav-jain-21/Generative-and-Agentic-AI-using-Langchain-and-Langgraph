from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()
loader= PyPDFLoader("dl-curriculum.pdf")
docs= loader.load()

vector_store= Chroma(
    embedding_function=HuggingFaceEndpointEmbeddings(
        repo_id="BAAI/bge-large-en-v1.5",
        task="feature-extraction",
    ),
    collection_name="deep_learning_docs",
    persist_directory="./"
)

vector_store.add_documents(docs)

