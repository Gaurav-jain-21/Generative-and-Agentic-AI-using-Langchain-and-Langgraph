from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

loader=PyPDFLoader("dl-curriculum.pdf")
docs= loader.load()

embeddings_model=HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)

vectorstore= Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model,
    collection_name="my_collection",
    persist_directory="./vectorstore"
)

retriever= vectorstore.as_retriever(search_kwargs={"k":2})
query="What is Deep Learning"
result= retriever.invoke(query)
for i , docs in enumerate(result):
    print(f"\n-- Result {i+1} --\n")
    print(docs.page_content)