
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("Document_loaders/ml.pdf")


docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)


chunks = splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding= embedding_model,
    persist_directory = "chroma-db2"


)

