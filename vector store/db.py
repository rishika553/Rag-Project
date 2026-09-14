from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
import os

load_dotenv()

docs = [
    Document(page_content="neural networks are used in deep learning"),
    Document(page_content="python is used for machine learning and data science."),
    Document(page_content="pandas is used for data analysis and manipulation in python."),
]

embedding_model = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
 
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

result = vectorstore.similarity_search("what is used for machine learning", k=2)
for r in result :
    print(r.page_content) 
    print(r.metadata)
retriver = vectorstore.as_retriever()

docs = retriver.invoke("explain deep learning")
for d in docs:
    print(d.page_content)