from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()






template = ChatPromptTemplate.from_messages(
    [("system", "you are helpful ai assistent that summarize the text"),
     ("human", "{data}")]
)


docs = data.load()

print(docs[0].page_content)

model = ChatOpenAI(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
prompt = template.format_messages(data=docs[0].page_content)

print("Model created successfully")

result = model.invoke(prompt)

print(result.content)