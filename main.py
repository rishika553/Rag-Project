from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import Chroma
from langchain_openrouter import ChatOpenRouter
import os

load_dotenv()

#embedding model 

embedding_model = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

#chroma vectorstore

vectorstore = Chroma(
    persist_directory="chroma-db2",
    embedding_function=embedding_model
)
print("Chroma document count:", vectorstore._collection.count())

#retriever

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k": 4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)
llm = ChatOpenRouter(
    model = "cohere/north-mini-code:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

#prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
    """You are a helpful AI Assistant.
    use only the provided context to answer the question.
    if the answer is not present in the context.
    say:"I counld not find the answer in the document."

    """),
    (
        "human",
    """context:

    {context}
    question:
    {question}
    """
    )
    ]
)
print("Rag System created")

print("press 0 to exit")

while True:
    query = input("YOU:")

    if query == "0":
        break

    docs = retriever.invoke(query)


    

   
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print(f"\n AI : {response.content}")
