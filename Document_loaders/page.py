from langchain_community.document_loaders import WebBaseLoader

url = "https://www.producthunt.com/"

data = WebBaseLoader(url)

docs = data.load()

print(docs[0].page_content)