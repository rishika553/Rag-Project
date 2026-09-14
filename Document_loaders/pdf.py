from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=100,
    chunk_overlap=1
)

data = PyPDFLoader("erp.pdf")

docs = data.load()

chunks = splitter.split_documents(docs)

print(len(chunks))
print(chunks[0].page_content)