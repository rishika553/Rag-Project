from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator = "",
    chunk_size=1000,
    chunk_overlap=1
)


data = TextLoader("notes.txt")

docs = data.load()

chunks = splitter.split_documents(docs)

print(len(chunks))
# print(chunks[0].page_content) 

for i in chunks:
    print(i.page_content)
    print()
    print()
    print()
