import arxiv
from langchain_community.retrievers import ArxivRetriever

client = arxiv.Client(
    page_size=5,
    delay_seconds=3,
    num_retries=3
)

retriever = ArxivRetriever(
    load_max_docs=2,
    load_all_available_meta=True,
    client=client
)

docs = retriever.invoke("large language models")

for i, doc in enumerate(docs):
    print(f"\nResult {i + 1}")
    print("Title:", doc.metadata.get("title"))
    print("Summary:", doc.page_content[:500])