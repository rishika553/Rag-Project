
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_classic.retrievers import MultiQueryRetriever
from dotenv import load_dotenv
import os

load_dotenv()

# Documents
docs = [
    Document(
        page_content="gradient descent is an optimization algorithm used to minimize the loss function in machine learning models. It works by iteratively adjusting the model's parameters in the direction of the negative gradient of the loss function with respect to those parameters. This process continues until the algorithm converges to a minimum value of the loss function, ideally leading to better model performance on unseen data."
    ),

    Document(
        page_content="backpropagation is a supervised learning algorithm used for training artificial neural networks. It calculates the gradient of the loss function with respect to each weight by applying the chain rule, allowing the network to update its weights in order to minimize the loss. Backpropagation is essential for enabling deep learning models to learn complex patterns in data."
    ),

    Document(
        page_content="support vector machines (SVMs) are a set of supervised learning methods used for classification, regression, and outlier detection. The basic idea of SVM is to find the hyperplane that best separates the data into different classes. SVMs can handle both linear and non-linear classification tasks by using kernel functions to transform the input space."
    ),

    Document(
        page_content="random forests are an ensemble learning method for classification, regression, and other tasks that operate by constructing a multitude of decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random forests correct for decision trees' habit of overfitting to their training set."
    ),
]

# Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

# Normal similarity retriever
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

print("\nSimilarity Search Results:")

similarity_docs = similarity_retriever.invoke(
    "what is gradient descent?"
)

for doc in similarity_docs:
    print(doc.page_content)


# OpenRouter LLM
llm = ChatOpenRouter(
    model="openai/gpt-4o-mini",
    temperature=0
)

# MultiQuery Retriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    ),
    llm=llm
)

print("\nMultiQuery Search Results:")

multiquery_docs = multiquery_retriever.invoke(
    "what is gradient descent?"
)

for doc in multiquery_docs:
    print(doc.page_content)
