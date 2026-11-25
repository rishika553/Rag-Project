import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------
# Load environment variables
# --------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------
# Custom CSS — chat bubble look & feel
# --------------------------------

st.markdown(
    """
    <style>
        /* Overall page */
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }

        /* Hide default streamlit chrome */
        #MainMenu, footer {visibility: hidden;}

        /* Header */
        .chat-header {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 18px 22px;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border-radius: 16px;
            margin-bottom: 18px;
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.25);
        }
        .chat-header .icon {
            font-size: 34px;
        }
        .chat-header h1 {
            color: white;
            font-size: 22px;
            margin: 0;
            font-weight: 700;
        }
        .chat-header p {
            color: rgba(255,255,255,0.85);
            margin: 2px 0 0 0;
            font-size: 13px;
        }

        /* Status pill */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(34,197,94,0.15);
            color: #4ade80;
            border: 1px solid rgba(74,222,128,0.35);
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 16px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4ade80;
            box-shadow: 0 0 8px #4ade80;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] .stMarkdown, 
        section[data-testid="stSidebar"] label {
            color: #e5e7eb;
        }

        /* Metric cards */
        .metric-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 8px;
        }
        .metric-card .label {
            color: #9ca3af;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-card .value {
            color: #f9fafb;
            font-size: 20px;
            font-weight: 700;
        }

        /* Chat bubbles */
        [data-testid="stChatMessage"] {
            border-radius: 16px;
            padding: 4px 6px;
            margin-bottom: 10px;
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #9ca3af;
        }
        .empty-state .big-icon {
            font-size: 52px;
            margin-bottom: 12px;
        }
        .empty-state h3 {
            color: #e5e7eb;
            margin-bottom: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------
# Check API key
# --------------------------------

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY is missing from your .env file.")
    st.stop()


# --------------------------------
# Header
# --------------------------------

st.markdown(
    """
    <div class="chat-header">
        <div class="icon">📚</div>
        <div>
            <h1>PDF Chat Assistant</h1>
            <p>Ask questions and get instant answers straight from your document</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------
# Embedding model & LLM (cached)
# --------------------------------

@st.cache_resource
def get_embedding_model():
    return OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


@st.cache_resource
def get_llm():
    return ChatOpenRouter(
       model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


embedding_model = get_embedding_model()
llm = get_llm()


# --------------------------------
# Prompt
# --------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the information provided in the context to answer the user's question.

Do not use outside knowledge.

If the answer is not present in the context, say:
"I could not find the answer in the document."

Give a clear and concise answer.
""",
        ),
        (
            "human",
            """Context:

{context}

Question:

{question}
""",
        ),
    ]
)


# --------------------------------
# Session state
# --------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {"pages": 0, "chunks": 0}


# --------------------------------
# Sidebar
# --------------------------------

with st.sidebar:

    st.markdown("### 📎 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:

        if st.session_state.file_name != uploaded_file.name:

            with st.spinner("Reading and indexing your PDF..."):

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_pdf_path = temp_file.name

                try:
                    loader = PyPDFLoader(temp_pdf_path)
                    docs = loader.load()

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    )
                    chunks = splitter.split_documents(docs)

                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embedding_model,
                        collection_name="pdf_rag",
                    )

                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
                    )

                    st.session_state.vectorstore = vectorstore
                    st.session_state.retriever = retriever
                    st.session_state.file_name = uploaded_file.name
                    st.session_state.doc_stats = {
                        "pages": len(docs),
                        "chunks": len(chunks),
                    }
                    # Fresh chat for the new document
                    st.session_state.messages = [
                        {
                            "role": "assistant",
                            "content": f"I've finished reading **{uploaded_file.name}**. "
                            f"Ask me anything about it!",
                        }
                    ]

                    st.success("Document ready!")

                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

                finally:
                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)

    if st.session_state.file_name:
        st.markdown("---")
        st.markdown("### 📊 Document Info")

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">File</div>
                <div class="value" style="font-size:14px;">{st.session_state.file_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">Pages</div>
                    <div class="value">{st.session_state.doc_stats['pages']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">Chunks</div>
                    <div class="value">{st.session_state.doc_stats['chunks']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat cleared. What would you like to know about the document?",
                }
            ]
            st.rerun()

        if st.button("♻️ Remove document", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.retriever = None
            st.session_state.file_name = None
            st.session_state.messages = []
            st.session_state.doc_stats = {"pages": 0, "chunks": 0}
            st.rerun()


# --------------------------------
# Main chat area
# --------------------------------

if st.session_state.retriever is None:

    st.markdown(
        """
        <div class="empty-state">
            <div class="big-icon">🗂️</div>
            <h3>No document loaded yet</h3>
            <p>Upload a PDF from the sidebar to start chatting with it in real time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class="status-pill">
            <div class="status-dot"></div>
            Ready to chat
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render chat history
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "📚"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Chat input pinned at the bottom
    user_question = st.chat_input("Ask a question about your PDF...")

    if user_question:

        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_question)

        # Generate and show assistant reply
        with st.chat_message("assistant", avatar="📚"):
            placeholder = st.empty()
            placeholder.markdown("▌ Thinking...")

            try:
                docs = st.session_state.retriever.invoke(user_question)
                context = "\n\n".join(doc.page_content for doc in docs)

                final_prompt = prompt.invoke(
                    {"context": context, "question": user_question}
                )

                response = llm.invoke(final_prompt)
                answer = response.content

                # Simple streaming effect for a livelier feel
                displayed = ""
                for word in answer.split(" "):
                    displayed += word + " "
                    placeholder.markdown(displayed + "▌")
                    time.sleep(0.01)
                placeholder.markdown(answer)

            except Exception as e:
                answer = f"Something went wrong: {e}"
                placeholder.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
