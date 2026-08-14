import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Chat", page_icon="📄")
st.title("📄 Chat with Any Document")
st.caption("Upload a PDF, then ask questions about it.")

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash")

def build_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the following context. "
    "If the answer isn't in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

def ask(vectorstore, llm, question, k=3):
    retrieved_docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    answer = response.content
    if isinstance(answer, list):
        answer = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
    return answer, retrieved_docs

# --- Sidebar: file upload ---
with st.sidebar:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    file_hash = None
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_hash = hash(file_bytes)

    if uploaded_file and st.session_state.get("last_file_hash") != file_hash:
        with st.spinner("Reading and indexing your PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            st.session_state.vectorstore = build_vectorstore(tmp_path)
            os.unlink(tmp_path)
            st.session_state.last_file_hash = file_hash
            st.session_state.messages = []
        st.success(f"Indexed: {uploaded_file.name}")

# --- Main chat area ---
if "vectorstore" not in st.session_state:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

llm = get_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask(st.session_state.vectorstore, llm, question)
        st.markdown(answer)
        with st.expander("📚 Sources used"):
            for i, doc in enumerate(sources, 1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Chunk {i} (page {page}):** {doc.page_content[:300]}...")
    st.session_state.messages.append({"role": "assistant", "content": answer})