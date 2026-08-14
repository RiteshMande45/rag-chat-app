import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Chat", page_icon="📄")
st.title("📄 Chat with Your Document")

@st.cache_resource(show_spinner="Building the knowledge base (first run only)...")
def load_pipeline():
    loader = PyPDFLoader("Understanding_Climate_Change.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the following context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    return vectorstore, llm, prompt

vectorstore, llm, prompt = load_pipeline()

def ask(question, k=3):
    retrieved_docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    answer = response.content
    if isinstance(answer, list):
        answer = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
    return answer

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
            answer = ask(question)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})