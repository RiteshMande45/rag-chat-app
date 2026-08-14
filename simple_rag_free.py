import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

pdf_path = "Understanding_Climate_Change.pdf"

print("Loading and splitting PDF...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

print("Creating embeddings (free, runs locally, may take a minute the first time)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Building FAISS vector index...")
vectorstore = FAISS.from_documents(chunks, embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the following context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

def ask(question, k=3):
    retrieved_docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question about the document (or type 'exit'): ")
        if q.lower() == "exit":
            break
        answer = ask(q)
        if isinstance(answer, list):
            answer = "".join(
                part.get("text", "") for part in answer if isinstance(part, dict)
            )
        print("\nAnswer:", answer)