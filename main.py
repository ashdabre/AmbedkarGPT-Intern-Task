# main.py
# RAG Q&A System using LangChain + ChromaDB + HuggingFace Embeddings + Ollama (Mistral)

import os
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

SPEECH_FILE = "speech.txt"
CHROMA_DIR = "chroma_store"


# =========================
# 1. Load speech text
# =========================
def load_text():
    print("[LOG] Loading speech.txt...")
    with open(SPEECH_FILE, "r", encoding="utf-8") as f:
        return f.read()


# =========================
# 2. Split text into chunks
# =========================
def split_into_chunks(text):
    print("[LOG] Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_text(text)


# =========================
# 3. Create vector store
# =========================
def create_vector_store(chunks):
    print("[LOG] Creating embeddings (MiniLM-L6)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("[LOG] Creating ChromaDB store...")
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    print("[LOG] Vector store saved successfully.")
    return vectordb


# =========================
# 4. Retrieve relevant chunks
# =========================
def retrieve_context(vectordb, query):
    return "\n".join([d.page_content for d in vectordb.similarity_search(query, k=3)])


# =========================
# 5. Generate answer using Ollama Mistral
# =========================
def generate_answer(question, context):
    print("[LOG] Querying Mistral (Ollama)...")
    model = OllamaLLM(model="mistral")

    prompt = f"""
Answer the question strictly based on the context below.
If answer is not present, say "The answer is not in the provided text."

Context:
{context}

Question: {question}
"""

    return model.invoke(prompt)


# =========================
# 6. SHOW CHROMA DATABASE
# =========================
def show_chroma_database(vectordb):
    print("\n========================================")
    print("📦 STORED DOCUMENTS IN CHROMA DATABASE")
    print("========================================")

    # FIXED: remove "ids" from include
    data = vectordb.get(include=["documents", "metadatas"])

    print(f"\nTotal Chunks Stored: {len(data['documents'])}\n")

    for i, doc in enumerate(data["documents"]):
        print(f"--- Chunk {i} (ID: {data['ids'][i]}) ---")
        print(doc)
        print()

    print("Note: Embeddings are stored internally but not shown here.")
    print("\n========================================\n")


# =========================
# MAIN SYSTEM
# =========================
def main():
    print("\n=== Ambedkar RAG Q&A System ===\n")

    start = time.time()

    # Build vector DB only once
    if not os.path.exists(CHROMA_DIR):
        print("[LOG] No vector store found. Creating new one...")
        text = load_text()
        chunks = split_into_chunks(text)
        create_vector_store(chunks)
        print("[LOG] Vector store created.\n")
    else:
        print("[LOG] Using existing vector store.\n")

    print("[LOG] Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("[LOG] Loading ChromaDB...")
    vectordb = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DIR)

    print(f"[LOG] System ready in {round(time.time() - start, 2)} sec.\n")

    # Menu Loop
    while True:
        print("\nChoose an option:")
        print("1. Ask a Question (RAG)")
        print("2. View Chroma Database")
        print("3. Exit")

        choice = input("\nEnter choice (1/2/3): ").strip()

        if choice == "1":
            question = input("\nEnter your question: ")
            context = retrieve_context(vectordb, question)
            answer = generate_answer(question, context)
            print("\nAnswer:\n", answer)

        elif choice == "2":
            show_chroma_database(vectordb)

        elif choice == "3":
            print("\nExiting...\n")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
