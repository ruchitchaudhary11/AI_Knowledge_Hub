from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ==================================================
# 1. Documents Directory
# ==================================================

DOCUMENTS_DIR = Path("documents")


# ==================================================
# 2. Find All PDFs
# ==================================================

pdf_files = list(
    DOCUMENTS_DIR.glob("*.pdf")
)


# ==================================================
# 3. Load All PDFs
# ==================================================

all_documents = []


for pdf_file in pdf_files:

    print(f"\nLoading: {pdf_file}")

    loader = PyPDFLoader(
        str(pdf_file)
    )

    documents = loader.load()

    all_documents.extend(
        documents
    )


# ==================================================
# 4. Create Text Splitter
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# ==================================================
# 5. Split Documents
# ==================================================

chunks = text_splitter.split_documents(
    all_documents
)


print("\n========================================")
print("DOCUMENT STATISTICS")
print("========================================")

print(
    "Total pages:",
    len(all_documents)
)

print(
    "Total chunks:",
    len(chunks)
)


# ==================================================
# 6. Create Embedding Model
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# 7. Create Multi-Document Vector Store
# ==================================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="multi_document_rag",
    persist_directory="./multi_document_chroma"
)


print("\n========================================")
print("VECTOR STORE CREATED")
print("========================================")

print(
    "Total chunks stored:",
    len(chunks)
)


# ==================================================
# 8. Create Retriever
# ==================================================

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)


# ==================================================
# 9. Test Retrieval
# ==================================================

question = "What is Docker?"

documents = retriever.invoke(
    question
)


# ==================================================
# 10. Display Results
# ==================================================

print("\n========================================")
print("RETRIEVED DOCUMENTS")
print("========================================")

print(
    "Number of documents:",
    len(documents)
)


for i, document in enumerate(documents):

    print(
        f"\n--- Document {i + 1} ---"
    )

    print(
        document.page_content[:300]
    )

    print("\nSource:")

    print(
        document.metadata.get(
            "source"
        )
    )

    print("Page:")

    print(
        document.metadata.get(
            "page_label"
        )
    )