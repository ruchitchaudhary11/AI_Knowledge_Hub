from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor


load_dotenv()


# ============================================
# 1. Load Embeddings
# ============================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================
# 2. Load Existing Vector Store
# ============================================

vector_store = Chroma(
    collection_name="multi_document_rag",
    embedding_function=embeddings,
    persist_directory="./multi_document_chroma"
)


# ============================================
# 3. Create Base Retriever
# ============================================

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 10
    }
)


# ============================================
# 4. Load LLM
# ============================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================
# 5. Create Compressor
# ============================================

compressor = LLMChainExtractor.from_llm(llm)


# ============================================
# 6. Create Compression Retriever
# ============================================

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)


# ============================================
# 7. User Question
# ============================================

question = "What are the advantages of Docker?"


# ============================================
# 8. Normal Retrieval
# ============================================

documents = retriever.invoke(question)

print("\n========================================")
print("NORMAL RETRIEVAL")
print("========================================")

for i, document in enumerate(documents):

    print(f"\n--- Document {i + 1} ---")

    print(document.page_content[:500])

    print(
        "\nSource:",
        document.metadata.get("source")
    )

    print(
        "Page:",
        document.metadata.get("page_label")
    )


# ============================================
# 9. Contextual Compression
# ============================================

compressed_documents = compression_retriever.invoke(
    question
)


# ============================================
# 10. Display Compressed Documents
# ============================================

print("\n========================================")
print("COMPRESSED DOCUMENTS")
print("========================================")

for i, document in enumerate(compressed_documents):

    print(f"\n--- Compressed Document {i + 1} ---")

    print(document.page_content)

    print(
        "\nSource:",
        document.metadata.get("source")
    )

    print(
        "Page:",
        document.metadata.get("page_label")
    )


print("\n========================================")
print("SUMMARY")
print("========================================")

print(
    "Original documents:",
    len(documents)
)

print(
    "Compressed documents:",
    len(compressed_documents)
)