from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder


load_dotenv()


# ============================================
# 1. Load Embedding Model
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
# 3. Create Retriever
# ============================================

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 10
    }
)


# ============================================
# 4. Load Cross-Encoder Reranker
# ============================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================
# 5. User Question
# ============================================

question = "What are the advantages of Docker?"


# ============================================
# 6. Retrieve Initial Documents
# ============================================

documents = retriever.invoke(question)

print("\n========================================")
print("RETRIEVED DOCUMENTS")
print("========================================")

for i, document in enumerate(documents):
    print(f"\n--- Document {i + 1} ---")
    print(document.page_content[:300])
    print("Source:", document.metadata.get("source"))
    print("Page:", document.metadata.get("page_label"))


# ============================================
# 7. Prepare Query-Document Pairs
# ============================================

pairs = []

for document in documents:
    pairs.append(
        (question, document.page_content)
    )


# ============================================
# 8. Calculate Relevance Scores
# ============================================

scores = reranker.predict(pairs)


# ============================================
# 9. Combine Documents with Scores
# ============================================

ranked_documents = list(
    zip(documents, scores)
)


# ============================================
# 10. Sort by Relevance
# ============================================

ranked_documents.sort(
    key=lambda x: x[1],
    reverse=True
)


# ============================================
# 11. Display Reranked Documents
# ============================================

print("\n========================================")
print("RERANKED DOCUMENTS")
print("========================================")

for i, (document, score) in enumerate(ranked_documents):

    print(f"\n--- Rank {i + 1} ---")

    print("Score:", round(float(score), 4))

    print(
        "Source:",
        document.metadata.get("source")
    )

    print(
        "Page:",
        document.metadata.get("page_label")
    )

    print(
        "Content:",
        document.page_content[:300]
    )