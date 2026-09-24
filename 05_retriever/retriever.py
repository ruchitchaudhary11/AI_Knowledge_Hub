from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. Create embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 2. Load existing Chroma vector store
# --------------------------------------------------

vector_store = Chroma(
    collection_name="docker_notes",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# --------------------------------------------------
# 3. Check stored documents
# --------------------------------------------------

print(
    "Documents in vector store:",
    vector_store._collection.count()
)


# --------------------------------------------------
# 4. Create Retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)


# --------------------------------------------------
# 5. Query
# --------------------------------------------------

query = "What is docker?"


# --------------------------------------------------
# 6. Retrieve documents
# --------------------------------------------------

results = retriever.invoke(query)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print(
    "\nNumber of retrieved documents:",
    len(results)
)


for i, document in enumerate(results):

    print("\n================================")
    print("RESULT:", i + 1)

    print("\nContent:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)