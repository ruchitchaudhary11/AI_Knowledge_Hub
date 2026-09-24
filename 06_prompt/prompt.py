from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate


# --------------------------------------------------
# 1. Create embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 2. Load existing vector store
# --------------------------------------------------

vector_store = Chroma(
    collection_name="docker_notes",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# --------------------------------------------------
# 3. Create retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)


# --------------------------------------------------
# 4. User question
# --------------------------------------------------

question = "What is docker?"


# --------------------------------------------------
# 5. Retrieve relevant chunks
# --------------------------------------------------

documents = retriever.invoke(question)


# --------------------------------------------------
# 6. Convert documents into context
# --------------------------------------------------

context = "\n\n".join(
    document.page_content
    for document in documents
)


# --------------------------------------------------
# 7. Create Prompt Template
# --------------------------------------------------

prompt = PromptTemplate(
    template="""
You are a helpful AI assistant.

Answer the question using only the provided context.

If the answer cannot be found in the context,
say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


# --------------------------------------------------
# 8. Format prompt
# --------------------------------------------------

formatted_prompt = prompt.format(
    context=context,
    question=question
)


# --------------------------------------------------
# 9. Display
# --------------------------------------------------

print(formatted_prompt)