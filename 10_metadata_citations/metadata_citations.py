from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==================================================
# 1. Load environment variables
# ==================================================

load_dotenv()


# ==================================================
# 2. Create Embedding Model
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# 3. Load Chroma Vector Store
# ==================================================

vector_store = Chroma(
    collection_name="docker_notes",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# ==================================================
# 4. Create Retriever
# ==================================================

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)


# ==================================================
# 5. Create LLM
# ==================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ==================================================
# 6. Create Answer Prompt
# ==================================================

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI assistant.

Answer the user's question using only the provided context.

If the answer cannot be found in the context,
say:

"I don't know based on the provided documents."

Do not use outside knowledge.

Context:
{context}
"""
    ),

    (
        "human",
        "{question}"
    )
])


# ==================================================
# 7. Create Answer Chain
# ==================================================

answer_chain = (
    answer_prompt
    | llm
    | StrOutputParser()
)


# ==================================================
# 8. Format Retrieved Documents
# ==================================================

def format_documents(documents):

    formatted_documents = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page_label",
            "Unknown"
        )

        content = document.page_content

        formatted_document = (
            f"[Source: {source} | Page: {page}]\n"
            f"{content}"
        )

        formatted_documents.append(
            formatted_document
        )

    return "\n\n".join(formatted_documents)


# ==================================================
# 9. Extract Sources from Metadata
# ==================================================

def extract_sources(documents):

    sources = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page_label",
            "Unknown"
        )

        source_info = f"{source} — Page {page}"

        sources.append(source_info)

    # Remove duplicate sources
    sources = list(dict.fromkeys(sources))

    return sources


# ==================================================
# 10. User Question
# ==================================================

question = "What is Docker?"


# ==================================================
# 11. Retrieve Relevant Documents
# ==================================================

documents = retriever.invoke(question)


print("\nRETRIEVED DOCUMENTS:", len(documents))


# ==================================================
# 12. Create Context
# ==================================================

context = format_documents(documents)


# ==================================================
# 13. Generate Answer
# ==================================================

answer = answer_chain.invoke({
    "context": context,
    "question": question
})


# ==================================================
# 14. Extract Real Sources
# ==================================================

sources = extract_sources(documents)


# ==================================================
# 15. Display Answer
# ==================================================

print("\n========================================")
print("ANSWER")
print("========================================")

print(answer)


# ==================================================
# 16. Display Sources
# ==================================================

print("\n========================================")
print("SOURCES")
print("========================================")

for source in sources:

    print(f"- {source}")