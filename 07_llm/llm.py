from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 3. Load Chroma
# --------------------------------------------------

vector_store = Chroma(
    collection_name="docker_notes",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# --------------------------------------------------
# 4. Retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)


# --------------------------------------------------
# 5. Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------------------------
# 6. Prompt
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
# 7. Question
# --------------------------------------------------

question = "What is Docker?"


# --------------------------------------------------
# 8. Retrieve
# --------------------------------------------------

documents = retriever.invoke(question)


# --------------------------------------------------
# 9. Create context
# --------------------------------------------------

context = "\n\n".join(
    document.page_content
    for document in documents
)


# --------------------------------------------------
# 10. Format prompt
# --------------------------------------------------

formatted_prompt = prompt.format(
    context=context,
    question=question
)


# --------------------------------------------------
# 11. Invoke Groq
# --------------------------------------------------

response = llm.invoke(formatted_prompt)


# --------------------------------------------------
# 12. Answer
# --------------------------------------------------

print("\nANSWER:")
print(response.content)