from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Create embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 3. Load Chroma vector store
# --------------------------------------------------

vector_store = Chroma(
    collection_name="docker_notes",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
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
# 5. Create Prompt
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
# 6. Create LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------------------------
# 7. Output Parser
# --------------------------------------------------

parser = StrOutputParser()


# --------------------------------------------------
# 8. Create RAG Chain
# --------------------------------------------------

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)


# --------------------------------------------------
# 9. Ask question
# --------------------------------------------------

question = "What is capital of India?"

answer = rag_chain.invoke(question)


# --------------------------------------------------
# 10. Display answer
# --------------------------------------------------

print("\nANSWER:")
print(answer)