from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Create embeddings
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
# 5. Create LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ==================================================
# PART A — QUESTION REWRITING
# ==================================================


# --------------------------------------------------
# 6. Question Rewriting Prompt
# --------------------------------------------------

contextualize_q_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
Given a chat history and the latest user question,
rewrite the latest question so that it can be understood
without the chat history.

Do not answer the question.
Only rewrite it as a standalone question.
"""
    ),

    MessagesPlaceholder(
        variable_name="chat_history"
    ),

    (
        "human",
        "{input}"
    )
])


# --------------------------------------------------
# 7. Question Rewriting Chain
# --------------------------------------------------

contextualize_question_chain = (
    contextualize_q_prompt
    | llm
    | StrOutputParser()
)


# ==================================================
# PART B — ANSWER GENERATION
# ==================================================


# --------------------------------------------------
# 8. Answer Prompt
# --------------------------------------------------

answer_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are a helpful AI assistant.

Answer the user's question using only the provided context.

If the answer cannot be found in the context,
say "I don't know based on the provided documents."

Keep the answer clear and concise.

Context:
{context}
"""
    ),

    MessagesPlaceholder(
        variable_name="chat_history"
    ),

    (
        "human",
        "{input}"
    )
])


# --------------------------------------------------
# 9. Answer Chain
# --------------------------------------------------

answer_chain = (
    answer_prompt
    | llm
    | StrOutputParser()
)


# ==================================================
# PART C — CONVERSATION
# ==================================================


# --------------------------------------------------
# 10. Chat History
# --------------------------------------------------

chat_history = [

    HumanMessage(
        content="What is Docker?"
    ),

    AIMessage(
        content="Docker is a platform used for containerization."
    )
]


# --------------------------------------------------
# 11. Current Question
# --------------------------------------------------

question = "What are its advantages?"


# ==================================================
# PART D — RETRIEVAL
# ==================================================


# --------------------------------------------------
# 12. Rewrite Question
# --------------------------------------------------

standalone_question = contextualize_question_chain.invoke({

    "chat_history": chat_history,

    "input": question
})


print("\nSTANDALONE QUESTION:")
print(standalone_question)


# --------------------------------------------------
# 13. Retrieve Relevant Documents
# --------------------------------------------------

documents = retriever.invoke(
    standalone_question
)


print("\nRETRIEVED DOCUMENTS:")
print(len(documents))


# --------------------------------------------------
# 14. Format Documents
# --------------------------------------------------

def format_documents(documents):

    return "\n\n".join(
        document.page_content
        for document in documents
    )


context = format_documents(documents)


# ==================================================
# PART E — ANSWER GENERATION
# ==================================================


# --------------------------------------------------
# 15. Generate Final Answer
# --------------------------------------------------

answer = answer_chain.invoke({

    "chat_history": chat_history,

    "input": question,

    "context": context
})


# --------------------------------------------------
# 16. Print Answer
# --------------------------------------------------

print("\nANSWER:")
print(answer)