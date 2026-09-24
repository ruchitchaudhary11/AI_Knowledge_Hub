from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


# ============================================
# 1. Load Embeddings
# ============================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================
# 2. Load Vector Store
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
        "k": 3,
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
# 5. Prompt
# ============================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI assistant.

Answer the question using ONLY the
provided context.

If the answer cannot be found in the
context, say:

"I don't know based on the provided documents."

Do not use outside knowledge.

Context:
{context}
"""
    ),
    ("human", "{question}")
])


# ============================================
# 6. RAG Chain
# ============================================

rag_chain = prompt | llm | StrOutputParser()


# ============================================
# 7. Evaluation Dataset
# ============================================

evaluation_dataset = [

    {
        "question": "What is Docker?",
        "reference_answer": (
            "Docker is a platform used to develop, "
            "ship, and run applications using containers."
        )
    },

    {
        "question": "What are the advantages of Docker?",
        "reference_answer": (
            "Docker provides portability, consistency, "
            "isolation, and makes application deployment easier."
        )
    },

    {
        "question": "What is a Docker container?",
        "reference_answer": (
            "A Docker container is a lightweight, "
            "isolated environment used to run an application "
            "and its dependencies."
        )
    },

    {
        "question": "What is Docker Compose?",
        "reference_answer": (
            "Docker Compose is a tool used to define "
            "and run multi-container applications."
        )
    }
]


# ============================================
# 8. Function to Format Documents
# ============================================

def format_documents(documents):

    return "\n\n".join(
        document.page_content
        for document in documents
    )


# ============================================
# 9. Run Evaluation
# ============================================

results = []


for example in evaluation_dataset:

    question = example["question"]

    print("\n========================================")
    print("QUESTION")
    print("========================================")
    print(question)


    # Retrieve documents
    documents = retriever.invoke(question)


    # Create context
    context = format_documents(documents)


    # Generate answer
    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })


    # Store result
    results.append({
        "question": question,
        "reference_answer": example["reference_answer"],
        "generated_answer": answer,
        "retrieved_documents": documents
    })


    print("\nREFERENCE ANSWER:")
    print(example["reference_answer"])


    print("\nGENERATED ANSWER:")
    print(answer)


# ============================================
# 10. Summary
# ============================================

print("\n\n========================================")
print("EVALUATION COMPLETE")
print("========================================")

print(
    "Total questions evaluated:",
    len(results)
)

# ============================================
# 11. Evaluation Prompt
# ============================================

evaluation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an evaluator for a Retrieval-Augmented
Generation system.

Evaluate the generated answer against the
reference answer.

Give a score from 1 to 5:

5 = Completely correct
4 = Mostly correct
3 = Partially correct
2 = Mostly incorrect
1 = Completely incorrect

Return ONLY the number.
"""
    ),
    (
        "human",
        """
Question:
{question}

Reference Answer:
{reference_answer}

Generated Answer:
{generated_answer}
"""
    )
])


# ============================================
# 12. Evaluation Chain
# ============================================

evaluation_chain = (
    evaluation_prompt
    | llm
    | StrOutputParser()
)


# ============================================
# 13. Evaluate Each Result
# ============================================

print("\n\n========================================")
print("ANSWER EVALUATION")
print("========================================")


total_score = 0


for result in results:

    score = evaluation_chain.invoke({
        "question": result["question"],
        "reference_answer": result["reference_answer"],
        "generated_answer": result["generated_answer"]
    })

    try:
        score = int(score.strip())
    except ValueError:
        score = 0

    total_score += score

    print("\n----------------------------------------")

    print("Question:")
    print(result["question"])

    print("\nGenerated Answer:")
    print(result["generated_answer"])

    print("\nScore:", score, "/ 5")


# ============================================
# 14. Calculate Average Score
# ============================================

average_score = total_score / len(results)


print("\n========================================")
print("FINAL EVALUATION")
print("========================================")

print(
    "Average Score:",
    round(average_score, 2),
    "/ 5"
)


# ============================================
# 15. Faithfulness Evaluation
# ============================================

faithfulness_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are evaluating the faithfulness of an answer
generated by a Retrieval-Augmented Generation system.

Determine whether the generated answer is completely
supported by the provided context.

Rules:

- If every claim in the answer is supported by the
  context, return 5.
- If most claims are supported but some are not,
  return 3.
- If the answer is mostly or completely unsupported,
  return 1.

Return ONLY the number: 1, 3, or 5.
"""
    ),
    (
        "human",
        """
Context:
{context}

Generated Answer:
{generated_answer}
"""
    )
])


# ============================================
# 16. Faithfulness Chain
# ============================================

faithfulness_chain = (
    faithfulness_prompt
    | llm
    | StrOutputParser()
)


# ============================================
# 17. Evaluate Faithfulness
# ============================================

print("\n\n========================================")
print("FAITHFULNESS EVALUATION")
print("========================================")


total_faithfulness_score = 0


for result in results:

    # Get document context
    context = format_documents(
        result["retrieved_documents"]
    )

    # Evaluate answer
    score = faithfulness_chain.invoke({
        "context": context,
        "generated_answer": result["generated_answer"]
    })

    try:
        score = int(score.strip())
    except ValueError:
        score = 0

    total_faithfulness_score += score

    print("\n----------------------------------------")

    print("Question:")
    print(result["question"])

    print("\nGenerated Answer:")
    print(result["generated_answer"])

    print("\nFaithfulness Score:", score, "/ 5")


# ============================================
# 18. Average Faithfulness
# ============================================

average_faithfulness = (
    total_faithfulness_score / len(results)
)


print("\n========================================")
print("FAITHFULNESS RESULT")
print("========================================")

print(
    "Average Faithfulness:",
    round(average_faithfulness, 2),
    "/ 5"
)