import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================================================
# GET DOCUMENTS FROM CHROMA
# =========================================================

def get_available_documents(vector_store):

    try:

        data = vector_store.get(
            include=["metadatas"]
        )

        metadatas = data.get(
            "metadatas",
            []
        )

        documents = set()

        for metadata in metadatas:

            if metadata:

                source = metadata.get(
                    "source"
                )

                if source:

                    documents.add(
                        source
                    )

        return sorted(documents)

    except Exception:

        return []


# =========================================================
# GET DOCUMENT CONTENT
# =========================================================

def get_document_content(
    vector_store,
    source
):

    try:

        data = vector_store.get(
            where={
                "source": source
            },
            include=[
                "documents",
                "metadatas"
            ]
        )

        documents = data.get(
            "documents",
            []
        )

        if not documents:

            return ""

        return "\n\n".join(
            documents
        )

    except Exception:

        return ""


# =========================================================
# EXTRACT TOPICS
# =========================================================

def extract_topics(
    document_content,
    llm
):

    prompt = ChatPromptTemplate.from_messages(
        [

            (
                "system",

                """
You are an educational content planner.

You are given content extracted from a document.

Analyze ONLY the provided document.

Identify the major concepts/topics that a student
should learn from this document.

Create a logical learning order from basic
concepts to advanced concepts.

Return ONLY valid JSON.

Format:

{{
    "topics": [
        {{
            "title": "Topic name",
            "description": "Short description"
        }}
    ]
}}

Do not include topics that are not supported
by the document.

Document:

{document}
"""
            )

        ]
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(
        {
            "document": document_content
        }
    )

    result = result.strip()

    # Remove markdown code fences
    if result.startswith("```"):

        result = result.replace(
            "```json",
            ""
        )

        result = result.replace(
            "```",
            ""
        )

        result = result.strip()

    try:

        data = json.loads(
            result
        )

        return data.get(
            "topics",
            []
        )

    except Exception:

        return []


# =========================================================
# TEACH TOPIC
# =========================================================

def teach_topic(
    document_content,
    topic,
    level,
    goal,
    llm
):

    prompt = ChatPromptTemplate.from_messages(
        [

            (
                "system",

                """
You are Nexus, an AI teacher.

Teach the requested topic using ONLY the
provided document.

Do not introduce facts that are not supported
by the document.

Student level:
{level}

Learning goal:
{goal}

Topic:
{topic}

Create the lesson using this structure:

## Explanation

Explain the concept clearly according
to the student's level.

## Simple Example

Give an example using information
supported by the document.

## Key Points

Give 3-5 important points.

Do NOT create a question at the end.
The quiz will be generated separately.

Document:

{document}
"""
            )

        ]
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(
        {
            "document": document_content,
            "topic": topic,
            "level": level,
            "goal": goal
        }
    )

    return result


# =========================================================
# GENERATE MCQ QUIZ
# =========================================================

def generate_quiz(
    document_content,
    topic,
    level,
    llm,
    num_questions=5
):

    prompt = ChatPromptTemplate.from_messages(
        [

            (
                "system",

                """
You are Nexus, an AI quiz generator.

Create a multiple-choice quiz about the
specified topic.

Use ONLY the provided document content.

Student level:
{level}

Topic:
{topic}

Create exactly {num_questions} questions.

Each question must have exactly FOUR options:
A, B, C, D.

Only ONE option must be correct.

Return ONLY valid JSON.

Use this exact format:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": {{
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D"
            }},
            "correct_answer": "A",
            "explanation": "Explanation of the correct answer."
        }}
    ]
}}

Rules:

1. Questions must be based ONLY on the document.
2. Questions must focus specifically on the topic.
3. Do not use outside knowledge.
4. Make questions appropriate for the student's level.
5. Make incorrect options plausible.
6. There must be exactly one correct answer.
7. Do not reveal the answer in the question.
8. Keep explanations concise.
9. Do not create duplicate questions.
10. Return ONLY JSON.

Document:

{document}
"""
            )

        ]
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(
        {
            "document": document_content,
            "topic": topic,
            "level": level,
            "num_questions": num_questions
        }
    )

    result = result.strip()

    # Remove markdown code fences
    if result.startswith("```"):

        result = result.replace(
            "```json",
            ""
        )

        result = result.replace(
            "```",
            ""
        )

        result = result.strip()

    try:

        data = json.loads(
            result
        )

        questions = data.get(
            "questions",
            []
        )

        # Basic validation
        valid_questions = []

        for question in questions:

            if not isinstance(
                question,
                dict
            ):

                continue

            options = question.get(
                "options",
                {}
            )

            correct_answer = question.get(
                "correct_answer"
            )

            if (
                question.get("question")
                and isinstance(options, dict)
                and all(
                    key in options
                    for key in ["A", "B", "C", "D"]
                )
                and correct_answer in [
                    "A",
                    "B",
                    "C",
                    "D"
                ]
            ):

                valid_questions.append(
                    question
                )

        return valid_questions

    except json.JSONDecodeError:

        return []