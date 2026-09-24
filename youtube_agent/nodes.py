import json
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .state import YouTubeLearningState
from .prompts import (
    ANALYZE_VIDEO_PROMPT,
    TEACH_TOPIC_PROMPT,
    QUIZ_PROMPT
)

from youtube_mode import (
    get_youtube_transcript,
    create_youtube_documents,
    create_youtube_vector_store
)

from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")


# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key
)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# NODE 1 — LOAD TRANSCRIPT
# ============================================================

def load_transcript(state: YouTubeLearningState):

    print("\n[1] Loading YouTube transcript...")

    url = state["youtube_url"]

    transcript = get_youtube_transcript(url)

    print("Transcript loaded.")

    return {
        "transcript": transcript
    }


# ============================================================
# NODE 2 — ANALYZE VIDEO
# ============================================================

def analyze_video(state: YouTubeLearningState):

    print("\n[2] Analyzing video...")

    transcript = state["transcript"]

    # ========================================================
    # Split transcript into sections
    # ========================================================

    analysis_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=300
    )

    analysis_chunks = analysis_splitter.split_text(
        transcript
    )

    print(
        f"Total analysis sections: "
        f"{len(analysis_chunks)}"
    )

    all_topics = []

    # ========================================================
    # Prompt
    # ========================================================

    prompt = ChatPromptTemplate.from_template(
        """
You are Nexus, an AI learning assistant.

Analyze this section of a YouTube video.

Identify ONLY the concepts that are explicitly taught
in this transcript section.

Student Level:
{level}

Learning Goal:
{goal}

Transcript Section:
{chunk}

Return ONLY a JSON list.

Each topic must contain:

- title
- description
- difficulty
- evidence

The evidence MUST be a short phrase or sentence
from the transcript that supports the topic.

Example:

[
    {{
        "title": "SQL",
        "description": "A language used to interact with databases.",
        "difficulty": "Beginner",
        "evidence": "SQL is used to communicate with databases."
    }}
]

Rules:

- Use ONLY the transcript section.
- Do NOT use outside knowledge.
- Do NOT infer concepts that are not explicitly discussed.
- Do NOT create generic or related topics.
- Return at most 5 topics.
- Keep each description short.
- Keep each evidence short.
- Return valid JSON only.
- No markdown.
- No ```json.

If there are no meaningful concepts, return [].
"""
    )

    chain = prompt | llm | StrOutputParser()

    # ========================================================
    # Analyze each section
    # ========================================================

    for i, chunk in enumerate(analysis_chunks):

        print(
            f"Analyzing section "
            f"{i + 1}/{len(analysis_chunks)}..."
        )

        try:

            result = chain.invoke({
                "level": state["level"],
                "goal": state["goal"],
                "chunk": chunk
            })

            # =================================================
            # Clean response
            # =================================================

            result = result.strip()

            if result.startswith("```json"):
                result = result[7:]

            if result.startswith("```"):
                result = result[3:]

            if result.endswith("```"):
                result = result[:-3]

            result = result.strip()

            # =================================================
            # Parse JSON
            # =================================================

            topics = json.loads(result)

            if isinstance(topics, list):

                for topic in topics:

                    if not isinstance(topic, dict):
                        continue

                    title = topic.get(
                        "title",
                        ""
                    ).strip()

                    description = topic.get(
                        "description",
                        ""
                    ).strip()

                    evidence = topic.get(
                        "evidence",
                        ""
                    ).strip()

                    if not title or not evidence:
                        continue

                    all_topics.append({
                        "title": title,
                        "description": description,
                        "difficulty": topic.get(
                            "difficulty",
                            "Beginner"
                        ),
                        "evidence": evidence
                    })

        except Exception as e:

            print(
                f"Could not analyze section "
                f"{i + 1}: {e}"
            )

    # ========================================================
    # Remove duplicate topics
    # ========================================================

    unique_topics = []

    seen = set()

    for topic in all_topics:

        title = topic.get(
            "title",
            ""
        ).strip()

        title_key = title.lower()

        if title and title_key not in seen:

            seen.add(title_key)

            unique_topics.append(topic)

    # ========================================================
    # Final result
    # ========================================================

    print(
        f"\nExtracted "
        f"{len(unique_topics)} unique topics."
    )

    return {
        "topics": unique_topics
    }


# ============================================================
# PARSE TOPICS
# ============================================================

def parse_topics(result):

    try:

        result = result.strip()

        # Remove markdown code fences
        if result.startswith("```json"):
            result = result[7:]

        if result.startswith("```"):
            result = result[3:]

        if result.endswith("```"):
            result = result[:-3]

        result = result.strip()

        topics = json.loads(result)

        if isinstance(topics, dict):

            if "topics" in topics:
                return topics["topics"]

        if isinstance(topics, list):
            return topics

    except Exception:

        print(
            "Could not parse structured topics."
        )

        print("Raw result:")
        print(result)

    return []


# ============================================================
# NODE 3 — CONSOLIDATE TOPICS
# ============================================================
def consolidate_topics(state: YouTubeLearningState):

    print("\n[3] Consolidating topics...")

    topics = state["topics"]

    if not topics:
        raise ValueError("No topics were extracted.")

    # ========================================================
    # Create numbered topic list
    # ========================================================

    topic_text = "\n".join(
        f"{i + 1}. {topic.get('title', '')} | "
        f"{topic.get('description', '')}"
        for i, topic in enumerate(topics)
    )

    # ========================================================
    # Prompt
    # ========================================================

    prompt = ChatPromptTemplate.from_template(
        """
You are Nexus, an AI learning assistant.

You are given concepts extracted from a YouTube video.

Your task is to select the most important concepts
for a learning path.

Student Level:
{level}

Learning Goal:
{goal}

Extracted Concepts:

{topics}

Rules:

- Select ONLY concepts from the provided numbered list.
- Do NOT invent any new concept.
- Do NOT modify any concept.
- Do NOT use outside knowledge.
- Remove duplicate concepts.
- Remove very similar concepts.
- Select the 8 most important concepts.
- Arrange them from basic to advanced.
- Return ONLY the NUMBERS of the selected concepts.
- Do not return topic names.
- Do not return descriptions.
- Do not return explanations.
- Return valid JSON.
- No markdown.

Example:

[
    1,
    4,
    7,
    12,
    18,
    23,
    31,
    35
]
"""
    )

    chain = prompt | llm | StrOutputParser()

    try:

        # ====================================================
        # Call LLM
        # ====================================================

        result = chain.invoke({
            "level": state["level"],
            "goal": state["goal"],
            "topics": topic_text
        })

        result = result.strip()

        print(
            f"Consolidation response length: "
            f"{len(result)} characters"
        )

        if not result:

            raise ValueError(
                "LLM returned an empty consolidation response."
            )

        # ====================================================
        # Remove markdown fences
        # ====================================================

        if result.startswith("```json"):
            result = result[7:]

        if result.startswith("```"):
            result = result[3:]

        if result.endswith("```"):
            result = result[:-3]

        result = result.strip()

        # ====================================================
        # Parse JSON
        # ====================================================

        selected_indices = json.loads(result)

        if not isinstance(selected_indices, list):

            raise ValueError(
                "Consolidation response must be a list."
            )

        if not selected_indices:

            raise ValueError(
                "No topics selected by the LLM."
            )

        # ====================================================
        # Convert selected indices → original topics
        # ====================================================

        valid_topics = []

        seen_indices = set()

        for index in selected_indices:

            # Make sure index is an integer
            if not isinstance(index, int):

                continue

            # JSON uses 1-based numbering
            actual_index = index - 1

            # Invalid index
            if actual_index < 0:
                continue

            if actual_index >= len(topics):
                continue

            # Avoid duplicates
            if actual_index in seen_indices:
                continue

            seen_indices.add(actual_index)

            original_topic = topics[actual_index]

            valid_topics.append({
                "title": original_topic.get(
                    "title",
                    ""
                ),

                "description": original_topic.get(
                    "description",
                    ""
                ),

                "difficulty": original_topic.get(
                    "difficulty",
                    "Beginner"
                ),

                # IMPORTANT:
                # Preserve original evidence
                "evidence": original_topic.get(
                    "evidence",
                    ""
                )
            })

        # ====================================================
        # Validate
        # ====================================================

        if not valid_topics:

            raise ValueError(
                "No valid topics were selected."
            )

        # Maximum 8 topics
        valid_topics = valid_topics[:8]

        print(
            f"Final learning path: "
            f"{len(valid_topics)} topics"
        )

        # ====================================================
        # Print learning path
        # ====================================================

        for i, topic in enumerate(valid_topics):

            print(
                f"{i + 1}. "
                f"{topic['title']}"
            )

        return {
            "topics": valid_topics
        }

    except Exception as e:

        print(
            f"Could not consolidate topics: {e}"
        )

        print("Raw response:")

        print(
            result
            if "result" in locals()
            else ""
        )

        raise ValueError(
            "Topic consolidation failed."
        )
# ============================================================
# NODE 4 — SELECT TOPIC
# ============================================================

def select_topic(state: YouTubeLearningState):

    print("\n[3] Selecting topic...")

    topics = state["topics"]

    index = state["current_topic_index"]

    if not topics:

        raise ValueError(
            "No topics were extracted from the video."
        )

    if index >= len(topics):

        return {
            "next_action": "finish"
        }

    topic = topics[index]

    print(
        f"Selected topic: "
        f"{topic.get('title', 'Unknown')}"
    )

    return {
        "current_topic": topic,
        "next_action": "teach"
    }


# ============================================================
# NODE 5 — TEACH TOPIC
# ============================================================

def teach_topic(state: YouTubeLearningState):

    print("\n[4] Teaching topic...")

    # --------------------------------------------------------
    # Create / load YouTube vector store
    # --------------------------------------------------------

    chunks = create_youtube_documents(
        state["transcript"],
        state["youtube_url"]
    )

    vector_store = create_youtube_vector_store(
        chunks,
        embeddings
    )

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15
        }
    )

    # --------------------------------------------------------
    # Current topic
    # --------------------------------------------------------

    topic_title = state["current_topic"]["title"]

    topic_description = state["current_topic"].get(
        "description",
        ""
    )

    topic_evidence = state["current_topic"].get(
        "evidence",
        ""
    )

    # --------------------------------------------------------
    # Better retrieval query
    # --------------------------------------------------------

    retrieval_query = f"""
Topic:
{topic_title}

Topic Description:
{topic_description}

Transcript Evidence:
{topic_evidence}
"""

    print(
        f"Retrieving context for: {topic_title}"
    )

    results = retriever.invoke(
        retrieval_query
    )

    if not results:

        raise ValueError(
            f"No transcript context found for topic: "
            f"{topic_title}"
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    print(
        f"Retrieved {len(results)} transcript chunks."
    )

    # --------------------------------------------------------
    # Generate lesson
    # --------------------------------------------------------

    prompt = ChatPromptTemplate.from_template(
        TEACH_TOPIC_PROMPT
    )

    chain = prompt | llm | StrOutputParser()

    lesson = chain.invoke({
        "topic": topic_title,
        "level": state["level"],
        "goal": state["goal"],
        "context": context
    })

    if not lesson.strip():

        raise ValueError(
            "Lesson generation returned empty output."
        )

    print("Lesson generated.")

    # IMPORTANT:
    # Save the retrieved context so generate_quiz()
    # can use exactly the same evidence.

    return {
        "lesson": lesson,
        "topic_context": context
    }


# ============================================================
# NODE 6 — GENERATE QUIZ
# ============================================================

def generate_quiz(state: YouTubeLearningState):

    print("\n[5] Generating quiz...")

    topic_title = state["current_topic"]["title"]

    topic_description = state["current_topic"].get(
        "description",
        ""
    )

    topic_evidence = state["current_topic"].get(
        "evidence",
        ""
    )

    context = state.get(
        "topic_context",
        ""
    )

    lesson = state["lesson"]

    # ========================================================
    # Quiz Prompt
    # ========================================================

    prompt = ChatPromptTemplate.from_template(
        """
You are Nexus, an AI learning assistant.

Create exactly 5 multiple-choice questions.

Selected Topic:
{topic}

Topic Description:
{description}

Original Transcript Evidence:
{evidence}

Transcript Context:
{context}

Lesson:
{lesson}

IMPORTANT RULES:

1. Use ONLY the provided Transcript Context,
   Original Transcript Evidence, and Lesson.

2. Every question must be about the Selected Topic.

3. Do NOT use outside knowledge.

4. Do NOT create questions about unrelated concepts.

5. Each question must have exactly 4 options.

6. correct_answer MUST be copied EXACTLY
   from one of the four options.

7. NEVER return "Option A", "Option B",
   "Option C", or "Option D" as correct_answer.

8. The correct_answer must be the complete
   option text.

9. Do NOT create questions whose answer is
   that the topic is absent from the transcript.

10. If there is not enough information to create
    questions about the selected topic, return [].

11. Return exactly 5 questions when enough
    information exists.

12. Return ONLY valid JSON.

13. Do not use markdown.

14. Do not use ```json.

15. Keep explanations short.

Return this format:

[
    {{
        "question": "Question text",
        "options": [
            "Complete option text 1",
            "Complete option text 2",
            "Complete option text 3",
            "Complete option text 4"
        ],
        "correct_answer": "Complete option text 2",
        "explanation": "Short explanation."
    }}
]
"""
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "topic": topic_title,
        "description": topic_description,
        "evidence": topic_evidence,
        "context": context,
        "lesson": lesson
    })

    result = result.strip()

    # ========================================================
    # Remove markdown fences
    # ========================================================

    if result.startswith("```json"):
        result = result[7:]

    if result.startswith("```"):
        result = result[3:]

    if result.endswith("```"):
        result = result[:-3]

    result = result.strip()

    # ========================================================
    # Parse JSON
    # ========================================================

    try:

        quiz = json.loads(result)

        if not isinstance(quiz, list):

            raise ValueError(
                "Quiz response is not a list."
            )

        if not quiz:

            raise ValueError(
                "Insufficient transcript evidence for quiz."
            )

        if len(quiz) != 5:

            raise ValueError(
                f"Expected 5 questions, got {len(quiz)}."
            )

        # ====================================================
        # Validate questions
        # ====================================================

        for question in quiz:

            if not isinstance(question, dict):

                raise ValueError(
                    "Invalid question format."
                )

            if "question" not in question:

                raise ValueError(
                    "Question field missing."
                )

            if "options" not in question:

                raise ValueError(
                    "Options field missing."
                )

            if "correct_answer" not in question:

                raise ValueError(
                    "Correct answer missing."
                )

            if "explanation" not in question:

                raise ValueError(
                    "Explanation missing."
                )

            options = question["options"]

            if not isinstance(options, list):

                raise ValueError(
                    "Options must be a list."
                )

            if len(options) != 4:

                raise ValueError(
                    "Each question must have 4 options."
                )

            if len(set(options)) != 4:

                raise ValueError(
                    "Options must be unique."
                )

            # =================================================
            # Critical validation
            # =================================================

            if (
                question["correct_answer"]
                not in options
            ):

                raise ValueError(
                    "Correct answer is not one of the options."
                )

        print(
            f"Quiz generated successfully: "
            f"{len(quiz)} questions"
        )

        return {
            "quiz": quiz
        }

    except Exception as e:

        print(
            f"Could not parse quiz: {e}"
        )

        print(
            "Raw response:"
        )

        print(result)

        raise ValueError(
            "Quiz generation failed."
        )


# ============================================================
# NODE 7 — EVALUATE QUIZ
# ============================================================

def evaluate_quiz(state: YouTubeLearningState):

    print("\n[6] Evaluating quiz...")

    quiz = state["quiz"]

    if not quiz:

        raise ValueError(
            "Cannot evaluate an empty quiz."
        )

    score = 0

    for question in quiz:

        # ----------------------------------------------------
        # TEMPORARY:
        # Automatically selecting correct answer.
        #
        # Later this will be replaced with actual answers
        # submitted by the user through Streamlit.
        # ----------------------------------------------------

        user_answer = question["correct_answer"]

        if user_answer == question["correct_answer"]:

            score += 1

    print(
        f"Score: {score}/{len(quiz)}"
    )

    # --------------------------------------------------------
    # Agent decision
    # --------------------------------------------------------

    if score < 3:

        next_action = "teach_again"

    else:

        next_action = "next_topic"

    return {
        "score": score,
        "next_action": next_action
    }


# ============================================================
# NODE 8 — TEACH AGAIN
# ============================================================

def teach_again(state: YouTubeLearningState):

    print(
        "\n[7] Student needs reinforcement."
    )

    return {}


# ============================================================
# NODE 9 — MOVE TO NEXT TOPIC
# ============================================================

def next_topic(state: YouTubeLearningState):

    print(
        "\n[8] Moving to next topic."
    )

    return {
        "current_topic_index":
            state["current_topic_index"] + 1
    }