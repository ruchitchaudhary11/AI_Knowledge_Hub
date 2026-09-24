# =========================================================
# IMPORTS
# =========================================================

import os
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from learning_mode import (
    get_available_documents,
    get_document_content,
    extract_topics,
    teach_topic,
    generate_quiz
)

from youtube_mode import (
    get_youtube_transcript,
    create_youtube_documents,
    create_youtube_vector_store
)

from youtube_agent.graph import app as youtube_agent


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Nexus - Document Intelligence",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🧠 Nexus")

st.caption(
    "AI-powered document intelligence using LangChain + RAG"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Nexus")

st.sidebar.subheader("Navigation")

page = st.sidebar.radio(
    "Navigate",
    [
        "🔎 Ask Documents",
        "📄 Documents",
        "🎓 Teach Me",
        "🎥 YouTube RAG",
        "🤖 YouTube Learning Agent"
    ]
)


# =========================================================
# CHECK GROQ API KEY
# =========================================================

if not api_key:

    st.sidebar.error(
        "GROQ_API_KEY not found in .env"
    )

    rag_ready = False

else:

    rag_ready = True


# =========================================================
# INITIALIZE EMBEDDINGS
# =========================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# INITIALIZE VECTOR STORE
# =========================================================

@st.cache_resource
def load_vector_store(_embeddings):

    return Chroma(
        collection_name="multi_document_rag",
        embedding_function=_embeddings,
        persist_directory="./multi_document_chroma"
    )


# =========================================================
# INITIALIZE LLM
# =========================================================

@st.cache_resource
def load_llm():

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key
    )


# =========================================================
# LOAD RAG COMPONENTS
# =========================================================

if rag_ready:

    try:

        embeddings = load_embeddings()

        vector_store = load_vector_store(
            embeddings
        )

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 10
            }
        )

        llm = load_llm()

    except Exception as e:

        rag_ready = False

        st.sidebar.error(
            f"RAG initialization failed: {e}"
        )


# =========================================================
# SIDEBAR - DOCUMENT UPLOAD
# =========================================================

st.sidebar.divider()

st.sidebar.subheader("📄 Add Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# =========================================================
# INDEX DOCUMENTS
# =========================================================

if st.sidebar.button(
    "📚 Index Documents",
    type="primary"
):

    if not uploaded_files:

        st.sidebar.warning(
            "Please upload at least one PDF."
        )

    elif not rag_ready:

        st.sidebar.error(
            "RAG system is not available."
        )

    else:

        try:

            documents_dir = Path("documents")

            documents_dir.mkdir(
                exist_ok=True
            )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            total_chunks = 0

            progress = st.sidebar.empty()

            for uploaded_file in uploaded_files:

                # -----------------------------------------
                # SAVE FILE
                # -----------------------------------------

                file_path = (
                    documents_dir
                    / uploaded_file.name
                )

                with open(
                    file_path,
                    "wb"
                ) as f:

                    f.write(
                        uploaded_file.getbuffer()
                    )

                # -----------------------------------------
                # LOAD PDF
                # -----------------------------------------

                loader = PyPDFLoader(
                    str(file_path)
                )

                docs = loader.load()

                # -----------------------------------------
                # NORMALIZE SOURCE METADATA
                # -----------------------------------------

                source_name = (
                    f"documents/"
                    f"{uploaded_file.name}"
                )

                for document in docs:

                    document.metadata[
                        "source"
                    ] = source_name

                # -----------------------------------------
                # SPLIT DOCUMENT
                # -----------------------------------------

                chunks = splitter.split_documents(
                    docs
                )

                # -----------------------------------------
                # REMOVE OLD VERSION
                # -----------------------------------------

                try:

                    vector_store.delete(
                        where={
                            "source": source_name
                        }
                    )

                except Exception:

                    pass

                # -----------------------------------------
                # ADD TO VECTOR STORE
                # -----------------------------------------

                vector_store.add_documents(
                    documents=chunks
                )

                total_chunks += len(chunks)

                progress.info(
                    f"Indexed: {uploaded_file.name}"
                )

            progress.success(
                f"Successfully indexed "
                f"{total_chunks} chunks."
            )

            st.session_state[
                "documents_indexed"
            ] = True

        except Exception as e:

            st.sidebar.error(
                f"Indexing failed: {e}"
            )


# =========================================================
# PAGE 1
# ASK DOCUMENTS
# =========================================================

if page == "🔎 Ask Documents":

    st.divider()

    st.header("🔎 Ask your documents")

    st.write(
        "Enter a question and Nexus will retrieve "
        "the most relevant information from your PDFs."
    )

    question = st.text_input(
        "Question",
        placeholder="Example: What is Docker Compose?"
    )

    ask_button = st.button(
        "🔍 Search",
        type="primary"
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        elif not rag_ready:

            st.error(
                "RAG system is not available."
            )

        else:

            try:

                with st.spinner(
                    "Searching your documents..."
                ):

                    documents = retriever.invoke(
                        question
                    )

                if not documents:

                    st.warning(
                        "No relevant information "
                        "was found in your documents."
                    )

                else:

                    context = "\n\n".join(
                        document.page_content
                        for document in documents
                    )

                    prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                """
You are Nexus, an AI document assistant.

Answer the user's question using ONLY
the information contained in the provided
document context.

If the answer cannot be found in the
provided context, say:

"I don't know based on the provided documents."

Do not use outside knowledge.

DOCUMENT CONTEXT:

{context}
"""
                            ),
                            (
                                "human",
                                "{question}"
                            )
                        ]
                    )

                    rag_chain = (
                        prompt
                        | llm
                        | StrOutputParser()
                    )

                    with st.spinner(
                        "Generating answer..."
                    ):

                        answer = rag_chain.invoke(
                            {
                                "context": context,
                                "question": question
                            }
                        )

                    st.subheader("💡 Answer")

                    st.write(answer)

                    st.subheader("📑 Sources")

                    for index, document in enumerate(
                        documents,
                        start=1
                    ):

                        source = document.metadata.get(
                            "source",
                            "Unknown document"
                        )

                        page_number = document.metadata.get(
                            "page_label",
                            document.metadata.get(
                                "page",
                                "?"
                            )
                        )

                        file_name = Path(
                            source
                        ).name

                        with st.expander(
                            f"Source {index}: "
                            f"{file_name} — Page {page_number}"
                        ):

                            st.write(
                                document.page_content
                            )

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )


# =========================================================
# PAGE 2
# DOCUMENTS
# =========================================================

elif page == "📄 Documents":

    st.header("📄 Your Documents")

    st.write(
        "View the documents currently available "
        "in the Nexus knowledge base."
    )

    if not rag_ready:

        st.error(
            "RAG system is not available."
        )

    else:

        try:

            available_documents = (
                get_available_documents(
                    vector_store
                )
            )

            if not available_documents:

                st.info(
                    "No documents have been indexed yet."
                )

            else:

                st.subheader(
                    f"📚 {len(available_documents)} "
                    f"document(s) indexed"
                )

                for index, source in enumerate(
                    available_documents,
                    start=1
                ):

                    file_name = Path(
                        source
                    ).name

                    st.write(
                        f"**{index}.** 📄 {file_name}"
                    )

        except Exception as e:

            st.error(
                f"Could not load documents: {e}"
            )


# =========================================================
# PAGE 3
# TEACH ME
# =========================================================

elif page == "🎓 Teach Me":

    st.header("🎓 Teach Me This Document")

    st.write(
        "Turn your document into a personalized "
        "learning experience."
    )

    if not rag_ready:

        st.error(
            "RAG system is not available."
        )

    else:

        available_documents = (
            get_available_documents(
                vector_store
            )
        )

        if not available_documents:

            st.info(
                "No indexed documents found. "
                "Upload and index a PDF first."
            )

        else:

            selected_document = st.selectbox(
                "📄 Select a document",
                available_documents,
                format_func=lambda x: Path(x).name
            )

            level = st.selectbox(
                "🎯 Learning level",
                [
                    "Beginner",
                    "College Student",
                    "Interview Preparation",
                    "Advanced"
                ]
            )

            goal = st.selectbox(
                "🎓 Learning goal",
                [
                    "Understand the concepts",
                    "Prepare for an exam",
                    "Prepare for an interview",
                    "Master the document"
                ]
            )

            if st.button(
                "🚀 Start Learning",
                type="primary"
            ):

                with st.spinner(
                    "Analyzing your document..."
                ):

                    document_content = (
                        get_document_content(
                            vector_store,
                            selected_document
                        )
                    )

                    if not document_content:

                        st.error(
                            "Could not retrieve "
                            "document content."
                        )

                    else:

                        topics = extract_topics(
                            document_content,
                            llm
                        )

                        st.session_state[
                            "learning_document"
                        ] = selected_document

                        st.session_state[
                            "learning_content"
                        ] = document_content

                        st.session_state[
                            "learning_topics"
                        ] = topics

                        st.session_state[
                            "learning_level"
                        ] = level

                        st.session_state[
                            "learning_goal"
                        ] = goal

                        st.success(
                            "Learning path created!"
                        )

            if (
                "learning_topics"
                in st.session_state
            ):

                topics = st.session_state[
                    "learning_topics"
                ]

                if not topics:

                    st.warning(
                        "Nexus could not identify "
                        "topics from this document."
                    )

                else:

                    st.divider()

                    st.subheader(
                        "🧠 Your Learning Path"
                    )

                    st.write(
                        f"**Level:** "
                        f"{st.session_state['learning_level']}"
                    )

                    st.write(
                        f"**Goal:** "
                        f"{st.session_state['learning_goal']}"
                    )

                    topic_names = [
                        topic.get(
                            "title",
                            "Untitled Topic"
                        )
                        for topic in topics
                    ]

                    selected_topic = st.selectbox(
                        "📚 Choose a topic to learn",
                        topic_names
                    )

                    topic_data = next(
                        (
                            topic
                            for topic in topics
                            if topic.get(
                                "title"
                            ) == selected_topic
                        ),
                        None
                    )

                    if topic_data:

                        st.info(
                            topic_data.get(
                                "description",
                                ""
                            )
                        )

                    if st.button(
                        "👨‍🏫 Teach Me",
                        type="primary"
                    ):

                        with st.spinner(
                            f"Preparing your lesson "
                            f"on {selected_topic}..."
                        ):

                            lesson = teach_topic(
                                st.session_state[
                                    "learning_content"
                                ],
                                selected_topic,
                                st.session_state[
                                    "learning_level"
                                ],
                                st.session_state[
                                    "learning_goal"
                                ],
                                llm
                            )

                        st.session_state[
                            "current_lesson"
                        ] = lesson

                        st.session_state[
                            "current_topic"
                        ] = selected_topic

                        st.session_state.pop(
                            "current_quiz",
                            None
                        )

                        st.session_state.pop(
                            "quiz_answers",
                            None
                        )

                        st.session_state.pop(
                            "quiz_score",
                            None
                        )

                        st.session_state[
                            "quiz_submitted"
                        ] = False

                    if (
                        "current_lesson"
                        in st.session_state
                    ):

                        st.divider()

                        st.subheader(
                            f"📖 "
                            f"{st.session_state['current_topic']}"
                        )

                        st.markdown(
                            st.session_state[
                                "current_lesson"
                            ]
                        )

                        st.divider()

                        st.subheader(
                            "🧠 Test Your Understanding"
                        )

                        st.write(
                            "Test your understanding with "
                            "5 multiple-choice questions."
                        )

                        if st.button(
                            "🎯 Generate Quiz",
                            type="primary"
                        ):

                            with st.spinner(
                                "Creating your quiz..."
                            ):

                                quiz = generate_quiz(
                                    st.session_state[
                                        "learning_content"
                                    ],
                                    st.session_state[
                                        "current_topic"
                                    ],
                                    st.session_state[
                                        "learning_level"
                                    ],
                                    llm,
                                    num_questions=5
                                )

                            if not quiz:

                                st.error(
                                    "Nexus could not generate "
                                    "the quiz. Please try again."
                                )

                            else:

                                st.session_state[
                                    "current_quiz"
                                ] = quiz

                                st.session_state[
                                    "quiz_submitted"
                                ] = False

                                st.session_state.pop(
                                    "quiz_answers",
                                    None
                                )

                                st.session_state.pop(
                                    "quiz_score",
                                    None
                                )

                                st.success(
                                    f"Quiz created with "
                                    f"{len(quiz)} questions!"
                                )

                        if (
                            "current_quiz"
                            in st.session_state
                            and not st.session_state.get(
                                "quiz_submitted",
                                False
                            )
                        ):

                            quiz = st.session_state[
                                "current_quiz"
                            ]

                            st.divider()

                            st.subheader(
                                "📝 Quiz"
                            )

                            st.write(
                                "Select one answer for "
                                "each question."
                            )

                            selected_answers = {}

                            for index, question_data in enumerate(
                                quiz,
                                start=1
                            ):

                                st.markdown(
                                    f"### Question {index}"
                                )

                                question_text = (
                                    question_data.get(
                                        "question",
                                        "Question unavailable"
                                    )
                                )

                                st.write(
                                    question_text
                                )

                                options = (
                                    question_data.get(
                                        "options",
                                        {}
                                    )
                                )

                                selected_answer = st.radio(
                                    "Choose your answer:",
                                    [
                                        "A",
                                        "B",
                                        "C",
                                        "D"
                                    ],
                                    format_func=lambda option:
                                        f"{option}. "
                                        f"{options.get(option, '')}",
                                    key=f"quiz_question_{index}"
                                )

                                selected_answers[
                                    index - 1
                                ] = selected_answer

                                st.divider()

                            if st.button(
                                "✅ Submit Quiz",
                                type="primary"
                            ):

                                score = 0

                                for index, question_data in enumerate(
                                    quiz
                                ):

                                    correct_answer = (
                                        question_data.get(
                                            "correct_answer"
                                        )
                                    )

                                    user_answer = (
                                        selected_answers[
                                            index
                                        ]
                                    )

                                    if (
                                        user_answer
                                        == correct_answer
                                    ):

                                        score += 1

                                st.session_state[
                                    "quiz_answers"
                                ] = selected_answers

                                st.session_state[
                                    "quiz_score"
                                ] = score

                                st.session_state[
                                    "quiz_submitted"
                                ] = True

                                st.rerun()

                        if st.session_state.get(
                            "quiz_submitted",
                            False
                        ):

                            quiz = st.session_state[
                                "current_quiz"
                            ]

                            answers = st.session_state[
                                "quiz_answers"
                            ]

                            score = st.session_state[
                                "quiz_score"
                            ]

                            total = len(quiz)

                            percentage = (
                                score / total
                            ) * 100

                            st.divider()

                            st.subheader(
                                "🏆 Quiz Result"
                            )

                            st.metric(
                                "Your Score",
                                f"{score}/{total}"
                            )

                            st.write(
                                f"**Percentage:** "
                                f"{percentage:.0f}%"
                            )

                            if percentage >= 80:

                                st.success(
                                    "🎉 Excellent! "
                                    "You have a strong understanding "
                                    "of this topic."
                                )

                            elif percentage >= 50:

                                st.info(
                                    "👍 Good effort! "
                                    "Review the incorrect answers "
                                    "and try again."
                                )

                            else:

                                st.warning(
                                    "📚 Keep practicing! "
                                    "You should review this topic "
                                    "before moving ahead."
                                )

                            st.subheader(
                                "📋 Answer Review"
                            )

                            for index, question_data in enumerate(
                                quiz,
                                start=1
                            ):

                                question_text = (
                                    question_data.get(
                                        "question",
                                        ""
                                    )
                                )

                                options = (
                                    question_data.get(
                                        "options",
                                        {}
                                    )
                                )

                                correct_answer = (
                                    question_data.get(
                                        "correct_answer"
                                    )
                                )

                                user_answer = answers[
                                    index - 1
                                ]

                                st.markdown(
                                    f"**Q{index}. "
                                    f"{question_text}**"
                                )

                                st.write(
                                    f"Your answer: "
                                    f"**{user_answer}. "
                                    f"{options.get(user_answer, '')}**"
                                )

                                st.write(
                                    f"Correct answer: "
                                    f"**{correct_answer}. "
                                    f"{options.get(correct_answer, '')}**"
                                )

                                if (
                                    user_answer
                                    == correct_answer
                                ):

                                    st.success(
                                        "✅ Correct"
                                    )

                                else:

                                    st.error(
                                        "❌ Incorrect"
                                    )

                                explanation = (
                                    question_data.get(
                                        "explanation",
                                        ""
                                    )
                                )

                                if explanation:

                                    st.info(
                                        f"💡 {explanation}"
                                    )

                                st.divider()

                            if st.button(
                                "🔄 Generate New Quiz"
                            ):

                                with st.spinner(
                                    "Creating a new quiz..."
                                ):

                                    new_quiz = generate_quiz(
                                        st.session_state[
                                            "learning_content"
                                        ],
                                        st.session_state[
                                            "current_topic"
                                        ],
                                        st.session_state[
                                            "learning_level"
                                        ],
                                        llm,
                                        num_questions=5
                                    )

                                if new_quiz:

                                    st.session_state[
                                        "current_quiz"
                                    ] = new_quiz

                                    st.session_state[
                                        "quiz_submitted"
                                    ] = False

                                    st.session_state.pop(
                                        "quiz_answers",
                                        None
                                    )

                                    st.session_state.pop(
                                        "quiz_score",
                                        None
                                    )

                                    st.rerun()

                                else:

                                    st.error(
                                        "Could not generate "
                                        "a new quiz."
                                    )


# =========================================================
# PAGE 4
# YOUTUBE RAG
# =========================================================

elif page == "🎥 YouTube RAG":

    st.header("🎥 Learn from YouTube")

    st.write(
        "Paste a YouTube video URL and ask questions "
        "about the video."
    )

    st.divider()

    youtube_url = st.text_input(
        "🔗 YouTube Video URL",
        placeholder=(
            "https://www.youtube.com/watch?v=..."
        )
    )

    if st.button(
        "🚀 Process Video",
        type="primary"
    ):

        if not youtube_url.strip():

            st.warning(
                "Please enter a YouTube URL."
            )

        elif not rag_ready:

            st.error(
                "RAG system is not available."
            )

        else:

            try:

                with st.spinner(
                    "🎙️ Fetching YouTube transcript..."
                ):

                    transcript = get_youtube_transcript(
                        youtube_url
                    )

                st.success(
                    "✅ Transcript fetched successfully!"
                )

                with st.spinner(
                    "✂️ Splitting transcript into chunks..."
                ):

                    chunks = create_youtube_documents(
                        transcript,
                        youtube_url
                    )

                with st.spinner(
                    "🧠 Creating YouTube knowledge base..."
                ):

                    youtube_vector_store = (
                        create_youtube_vector_store(
                            chunks,
                            embeddings
                        )
                    )

                st.session_state[
                    "youtube_vector_store"
                ] = youtube_vector_store

                st.session_state[
                    "youtube_url"
                ] = youtube_url

                st.session_state[
                    "youtube_chunks"
                ] = len(chunks)

                st.session_state[
                    "youtube_processed"
                ] = True

                st.success(
                    f"🎉 Video processed successfully! "
                    f"{len(chunks)} chunks created."
                )

            except Exception as e:

                st.error(
                    f"Could not process the video: {e}"
                )

    if st.session_state.get(
        "youtube_processed",
        False
    ):

        st.divider()

        st.subheader(
            "✅ Video Ready"
        )

        st.write(
            f"**Chunks:** "
            f"{st.session_state['youtube_chunks']}"
        )

        st.subheader(
            "💬 Ask about the video"
        )

        youtube_question = st.text_input(
            "Your question",
            placeholder=(
                "Example: What is persistence in LangGraph?"
            ),
            key="youtube_question"
        )

        if st.button(
            "🔍 Ask",
            type="primary"
        ):

            if not youtube_question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                try:

                    youtube_retriever = (
                        st.session_state[
                            "youtube_vector_store"
                        ].as_retriever(
                            search_type="mmr",
                            search_kwargs={
                                "k": 3,
                                "fetch_k": 10
                            }
                        )
                    )

                    with st.spinner(
                        "🔎 Searching the video..."
                    ):

                        youtube_documents = (
                            youtube_retriever.invoke(
                                youtube_question
                            )
                        )

                    if not youtube_documents:

                        st.warning(
                            "No relevant information "
                            "was found in the video."
                        )

                    else:

                        youtube_context = (
                            "\n\n".join(
                                document.page_content
                                for document
                                in youtube_documents
                            )
                        )

                        youtube_prompt = (
                            ChatPromptTemplate.from_messages(
                                [
                                    (
                                        "system",
                                        """
You are Nexus, an AI learning assistant.

Answer the user's question using ONLY
the provided YouTube transcript context.

Rules:

1. Do not use outside knowledge.

2. If the answer is not present in the
provided transcript, say:

"I don't know based on the video transcript."

3. Explain the answer clearly.

4. Use simple language.

5. The transcript may contain Hindi,
Hinglish, or English.

6. Answer in clear English.

7. If the video explains a concept
with an example, include the example
when relevant.

YouTube Transcript Context:

{context}
"""
                                    ),
                                    (
                                        "human",
                                        "{question}"
                                    )
                                ]
                            )
                        )

                        youtube_chain = (
                            youtube_prompt
                            | llm
                            | StrOutputParser()
                        )

                        with st.spinner(
                            "🤖 Generating answer..."
                        ):

                            youtube_answer = (
                                youtube_chain.invoke(
                                    {
                                        "context":
                                            youtube_context,
                                        "question":
                                            youtube_question
                                    }
                                )
                            )

                        st.subheader(
                            "💡 Answer"
                        )

                        st.write(
                            youtube_answer
                        )

                        st.subheader(
                            "📚 Video Sources"
                        )

                        for index, document in enumerate(
                            youtube_documents,
                            start=1
                        ):

                            with st.expander(
                                f"Source {index}"
                            ):

                                st.write(
                                    document.page_content
                                )

                                st.caption(
                                    f"Source: "
                                    f"{document.metadata.get(
                                        'source',
                                        'YouTube'
                                    )}"
                                )

                except Exception as e:

                    st.error(
                        f"Something went wrong: {e}"
                    )


# =========================================================
# PAGE 5
# YOUTUBE LEARNING AGENT
# =========================================================

elif page == "🤖 YouTube Learning Agent":

    st.header("🎓 YouTube Learning Agent")

    st.write(
        "Turn any YouTube video into a personalized "
        "step-by-step learning experience."
    )

    st.divider()

    # =====================================================
    # VIDEO URL
    # =====================================================

    youtube_agent_url = st.text_input(
        "🔗 YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    # =====================================================
    # LEVEL
    # =====================================================

    agent_level = st.selectbox(
        "🎯 Your Level",
        [
            "Beginner",
            "College Student",
            "Interview Preparation",
            "Advanced"
        ]
    )

    # =====================================================
    # GOAL
    # =====================================================

    agent_goal = st.selectbox(
        "🎓 Learning Goal",
        [
            "Understand concepts",
            "Exam Preparation",
            "Interview Preparation",
            "Master the topic"
        ]
    )

    # =====================================================
    # START AGENT
    # =====================================================

    if st.button(
        "🚀 Start Learning",
        type="primary"
    ):

        if not youtube_agent_url.strip():

            st.warning(
                "Please enter a YouTube URL."
            )

        else:

            # =============================================
            # INITIAL STATE
            # =============================================

            initial_state = {

                "youtube_url":
                    youtube_agent_url,

                "level":
                    agent_level,

                "goal":
                    agent_goal,

                "transcript":
                    "",

                "topics":
                    [],

                "current_topic_index":
                    0,

                "current_topic":
                    {},

                # IMPORTANT:
                # Context retrieved for the selected topic.
                "topic_context":
                    "",

                "lesson":
                    "",

                "quiz":
                    [],

                "score":
                    0,

                "next_action":
                    ""
            }

            # =============================================
            # RUN AGENT
            # =============================================

            try:

                with st.spinner(
                    "🤖 Nexus is analyzing the video..."
                ):

                    result = youtube_agent.invoke(
                        initial_state
                    )

                st.session_state[
                    "youtube_agent_result"
                ] = result

                st.success(
                    "🎉 Learning session completed!"
                )

                # =========================================
                # LEARNING PATH
                # =========================================

                st.divider()

                st.subheader(
                    "🧠 Learning Path"
                )

                topics = result.get(
                    "topics",
                    []
                )

                if topics:

                    for index, topic in enumerate(
                        topics,
                        start=1
                    ):

                        st.markdown(
                            f"**{index}. "
                            f"{topic.get('title', 'Unknown Topic')}**"
                        )

                        description = topic.get(
                            "description",
                            ""
                        )

                        if description:

                            st.caption(
                                description
                            )

                else:

                    st.info(
                        "No learning topics were returned."
                    )

                # =========================================
                # CURRENT TOPIC
                # =========================================

                current_topic = result.get(
                    "current_topic",
                    {}
                )

                if current_topic:

                    st.divider()

                    st.subheader(
                        f"📌 "
                        f"{current_topic.get('title', 'Current Topic')}"
                    )

                    description = current_topic.get(
                        "description",
                        ""
                    )

                    if description:

                        st.write(
                            description
                        )

                # =========================================
                # LESSON
                # =========================================

                lesson = result.get(
                    "lesson",
                    ""
                )

                if lesson:

                    st.divider()

                    st.subheader(
                        "📖 Lesson"
                    )

                    st.markdown(
                        lesson
                    )

                # =========================================
                # QUIZ
                # =========================================

                quiz = result.get(
                    "quiz",
                    []
                )

                if quiz:

                    st.divider()

                    st.subheader(
                        "📝 Quiz"
                    )

                    for index, question in enumerate(
                        quiz,
                        start=1
                    ):

                        st.markdown(
                            f"### Q{index}. "
                            f"{question.get('question', '')}"
                        )

                        options = question.get(
                            "options",
                            []
                        )

                        if isinstance(
                            options,
                            list
                        ):

                            for option in options:

                                st.write(
                                    f"- {option}"
                                )

                        elif isinstance(
                            options,
                            dict
                        ):

                            for key, value in options.items():

                                st.write(
                                    f"- {key}. {value}"
                                )

                        st.info(
                            "Correct Answer: "
                            f"{question.get('correct_answer', '')}"
                        )

                        explanation = question.get(
                            "explanation",
                            ""
                        )

                        if explanation:

                            st.caption(
                                f"💡 {explanation}"
                            )

                # =========================================
                # SCORE
                # =========================================

                st.divider()

                score = result.get(
                    "score",
                    0
                )

                total_questions = len(
                    quiz
                )

                st.subheader(
                    "🏆 Score"
                )

                st.metric(
                    "Current Score",
                    f"{score}/{total_questions}"
                )

            except Exception as e:

                st.error(
                    f"Agent failed: {e}"
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Nexus | LangChain + Chroma + HuggingFace + Groq"
)