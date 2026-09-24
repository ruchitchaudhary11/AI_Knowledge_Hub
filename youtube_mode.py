import re
import os

from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# =========================================
# LOAD ENVIRONMENT VARIABLES
# =========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found in .env"
    )


# =========================================
# LLM
# =========================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key
)


# =========================================
# 1. EXTRACT VIDEO ID
# =========================================

def extract_video_id(url):
    """
    Extract the YouTube video ID from a URL.
    """

    patterns = [
        r"(?:youtube\.com/watch\?v=)([^&]+)",
        r"(?:youtu\.be/)([^?&]+)",
        r"(?:youtube\.com/shorts/)([^?&]+)",
        r"(?:youtube\.com/embed/)([^?&]+)",
    ]

    for pattern in patterns:

        match = re.search(pattern, url)

        if match:
            return match.group(1)

    return None


# =========================================
# 2. GET YOUTUBE TRANSCRIPT
# =========================================

def get_youtube_transcript(url):
    """
    Fetch transcript from YouTube.
    """

    video_id = extract_video_id(url)

    if not video_id:
        raise ValueError(
            "Invalid YouTube URL."
        )

    api = YouTubeTranscriptApi()

    transcript = api.fetch(
        video_id,
        languages=["en", "hi"]
    )

    transcript_text = " ".join(
        snippet.text
        for snippet in transcript
    )

    if not transcript_text.strip():
        raise ValueError(
            "Transcript is empty."
        )

    return transcript_text


# =========================================
# 3. CREATE DOCUMENT CHUNKS
# =========================================

def create_youtube_documents(
    transcript_text,
    video_url
):
    """
    Convert transcript into LangChain Documents
    and split it into chunks.
    """

    document = Document(
        page_content=transcript_text,
        metadata={
            "source": video_url,
            "type": "youtube"
        }
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        [document]
    )

    return chunks


# =========================================
# 4. CREATE VECTOR STORE
# =========================================

def create_youtube_vector_store(
    chunks,
    embeddings
):
    """
    Store YouTube transcript chunks in Chroma.
    """

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="youtube_rag",
        persist_directory="./youtube_chroma"
    )

    return vector_store


# =========================================
# 5. GENERATE ANSWER
# =========================================

def generate_answer(
    query,
    context
):
    """
    Generate an answer using the retrieved
    YouTube transcript context.
    """

    prompt = ChatPromptTemplate.from_messages(
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

7. If the transcript explains a concept
   with an example, include that example
   when relevant.

YouTube Transcript Context:

{context}

User Question:

{question}
"""
            )
        ]
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": query
        }
    )

    return answer


# =========================================
# TEST
# =========================================

if __name__ == "__main__":

    url = input(
        "Enter YouTube URL: "
    )

    try:

        # ---------------------------------
        # 1. Fetch transcript
        # ---------------------------------

        transcript = get_youtube_transcript(
            url
        )

        print(
            "\nTranscript fetched successfully!"
        )

        print(
            f"Total characters: "
            f"{len(transcript)}"
        )


        # ---------------------------------
        # 2. Create chunks
        # ---------------------------------

        chunks = create_youtube_documents(
            transcript,
            url
        )

        print(
            f"Total chunks created: "
            f"{len(chunks)}"
        )


        # ---------------------------------
        # 3. Create embeddings
        # ---------------------------------

        print(
            "\nLoading embedding model..."
        )

        embeddings = HuggingFaceEmbeddings(
            model_name=
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print(
            "Embedding model loaded."
        )


        # ---------------------------------
        # 4. Create Chroma
        # ---------------------------------

        print(
            "\nCreating Chroma vector store..."
        )

        vector_store = (
            create_youtube_vector_store(
                chunks,
                embeddings
            )
        )

        print(
            "YouTube knowledge base created!"
        )


        # ---------------------------------
        # 5. Ask question
        # ---------------------------------

        query = input(
            "\nAsk something about the video: "
        )


        # ---------------------------------
        # 6. Create MMR retriever
        # ---------------------------------

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 10
            }
        )


        # ---------------------------------
        # 7. Retrieve relevant chunks
        # ---------------------------------

        results = retriever.invoke(query)

        if not results:

            print(
                "\nNo relevant information found."
            )

        else:

            print(
                "\nRelevant chunks retrieved:"
            )

            print(
                f"Number of chunks: "
                f"{len(results)}"
            )


            # -----------------------------
            # 8. Create context
            # -----------------------------

            context = "\n\n".join(
                doc.page_content
                for doc in results
            )


            # -----------------------------
            # 9. Generate answer
            # -----------------------------

            print(
                "\nGenerating answer..."
            )

            answer = generate_answer(
                query,
                context
            )


            # -----------------------------
            # 10. Display answer
            # -----------------------------

            print(
                "\n================================="
            )

            print(
                "NEXUS ANSWER"
            )

            print(
                "=================================\n"
            )

            print(answer)


            # -----------------------------
            # 11. Display sources
            # -----------------------------

            print(
                "\n================================="
            )

            print(
                "SOURCES"
            )

            print(
                "================================="
            )

            for i, doc in enumerate(
                results,
                start=1
            ):

                print(
                    f"\n--- Source {i} ---"
                )

                print(
                    doc.page_content[:500]
                )

                print(
                    "\nMetadata:"
                )

                print(
                    doc.metadata
                )


    except Exception as e:

        print(
            "\nError:"
        )

        print(e)