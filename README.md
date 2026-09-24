# Nexus --- AI Knowledge Hub

Nexus is an **AI-powered document intelligence and learning platform**
built with Python, LangChain, RAG, Streamlit, and LangGraph.

The project started as a step-by-step implementation of a
Retrieval-Augmented Generation (RAG) pipeline and has evolved into a
learning-focused platform for documents and YouTube videos.

## 🚀 Features

### 📄 Document Intelligence

-   Upload and process PDF documents.
-   Split documents into meaningful chunks.
-   Generate semantic embeddings using Hugging Face.
-   Store embeddings in ChromaDB.
-   Retrieve relevant context for questions.
-   Generate grounded answers using an LLM.

### 🔎 Advanced RAG

Implemented concepts include:

-   Document Loading
-   Text Splitting
-   Embeddings
-   Vector Stores
-   Semantic Retrieval
-   MMR Retrieval
-   Prompt Engineering
-   LLM Integration
-   Conversational RAG
-   Metadata & Citations
-   Multi-Document RAG
-   Reranking
-   Contextual Compression
-   RAG Evaluation

### 🎓 Teach Me

The Teach Me feature turns a document into an interactive learning
experience.

Users can: - Select a document. - Choose a learning level. - Select a
learning goal. - Learn concepts from the document. - Take MCQ-based
quizzes. - Review scores and explanations.

### 🎥 YouTube RAG

Nexus can learn from YouTube videos.

``` text
YouTube URL
     ↓
Transcript Extraction
     ↓
Text Splitting
     ↓
Embeddings
     ↓
ChromaDB
     ↓
Retriever
     ↓
Relevant Transcript Context
     ↓
LLM
     ↓
Answer
```

### 🤖 YouTube Learning Agent

An experimental YouTube Learning Agent is built with LangGraph.

``` text
YouTube Video
     ↓
Transcript
     ↓
Video Analysis
     ↓
Topic Extraction
     ↓
Topic Selection
     ↓
Teaching
     ↓
Quiz Generation
     ↓
Evaluation
     ↓
Next Topic / Re-teach
```

This component is under active development.

## 🧠 Technology Stack

  Category          Technologies
  ----------------- --------------------------------------
  Language          Python
  UI                Streamlit
  LLM Framework     LangChain
  Agent Framework   LangGraph
  LLM               Groq
  Embeddings        Hugging Face / Sentence Transformers
  Vector Database   ChromaDB
  PDF Processing    PyPDFLoader
  YouTube           YouTube Transcript API
  Observability     LangSmith
  Version Control   Git & GitHub

### Embedding Model

``` text
sentence-transformers/all-MiniLM-L6-v2
```

## 📁 Project Structure

``` text
AI-Knowledge-Hub/
│
├── 01_document_loader/
├── 02_text_splitter/
├── 03_embeddings/
├── 04_vector_store/
├── 05_retriever/
├── 06_prompt/
├── 07_llm/
├── 08_rag/
├── 09_conversational_rag/
├── 10_metadata_citations/
├── 11_multi_document_rag/
├── 12_reranking/
├── 13_contextual_compression/
├── 14_rag_evaluation/
│
├── youtube_agent/
│   ├── __init__.py
│   ├── state.py
│   ├── nodes.py
│   ├── graph.py
│   └── prompts.py
│
├── app.py
├── learning_mode.py
├── youtube_mode.py
├── .gitignore
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

``` bash
git clone https://github.com/ruchitchaudhary11/AI_Knowledge_Hub.git
cd AI_Knowledge_Hub
```

### 2. Create a virtual environment

Windows:

``` powershell
python -m venv venv
.env\Scripts\activate
```

### 3. Install dependencies

If `requirements.txt` is present:

``` powershell
pip install -r requirements.txt
```

Otherwise install the project's required LangChain integrations,
Streamlit, ChromaDB, Hugging Face, Groq, LangGraph, and YouTube
Transcript API packages.

## 🔐 Environment Variables

Create a `.env` file in the project root:

``` env
GROQ_API_KEY=your_groq_api_key

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=Nexus-AI-Knowledge-Hub
```

**Never commit `.env` to GitHub.**

Recommended `.gitignore` entries:

``` gitignore
.env
venv/
.venv/
__pycache__/
*.pyc
chroma_db/
multi_document_chroma/
youtube_chroma/
```

## ▶️ Run the Application

``` powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## 🏗️ Core RAG Architecture

``` text
                 Document
                    │
                    ▼
            Document Loader
                    │
                    ▼
              Text Splitter
                    │
                    ▼
                 Chunks
                    │
                    ▼
               Embeddings
                    │
                    ▼
               ChromaDB
                    │
                    ▼
               Retriever
                    │
              User Question
                    │
                    ▼
             Relevant Chunks
                    │
                    ▼
                 Prompt
                    │
                    ▼
                  LLM
                    │
                    ▼
                 Answer
```

## 📚 Learning Workflow

``` text
Select Document
       ↓
Choose Level
       ↓
Choose Learning Goal
       ↓
Extract Topics
       ↓
Generate Lesson
       ↓
Generate MCQs
       ↓
Evaluate
       ↓
Show Score & Explanations
```

## 🎯 Project Goals

Nexus is a practical project for exploring:

-   Retrieval-Augmented Generation
-   Semantic Search
-   Vector Databases
-   Embeddings
-   LLM Applications
-   Conversational AI
-   Agentic AI
-   LangGraph
-   RAG Evaluation
-   Document Intelligence
-   AI-powered Learning Systems

The project is developed feature-by-feature to understand how each
component works instead of treating RAG as a black box.

## 🔬 Future Improvements

Planned areas include:

-   Transformer-based neural reranking
-   Improved YouTube Learning Agent
-   Adaptive learning based on quiz performance
-   Learner mastery prediction
-   Better agent decision-making
-   More robust RAG evaluation
-   Improved LangSmith tracing
-   Additional document formats
-   Learning analytics
-   More efficient LLM token usage

## 👨‍💻 Author

**Ruchit Chaudhary**

B.Tech --- Artificial Intelligence\
GL Bajaj Institute of Technology and Management

GitHub: https://github.com/ruchitchaudhary11

## ⭐ Project Status

Nexus is an **actively developing project**. Core document RAG,
multi-document RAG, retrieval improvements, learning features, and
YouTube RAG are implemented, while the advanced agentic learning
workflow is still being improved.
