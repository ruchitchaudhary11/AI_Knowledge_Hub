from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

loader = PyPDFLoader(
    "document_loader/docker_notes.pdf"
)

documents = loader.load()

print("Documents:", len(documents))


# --------------------------------------------------
# 2. Split documents into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Chunks:", len(chunks))


# --------------------------------------------------
# 3. Create embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 4. Create Chroma vector store
# --------------------------------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="docker_notes",
    persist_directory="./chroma_db"
)

print("Vector store created successfully!")

query = "What is Docker?"
results = vector_store.similarity_search(query,
                                         k=3
                                         )
print("Results:")
for result in results:
    print(result.page_content)