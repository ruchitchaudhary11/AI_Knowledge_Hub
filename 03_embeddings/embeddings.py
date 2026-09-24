from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------------------------
# 1. Load the PDF
# --------------------------------------------------

loader = PyPDFLoader(
    "document_loader/docker_notes.pdf"
)

documents = loader.load()

print("Number of documents:", len(documents))


# --------------------------------------------------
# 2. Split documents into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# --------------------------------------------------
# 3. Create embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 4. Extract text from chunks
# --------------------------------------------------

texts = [
    chunk.page_content
    for chunk in chunks
]


# --------------------------------------------------
# 5. Generate embeddings
# --------------------------------------------------

vectors = embeddings.embed_documents(texts)


# --------------------------------------------------
# 6. Display information
# --------------------------------------------------

print("Number of vectors:", len(vectors))

print("Embedding dimension:", len(vectors[0]))

print("\nFirst chunk:")
print(chunks[0].page_content[:500])

print("\nFirst vector:")
print(vectors[0][:10])