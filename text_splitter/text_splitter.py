from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyPDFLoader("document_loader/docker_notes.pdf")
documents = loader.load()

print("Original documents:", len(documents))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))


for i, chunk in enumerate(chunks[:5]):

    print("\n================================")
    print("CHUNK:", i + 1)

    print("\nContent:")
    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)