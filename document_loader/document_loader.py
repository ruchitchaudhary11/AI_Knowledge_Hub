from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("01_document_loader/machine_learning.pdf")

documents = loader.load()

print("Total pages:", len(documents))

for i, document in enumerate(documents[:3]):
    print("\n-----------------------------")
    print("Document:", i + 1)

    print("\nContent:")
    print(document.page_content[:500])

    print("\nMetadata:")
    print(document.metadata)