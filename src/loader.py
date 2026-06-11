from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader 

loader = DirectoryLoader(
    path="papers/",
    glob="*.pdf",
    loader_cls=PyMuPDFLoader
)

docs = loader.load()

print(docs[0].metadata)
print(docs[100].metadata)
