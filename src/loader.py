from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader 

class Loader:
    def __init__(self, directory):
        self.dir = directory

    def corpus_loader(self):
        """ Uses langchain's PymuPDFLoadercomponent and give the pdf into [{"metadata": , "page_content": },{}...each page-section has one dict] """
        
        loader = DirectoryLoader(
            path=self.dir,
            glob="*.pdf",
            loader_cls=PyMuPDFLoader
        )

        self.docs = loader.load()

        return self.docs

