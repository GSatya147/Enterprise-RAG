import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from src.loader import Loader

class Chunker:
    def __init__(self, chunk_size, loader_docs, tokenizer):
        self.CHUNK_SIZE = chunk_size
        self.CHUNK_OVERLAP = int(0.15 * self.CHUNK_SIZE)

        self.data = loader_docs
        self.tokenizer = tokenizer

    def corpus_chunker(self)-> list[dict]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            add_start_index=True,
            length_function=lambda text: len(self.tokenizer.encode(text))
        )

        self.chunks = splitter.split_documents(self.data)

        document_dict = dict()
        chunk_dict_list = []

        for i in range(len(self.chunks)):
            document_dict = {
                "content": self.chunks[i].page_content,
                "metadata": {
                    "source": self.chunks[i].metadata.get("source"),
                    "document_name": os.path.basename(self.chunks[i].metadata.get("source")),
                    "chunk_index": i,
                    "token_count": len(self.tokenizer.encode(self.chunks[i].page_content))
                }
            }
            chunk_dict_list.append(document_dict)

        return chunk_dict_list

    def sanity_check(self)-> None:
        print(f"documents length: {len(self.data)}")
        print(f"Chunks: {len(self.chunks)}")
        print(f"First chunk length: {len(self.tokenizer.encode(self.chunks[0].page_content))}")
        print(f"Pen-ultimate chunk length: {len(self.tokenizer.encode(self.chunks[-2].page_content))}")
        print(f"Ultimate chunk length: {len(self.tokenizer.encode(self.chunks[-1].page_content))}") 

        print(f"Chunk no: 50 , content: {self.chunks[49].page_content}")
        print(f"Chunk no: 100, content: {self.chunks[99].page_content}")
        print(f"Chunk no: 150, content: {self.chunks[149].page_content}")

if __name__=="__main__":
    loader_obj = Loader("papers/") 
    docs = loader_obj.corpus_loader()

    try:
        model = SentenceTransformer("voyageai/voyage-4-nano", trust_remote_code=True, truncate_dim=1024)
        tokenizer = model.tokenizer
    except Exception as e:
        print(e)

    chunker_obj = Chunker(1000, docs, tokenizer)
    chunks = chunker_obj.corpus_chunker()

    chunker_obj.sanity_check()