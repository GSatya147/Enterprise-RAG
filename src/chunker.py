import os

from loader import Loader

class Chunker:
    def __init__(self, chunk_size, loader_docs, model):
        self.CHUNK_SIZE = chunk_size
        self.CHUNK_OVERLAP = int(0.15 * self.CHUNK_SIZE)
        self.model = model



        try:
            self.tokenizer = self.model.tokenizer

            self.tokenized_data = self.tokenizer.encode(data)
        except Exception as e:
            print(e)

    def corpus_chunker(self)-> list[dict]:
        self.chunks = []

        i=0
        while i < len(self.tokenized_data):
            self.chunks.append(self.tokenized_data[i: self.CHUNK_SIZE + i])
            i+=self.CHUNK_SIZE - self.CHUNK_OVERLAP

        document_dict = dict()
        chunk_dict_list = []

        for i in range(len(self.chunks)):
            document_dict = {
                "content": self.tokenizer.decode(self.chunks[i]),
                "metadata": {
                    "source": "manual",
                    "document_name": os.path.basename(self.FILE_PATH),
                    "chunk_index": i,
                    "token_count": len(self.chunks[i])
                }
            }
            chunk_dict_list.append(document_dict)

        return chunk_dict_list

    def sanity_check(self)-> None:
        print(f"Total tokens: {len(self.tokenized_data)}")
        print(f"Chunks: {len(self.chunks)}")
        print(f"First chunk length: {len(self.chunks[0])}")
        print(f"Pen-ultimate chunk length: {len(self.chunks[-2])}")
        print(f"Ultimate chunk length: {len(self.chunks[-1])}") 

if __name__=="__main__":
    pass  