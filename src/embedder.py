import chromadb
from chunker import Chunker
from loader import Loader
from sentence_transformers import SentenceTransformer

class VectorEmbedder:
    def __init__(self, chunk_docs, model):
        self.EMBED_AND_STORE = True
        self.chunker_result = chunk_docs
        self.model = model

    def corpus_embedder(self):
        try:
            self.embeddings = []
            running_tokens = 0
            current_batch = []
            for chunk in self.chunker_result:
                if running_tokens + chunk["metadata"]["token_count"] > 5000: 
                    self.embeddings.extend(self.model.encode(current_batch))

                    running_tokens = 0
                    current_batch = []
                
                running_tokens+=chunk["metadata"]["token_count"]
                current_batch.append(chunk["content"])
            
            if current_batch:
                self.embeddings.extend(self.model.encode(current_batch))

        except Exception as e:
            print(e)
        
        return self.embeddings
    
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

    embedder_obj = VectorEmbedder(chunks, model=model)
    if embedder_obj.EMBED_AND_STORE:
        embeddings = embedder_obj.corpus_embedder()
        print(embeddings)
