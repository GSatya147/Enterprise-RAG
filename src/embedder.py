import os

import chromadb
from chunker import Chunker
from dotenv import load_dotenv
from loader import Loader
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()
key=os.getenv("PINECONE_API_KEY")
if not key:
    raise EnvironmentError("Environment variable error")

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
    
    def vector_store(self):
        if self.EMBED_AND_STORE:
            self.corpus_embedder()

            data_dict_field: list[dict] = []

            for i,chunk in enumerate(self.chunker_result):
                field = {
                    "id": f"doc_{i}",
                    "values": self.embeddings[i].tolist(),
                    "metadata":{
                        "source": chunk["metadata"].get("source"),
                        "content": chunk.get("content")
                    }
                }
                data_dict_field.append(field)
            
        try:
            pc = Pinecone(api_key=key)
            index_name = "satya-RAGLens"

            if not pc.has_index(index_name):
                pc.create_index(
                    name=index_name,
                    vector_type="dense",
                    dimension=1024,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )

            index = pc.Index(index_name)

            if self.EMBED_AND_STORE:
                index.upsert(vectors=data_dict_field, namespace="RAGLens-references")
        
        except Exception as e:
            print(e)
        
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
    embeddings = embedder_obj.vector_store()

    if embedder_obj.EMBED_AND_STORE:
        print(embedder_obj.embeddings)
        print(len(embedder_obj.embeddings))
