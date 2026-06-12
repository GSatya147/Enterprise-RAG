import os

from chunker import Chunker
from dotenv import load_dotenv
from loader import Loader
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from sentence_transformers import SentenceTransformer
import voyageai

load_dotenv()
pinecone_key=os.getenv("PINECONE_API_KEY")
voyage_key=os.getenv("VOYAGE_API_KEY")

class VectorEmbedder:
    def __init__(self, chunk_docs):
        self.EMBED_AND_STORE = True
        self.chunker_result = chunk_docs

        try:
            self.vo_client = voyageai.Client(api_key=voyage_key)
        except Exception as e:
            print(e)

    def corpus_embedder(self):
        try:
            self.embeddings = []
            running_tokens = 0
            current_batch = []
            for i, chunk in enumerate(self.chunker_result):
                if running_tokens + chunk["metadata"]["token_count"] > 5000: 
                    self.embeddings.extend(self.vo_client.embed(texts=current_batch, model=os.getenv("EMBEDDING_MODEL")).embeddings)
                    print(f"Batch: {i+1}, done")
                    running_tokens = 0
                    current_batch = []
                
                running_tokens+=chunk["metadata"]["token_count"]
                current_batch.append(chunk["content"])
            
            if current_batch:
                self.embeddings.extend(self.vo_client.embed(texts=current_batch, model=os.getenv("EMBEDDING_MODEL")).embeddings)

            return self.embeddings
        
        except Exception as e:
            print(e)
        
    def vector_store(self):
        if self.EMBED_AND_STORE:
            self.corpus_embedder()

            data_dict_field: list[dict] = []

            for i,chunk in enumerate(self.chunker_result):
                field = {
                    "id": f"doc_{i}",
                    "values": self.embeddings[i],
                    "metadata":{
                        "source": chunk["metadata"].get("source"),
                        "content": chunk.get("content")
                    }
                }
                data_dict_field.append(field)
            
            try:
                pc = Pinecone(api_key=pinecone_key)
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

    embedder_obj = VectorEmbedder(chunks)
    embeddings = embedder_obj.vector_store()

    if embedder_obj.EMBED_AND_STORE:
        print(embedder_obj.embeddings)
        print(len(embedder_obj.embeddings))
