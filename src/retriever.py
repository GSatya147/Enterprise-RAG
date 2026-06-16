import os, math

from dotenv import load_dotenv
import numpy as np
from pinecone.grpc import PineconeGRPC as Pinecone
import voyageai

load_dotenv()
voyage_key = os.getenv("VOYAGE_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

class Retriever:
    def __init__(self, user_query):
        try:
            self.vo = voyageai.Client(api_key=voyage_key)
            self.query_embeddings = self.vo.embed(texts=[user_query], model=os.getenv("EMBEDDING_MODEL")).embeddings
        
        except Exception as e:
            print(e)

    def corpus_retriever(self, n_results=20, filter=None):
        try:
            pc = Pinecone(api_key=pinecone_key)
            index_name = "satya-raglens"

            index = pc.Index(index_name)

            query_qwargs = {
                "namespace" : "raglens-references",
                "vector" : self.query_embeddings[0], 
                "top_k" : n_results,
                "include_metadata" : True,
                "include_values" : False
            }

            if filter is not None:
                query_qwargs["filter"] = filter
            
            query_results = index.query(**query_qwargs)

            self.retriever_results_dict = []
            for match in query_results.matches:
                field = {
                    "id" : match.id,
                    "content" : match.metadata.get("content"),
                    "source" : match.metadata.get("source"),
                }
                self.retriever_results_dict.extend([field])
                
            return self.retriever_results_dict

        except Exception as e:
            print(e)
    
    @staticmethod
    def cosine_similarity(vector_a, vector_b):
        dot_product = np.dot(vector_a, vector_b)
        mag_prod = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

        return dot_product / mag_prod

    def mmr_retriever(self, top_k=5, lam = 0.5):
        self.corpus_retriever()
        embeddings = self.vo.embed(texts=[content.get("content") for content in self.retriever_results_dict], model=os.getenv("EMBEDDING_MODEL")).embeddings
        candidates = list(zip(self.retriever_results_dict, embeddings))

        selected = []
        selected_embeddings = []
        while len(selected) < top_k:
            best_embedding = None
            best_chunk = None
            best_score = -math.inf
            
            for chunk, embedding in candidates:
                relevance = self.cosine_similarity(self.query_embeddings[0], embedding)
                
                if selected:
                    diversity_penalty = max(self.cosine_similarity(embedding, s) for s in selected_embeddings)
                else:
                    diversity_penalty = 0
                
                score = lam * relevance - (1 - lam) * diversity_penalty 

                if score > best_score:
                    best_score = score
                    best_chunk = chunk
                    best_embedding = embedding
            
            selected.append(best_chunk)
            selected_embeddings.append(best_embedding)
            candidates.remove((best_chunk, best_embedding))

        return selected
                


# if __name__=="__main__":
#     user_query = input(">> ")

#     retriever_obj = Retriever(user_query=user_query)
#     retriever_results = retriever_obj.corpus_retriever()

#     print(retriever_results)
