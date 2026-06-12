import os

from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
import voyageai

load_dotenv()
voyage_key = os.getenv("VOYAGE_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

class Retriever:
    def __init__(self, user_query):
        try:
            vo = voyageai.Client(api_key=voyage_key)
            self.query_embeddings = vo.embed(texts=[user_query], model=os.getenv("EMBEDDING_MODEL")).embeddings
        
        except Exception as e:
            print(e)

    def corpus_retriever(self, n_results=10, filter=None):
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

# if __name__=="__main__":
#     user_query = input(">> ")

#     retriever_obj = Retriever(user_query=user_query)
#     retriever_results = retriever_obj.corpus_retriever()

#     print(retriever_results)
