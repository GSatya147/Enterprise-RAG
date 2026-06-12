import os

from dotenv import load_dotenv
import voyageai

load_dotenv()
voyage_key = os.getenv("VOYAGE_API_KEY")

class Reranker:
    def __init__(self, user_query, top_n=3):
        self.user_query = user_query
        self.top_n = top_n

        try:
            self.vo = voyageai.Client(api_key=voyage_key)

        except Exception as e:
            print(e)
    
    def corpus_reranker(self, retriever_results):
        
        try:
            reranker_results = self.vo.rerank(
                    query=self.user_query,
                    documents=[chunk["content"] for chunk in retriever_results],
                    model="rerank-2",
                    top_k=self.top_n
                )
            
            reranker_results_dict = []
            for r in reranker_results.results:
                field = {
                    "id" : r.index,
                    "content" : r.document,
                    "source" : retriever_results[r.index].get("source"),
                    "score" : r.relevance_score,
                }
                reranker_results_dict.extend([field])

            return reranker_results_dict
        
        except Exception as e:
            print(e)
       
# if __name__=="__main__":
#     user_query = input(">> ")

#     retriever_obj = Retriever(user_query=user_query)
#     retriever_results = retriever_obj.corpus_retriever()

#     reranker_obj = Reranker(user_query=user_query)
#     reranker_results = reranker_obj.corpus_reranker(retriever_results=retriever_results)

#     print(reranker_results)