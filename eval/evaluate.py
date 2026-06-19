import asyncio
import json
import os

import litellm

from src.conversation_manager import ConversationManager
from src.generator import ResponseGenerator
from src.reranker import Reranker
from src.retriever import Retriever

class EvaluateRAG:
    def __init__(self):
        self.data_list = []

        try:
            pass # LLm client call
        
        except Exception as e:
            print(e)

    def load_dataset(self):

        with open("./eval/datasets/QA_deepseek_dataset.jsonl", "r") as f:
            self.data_list = [json.loads(content) for content in f.readlines() if content is not None and content != "\n"]
            
        return self.data_list
    
    async def run_pipeline(self, semaphore, question, ground_truth):
        async with semaphore:
            
            retriever_obj = Retriever(user_query=question)
            retriever_results = retriever_obj.corpus_retriever()

            reranker_obj = Reranker(user_query=question)
            reranker_results = reranker_obj.corpus_reranker(retriever_results=retriever_results)
            
            conv = ConversationManager()
            conv.add_context(question, "user") 
            
            generator_response = ""
            generator_obj = ResponseGenerator(context=conv.get_history())
            for chunk in generator_obj.response_generator(reranker_results=reranker_results):
                generator_response += chunk

            return {
                "question": question,
                "answer": generator_response,        
                "contexts": [f'{chunk.get("content")}' for chunk in reranker_results],        
                "ground_truth": ground_truth             
            }

    async def ragas_evaluate(self, semaphore):
        async with semaphore:
            pass

async def main():
    semaphore = asyncio.Semaphore(3) 

    eval_obj = EvaluateRAG()
    eval_obj.load_dataset()

    tasks = [eval_obj.run_pipeline(semaphore, question=content.get("question"), ground_truth = content.get("ground_truth")) for content in eval_obj.data_list]
    results = await asyncio.gather(*tasks)

    print(results)
    print(len(results))
    
asyncio.run(main())