import asyncio
import json
import os

from dotenv import load_dotenv
import litellm
import sys
from unittest.mock import MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
from ragas.llms import llm_factory
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy
from ragas import evaluate
from ragas import EvaluationDataset

from src.conversation_manager import ConversationManager
from src.generator import ResponseGenerator
from src.reranker import Reranker
from src.retriever import Retriever

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"

judge_llm = llm_factory(model="deepseek-chat")
class EvaluateRAG:
    def __init__(self):
        self.data_list = []

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

            field = {
                "question": question,
                "generated_answer": generator_response,        
                "retrieved_context": [f'{chunk.get("content")}' for chunk in reranker_results],        
                "ground_truth": ground_truth             
            }

            with open("./eval/results/pipeline_results_dict.jsonl", "a") as af:
                af.write(json.dumps(field) + "\n")

            return field

    def ragas_evaluate(self, pipeline_dict):
        judge_llm = llm_factory(model="deepseek-v4-flash")

        metrics = [
                ContextPrecision(llm=judge_llm),
                ContextRecall(llm=judge_llm),
                Faithfulness(llm=judge_llm),
                AnswerRelevancy(llm=judge_llm),
        ]

        mapped_list = [
                {
                    "user_input": row.get("question"),
                    "response": row.get("answer"),
                    "retrieved_contexts": [
                        chunk.encode("utf-8", errors="ignore").decode("utf-8") 
                        for chunk in row.get("contexts")
                    ],
                    "reference": row.get("ground_truth")
                }
                for row in pipeline_dict
        ]

        ragas_dataset = EvaluationDataset.from_list(mapped_list)
        results = evaluate(ragas_dataset, metrics=metrics)

        results_dict = results.to_pandas().to_dict(orient="records")

        with open("./eval/results/ragas_metrics.jsonl", "a") as af:
            json.dump(results_dict, af, indent=2)
            af.write(json.dumps('\n'))
            
        return results

async def main():
    # semaphore = asyncio.Semaphore(3) 

    eval_obj = EvaluateRAG()
    eval_obj.load_dataset()

    # tasks = [eval_obj.run_pipeline(semaphore, question=content.get("question"), ground_truth = content.get("ground_truth")) for content in eval_obj.data_list]
    # results = await asyncio.gather(*tasks)

    with open("./eval/results/pipeline_results_dict.jsonl", "r") as rf:
        pipeline_dict = [json.loads(line) for line in rf.readlines()]

    results = eval_obj.ragas_evaluate(pipeline_dict=pipeline_dict)
    print(results)
    
asyncio.run(main())