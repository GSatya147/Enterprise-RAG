import asyncio
import json
import os
import random
import time

from collections import defaultdict
from dotenv import load_dotenv
import litellm
from sentence_transformers import SentenceTransformer

from src.chunker import Chunker
from src.loader import Loader

load_dotenv()

class QAGenerator:
    def __init__(self, chunks):
        self.chunks = chunks
        self.n_sample = 27

    def qa_generator(self, chunk_content):

        system_prompt = '''
            You are a question-answer pair generator for a RAG evaluation dataset.
            Given a single chunk of text from an academic paper, generate exactly one 
            question-answer pair that is:
            - Answerable solely from the provided chunk
            - Specific and factual, not vague or general
            - Requires reading the chunk to answer — not common knowledge

            Return ONLY valid JSON in this exact format, no preamble, no explanation:
            {"question": "...", "ground_truth": "...", "valid": true}

            If the chunk is too short, too generic, or contains no answerable facts, return:
            {"valid": false}

            <context>
            chunk: CONTEXT
            </context>

            <examples>
            Example 1:
            Chunk: "RAGAS evaluates RAG pipelines using four metrics: Faithfulness, 
            Answer Relevance, Context Recall, and Context Precision. Faithfulness measures 
            whether the generated answer is grounded in the retrieved context by decomposing 
            the answer into individual claims and verifying each against the context."

            Output:
            {"question": "How does RAGAS measure Faithfulness in a RAG pipeline?", 
            "ground_truth": "RAGAS measures Faithfulness by decomposing the generated answer into individual claims and verifying each claim against the retrieved context.", 
            "valid": true}

            Example 2:
            Chunk: "The ReAct framework interleaves reasoning traces and actions in an 
            alternating fashion. At each step, the model generates a thought, executes an 
            action against an external tool, and observes the result before proceeding to 
            the next reasoning step."

            Output:
            {"question": "How does the ReAct framework structure its reasoning and action steps?", 
            "ground_truth": "ReAct interleaves reasoning traces and actions alternately — at each step the model generates a thought, executes an action against an external tool, and observes the result before the next step.", 
            "valid": true}

            Example 3:
            Chunk: "Table 3. Results on HotpotQA dataset."

            Output:
            {"valid": false}
            </examples>
        '''

        messages = [{"role" : "user", "content" : system_prompt.replace("CONTEXT", f'"{chunk_content.get("content")}"')}]

        try:
            response = litellm.completion(
                model=os.getenv("DEEPSEEK_MODEL"),
                messages=messages,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                num_retries=3
            )

            return response.choices[0].message.content
        
        except Exception as e:
            print(e)

    def dataset_generator(self):
        # group by source paper
        grouped = defaultdict(list)
        for chunk in self.chunks:
            grouped[chunk["metadata"].get("source")].append(chunk)

        stratified_chunks = []
        for each_source in grouped.keys():
            sample = random.sample(grouped.get(each_source), 3)
            stratified_chunks.extend(sample)
        
        random.shuffle(stratified_chunks)
        for chunk in stratified_chunks:
            qa_pairs = self.qa_generator(chunk)
            time.sleep(3)
            json_string = json.loads(qa_pairs)

            with open("./eval/results/QA_deepseek_dataset.jsonl", "a") as f:
                f.write(json.dumps(json_string) + "\n")

if __name__=="__main__":
    loader_obj = Loader("./papers")
    docs = loader_obj.corpus_loader()

    try:
        model = SentenceTransformer("voyageai/voyage-4-nano", trust_remote_code=True, truncate_dim=1024)
        tokenizer = model.tokenizer
    except Exception as e:
        print(e)

    chunker_obj = Chunker(1000, docs, tokenizer)
    chunks_list = chunker_obj.corpus_chunker()

    qa_obj = QAGenerator(chunks_list)
    qa_obj.dataset_generator()




    