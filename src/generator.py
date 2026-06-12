import os
from google import genai
from google.genai import types, errors
from dotenv import load_dotenv

load_dotenv()

class ResponseGenerator:
    def __init__(self):
        try:
            self.client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY"),
                http_options=types.HttpOptions(
                    timeout=60000,
                    retry_options=types.HttpRetryOptions(
                        attempts=3,  # Maximum 3 attempts (including original)
                        initial_delay=1.0,  # 1 second initial delay
                        max_delay=30.0,  # Maximum 60 seconds between retries
                    ),
                ),
            )

        except Exception as e:
            print(e)

    def response_generator(self, user_query, reranker_results):
        context = "\n\n".join(
            [f"content: {reranker_results[i].get("content")}, id : {reranker_results[i].get("id")}, source : {reranker_results[i].get("source")}"
             for i in range(len(reranker_results))]
        )

        sys_prompt = f"""
            you are a RAG, Agentic AI knowledge specialist, answer only on the basis of provided context:
            <context>{context}</context>
            if the given context is inadequate, just answer "It is out of the provided knowledge" ONLY.
        """

        try:
            response_stream = self.client.models.generate_content_stream(
                model=os.getenv("GEMINI_MODEL"),
                contents=user_query,
                config=types.GenerateContentConfig(system_instruction=sys_prompt),
            )

            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except errors.ClientError as e:  # explore other codes like 400, 401, 404
            if e.code == 429:
                print(f"Rate limited error: {e.message}")
            else:
                print(f"Client error: {e.message}")

        except errors.ServerError as e:
            print(f"Server error: {e.message} ")

        except Exception as e:
            print(f"Unexpected error: {e}")

# if __name__=="__main__":
#     user_query = input(">> ")

#     retriever_obj = Retriever(user_query=user_query)
#     retriever_results = retriever_obj.corpus_retriever()

#     reranker_obj = Reranker(user_query=user_query)
#     reranker_results = reranker_obj.corpus_reranker(retriever_results=retriever_results)

#     generator_obj = ResponseGenerator()
#     for chunk in generator_obj.response_generator(user_query=user_query, reranker_results=reranker_results):
#         print(chunk, end=" ")