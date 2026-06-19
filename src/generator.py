import os

from dotenv import load_dotenv
from litellm import completion, RateLimitError, APIError, ServiceUnavailableError, Timeout, AuthenticationError, BadRequestError

load_dotenv()

class ResponseGenerator:
    def __init__(self, context=None):
        self.context_history = context

    def response_generator(self, reranker_results):
        context = "\n\n".join(
            [f"content: {reranker_results[i].get("content")}, id : {reranker_results[i].get("id")}, source : {reranker_results[i].get("source")}"
             for i in range(len(reranker_results))]
        )

        self.sys_prompt = f"""
            you are a RAG, Agentic AI knowledge specialist, answer only on the basis of provided context:
            <context>{context}</context>
            if the given context is inadequate, just answer "It is out of the provided knowledge" ONLY.
        """

        try:
            response_stream = completion(
                model=os.getenv("DEEPSEEK_MODEL"),
                messages=self.context_history,
                stream=True,
                num_retries=3,
            )

            for chunk in response_stream:
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    yield delta

        except RateLimitError as e:
            print(f"Rate limit error: {e.message}")

        except APIError as e:
            print(f"API error: {e.message} ")

        except BadRequestError as e:
            print(f"Bad request error: {e.message} ")

        except Timeout as e:
            print(f"Time out error: {e.message} ")
        
        except AuthenticationError as e:
            print(f"Auth error: {e.message} ")

        except ServiceUnavailableError as e:
            print(f"Service unavailable error: {e.message} ")

        except Exception as e:
            print(f"Unexpected error: {e}")

# if __name__=="__main__":
#     user_query = input(">> ")

#     retriever_obj = Retriever(user_query=user_query)
#     retriever_results = retriever_obj.corpus_retriever()

#     reranker_obj = Reranker(user_query=user_query)
#     reranker_results = reranker_obj.corpus_reranker(retriever_results=retriever_results)

#     conv = ConversationManager()
#     conv.add_context(user_query, "user") 

#     generator_obj = ResponseGenerator(context=conv.get_history())
#     for chunk in generator_obj.response_generator(reranker_results=reranker_results):
#         print(chunk, end=" ")
