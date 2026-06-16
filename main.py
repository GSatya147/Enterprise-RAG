import os
import streamlit as st

from src.conversation_manager import ConversationManager
from src.generator import ResponseGenerator
from src.reranker import Reranker
from src.retriever import Retriever

if 'CONTEXT' not in st.session_state:
    st.session_state.CONTEXT = ConversationManager()
if 'sources_list' not in st.session_state:
    st.session_state.sources_list = []

CONTEXT = st.session_state.CONTEXT
sources_list = st.session_state.sources_list

# sidebar
with st.sidebar:
    st.subheader("sources")
    for source in st.session_state.sources_list:
        st.write(os.path.basename(source))

# render chat history
history = st.session_state.CONTEXT.get_history()
for message in history:
    with st.chat_message(message["role"]):
        st.write(message["parts"][0]["text"])

if prompt := st.chat_input("Type here"):
    try:
        with st.chat_message("user"):
            st.write(prompt)

        CONTEXT.add_context(prompt, "user")

        retriever_obj = Retriever(user_query=prompt)
        retriever_results = retriever_obj.mmr_retriever()

        reranker_obj = Reranker(user_query=prompt)
        reranker_results = reranker_obj.corpus_reranker(retriever_results=retriever_results)

        st.session_state.sources_list = [r.get("source") for r in reranker_results]

        generator_obj = ResponseGenerator(CONTEXT.get_history())
        with st.chat_message("assistant"):
            assistant_response_string = st.write_stream(generator_obj.response_generator(reranker_results=reranker_results))

        CONTEXT.add_context(assistant_response_string, "model")

    except Exception as e:
        print(e)

