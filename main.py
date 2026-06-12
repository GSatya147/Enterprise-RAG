import os
import streamlit as st

from src.generator import ResponseGenerator
from src.reranker import Reranker
from src.retriever import Retriever

if 'CONTEXT' not in st.session_state:
    st.session_state.CONTEXT = []
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
for message in st.session_state.CONTEXT:
    with st.chat_message(message["role"]):
        st.write(message["parts"])

if prompt := st.chat_input("Type here"):
    try:
        with st.chat_message("user"):
            st.write(prompt)

        CONTEXT += [{"role": "user", "parts": prompt}]

        retriever_obj = Retriever(user_query=prompt)
        retriever_results = retriever_obj.corpus_retriever()

        reranker_obj = Reranker(user_query=prompt)
        reranker_results = reranker_obj.corpus_reranker(retriever_results=retriever_results)

        st.session_state.sources_list = [r.get("source") for r in reranker_results]

        generator_obj = ResponseGenerator()
        with st.chat_message("assistant"):
            assistant_response_string = st.write_stream(generator_obj.response_generator(user_query=prompt, reranker_results=reranker_results))

        CONTEXT += [{"role": "model", "parts": assistant_response_string}]

    except Exception as e:
        print(e)

