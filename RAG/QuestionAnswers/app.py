import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub, HuggingFacePipeline, PromptTemplate, LLMChain

import streamlit as st
from streamlit_chat import message
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

model_name = "MBZUAI/LaMini-Flan-T5-783M"
persist_directory = "../ChromaDB"

if "prompts" not in st.session_state:
    st.session_state.prompts = []
if "responses" not in st.session_state:
    st.session_state.responses = []


@st.cache_resource
def getChromaDB():
    print("Embeddings")
    embeddings = HuggingFaceEmbeddings()

    print("Initialize PeristedChromaDB")
    chroma = Chroma(
        collection_name="corporate_db",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    return chroma


@st.cache_resource
def getRetrievalQAChain(_chroma):
    print("Initializing Model")
    llm = HuggingFaceHub(
        repo_id=model_name,
        model_kwargs={"temperature": 0, "max_length": 512},
    )

    print("Creating Chain")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_chroma.as_retriever(),
        input_key="question",
    )

    return chain


def send_click(chain):
    if st.session_state.user != "":
        prompt = st.session_state.user
        print(f"Query {prompt}")
        response = chain.run(prompt)

        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(response)


def main(chain):
    st.title(":blue[Corporate Brain] ☕")

    st.text_input("Ask Something:", key="user")
    if st.button("Send"):
        send_click(chain)
    # st.button("Send", on_click=send_click)

    if st.session_state.prompts:
        for i in range(len(st.session_state.responses) - 1, -1, -1):
            message(st.session_state.responses[i], key=str(i), seed="Milo")
            message(
                st.session_state.prompts[i], is_user=True, key=str(i) + "_user", seed=83
            )


if __name__ == "__main__":
    chroma = getChromaDB()
    chain = getRetrievalQAChain(chroma)
    main(chain)
