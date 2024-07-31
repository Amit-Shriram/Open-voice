from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage, ChatMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_huggingface import HuggingFaceEndpoint
import os
from langchain_huggingface import ChatHuggingFace
# from transformers import AutoTokenizer
import streamlit as st
from pypdf import PdfReader
from langchain_core.messages import AIMessage, ToolMessage
from typing import Literal
from langgraph.graph import END
from dotenv import load_dotenv
load_dotenv()

## save HUGGINGFACEHUB_API_TOKEN = your_huggingface_token in .env file
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

TOTAL_QUESTIONS = 2
pdf_flag = False

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            if raw_text:
                st.success("Done")
                ans = raw_text
                pdf_flag = True

if "messages" not in st.session_state:
    st.session_state.messages = []
l = len(st.session_state.messages)

## CHECKING NUMBER OF QUESTIONS
if l > (2*(TOTAL_QUESTIONS-1)): # +1 for the candidate's background context as input
    asked = True
else:
    asked = False

if l == 1:
    with st.chat_message(st.session_state.messages[0]["role"]):
        st.markdown(st.session_state.messages[0]["content"])
elif l>1:
    with st.chat_message(st.session_state.messages[0]["role"]):
        st.markdown(st.session_state.messages[0]["content"])
    for i in range(1,l):
        with st.chat_message(st.session_state.messages[i]["role"]):
            st.markdown(st.session_state.messages[i]["content"])

llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="text-generation",
        temperature=0.75,
        verbose=True,
        huggingfacehub_api_token = hf_token,
        streaming=True
    )

template = """Your job is to interview the user based on the background information about the candidate.

Give follow up questions taking the candidate's answers into consideration.

Only output the question to be asked.

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool.
"""

def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages


chat_model = ChatHuggingFace(llm=llm)
chain = get_messages_info | chat_model

## New system prompt
prompt_system = """Based on the following requirements, generate an interview question relevant to the relevant job role. Do not be too specific:

{reqs}"""

## Function to get the messages for the question
# Will only get messages AFTER the tool call
def get_prompt_messages(messages: list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)
    return [ChatMessage(content=prompt_system.format(reqs=tool_call), role="system")] + other_msgs 

prompt_gen_chain = get_prompt_messages | llm


def get_state(messages) -> Literal["add_tool_message","bye", "info", "__end__"]:
    if asked:
        print("bye")
        return "bye"
    elif isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif (not isinstance(messages[-1], ChatMessage)) :
        return END
    return "info"

def end_result(messages: list):
    return ChatMessage(content="Thank you for your time.", role="system")


from langgraph.graph import MessageGraph
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
workflow = MessageGraph()
workflow.add_node("info", chain)
workflow.add_node("prompt", prompt_gen_chain)
workflow.add_node("bye", end_result)


def add_tool_message(state: list):
    return ToolMessage(
        content="Question generated!", tool_call_id=state[-1].tool_calls[0]["id"],
    )

workflow.add_node("add_tool_message", add_tool_message)

workflow.add_conditional_edges("info", get_state)
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)
workflow.add_edge("bye", END)
workflow.set_entry_point("info")
graph = workflow.compile(checkpointer=memory)

import uuid

config = {"configurable": {"thread_id": str(uuid.uuid4())}}
if pdf_flag:
    ans = raw_text
else:
    raw_text = ""
if ans := st.chat_input(placeholder="Answer: ", disabled=asked) or raw_text:
    if not(raw_text):
        with st.chat_message("user"):
            st.markdown(ans)
            st.session_state.messages.append({"role": "user", "content": ans})  

    if asked:
        with st.chat_message("ai"):
            st.markdown("Thank you for your time.")
     
    else:
        output = None
        for output in graph.stream(
            [ChatMessage(content=ans, role='user')], config=config, stream_mode='updates'
        ):
            last_message = next(iter(output.values()))
            if not(asked):
                with st.chat_message("ai"):
                    st.markdown(last_message.content)
                    st.session_state.messages.append({"role": "ai", "content": last_message.content})