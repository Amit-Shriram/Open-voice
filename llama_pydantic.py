from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage, ChatMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_huggingface import HuggingFaceEndpoint
import os
from langchain_huggingface import ChatHuggingFace
from transformers import AutoTokenizer
import streamlit as st
from pypdf import PdfReader
from langchain_core.messages import AIMessage, ToolMessage
from typing import Literal
from langgraph.graph import END
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
load_dotenv()

## save HUGGINGFACEHUB_API_TOKEN = your_huggingface_token in .env file
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

TOTAL_QUESTIONS = 2

class Response(BaseModel):
    """question being asked and response being ranked"""
    description: str = Field(...,
                         description="question to ask the candidate during the interview"
                    )
    # role: str = Field(description="role of the speaker")
    rate: int = Field(
        ...,
        description="number between 1 to 10 to rate the previous answer of the candidate"
    )

    
parser = PydanticOutputParser(pydantic_object=Response)

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

if "messages" not in st.session_state:
    st.session_state.messages = []
l = len(st.session_state.messages)

## CHECKING NUMBER OF QUESTIONS
if l > (1 + 2*(TOTAL_QUESTIONS-1)): # +1 for the candidate's background context as input
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

Rate the candidate's previous answer on a scale of 0 to 10.

Only provide the question to be asked in json format. 

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool.

Give the answer without the description of the input.

Please output your response in the demanded json format:
{format_instructions}

"""
prompt_template = ChatPromptTemplate.from_template(template)
# messages = p.format_messages(format_instructions=parser.get_format_instructions())

def get_messages_info(messages):
    return [SystemMessage(content=prompt_template.format(format_instructions=parser.get_format_instructions()))] + messages


# chat_model = ChatHuggingFace(llm=llm)
chain = get_messages_info | llm | parser

## New system prompt
prompt_system = """Based on the following requirements, generate an interview question and Please the output of your response in the demanded json format::

{reqs}
{format_instructions}

"""
p = ChatPromptTemplate.from_template(prompt_system)
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
    return [ChatMessage(content=p.format(
        reqs=tool_call,
        format_instructions=parser.get_format_instructions()
        ),
        role="system")] + other_msgs 

prompt_gen_chain = get_prompt_messages | llm | parser


def get_state(messages) -> Literal["add_tool_message","bye", "info", "__end__"]:
    print(messages)
    if asked:
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
        content="Question generated!", tool_call_id=state[-1].tool_calls[0]["id"]
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
if ans := st.chat_input(placeholder="Answer: ", disabled=asked):
    with st.chat_message("user"):
        st.markdown(ans)
        st.session_state.messages.append({"role": "user", "content": ans})  

    if asked:
        with st.chat_message("ai"):
            st.markdown("Thank you for your time.")
     
    else:
        output = ""
        try:
            for output in graph.stream(
                [ChatMessage(content=ans, role='user')], config=config, stream_mode='updates'
            ):
                print(output)
                last_message = next(iter(output.values()))
                print("LAST",last_message)
                with st.chat_message("ai"):
                    st.markdown(last_message.description)
                    st.session_state.messages.append({"role": "ai", "content": last_message.description})
        except Exception as e:
            last_message = str(e)
            if not last_message.startswith("OutputParserException:"):
                print("error")
                raise e
            last_message = last_message.removeprefix("OutputParserException: Failed to parse Response from completion [{\"description\": \"").removesuffix("\", \"rate\": 0}]. Got: 1 validation error for Response")
            with st.chat_message("ai"):
                st.markdown(last_message)