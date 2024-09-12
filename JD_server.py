# # in this we modify the server for JD integration

# from fastapi import FastAPI, WebSocket
# from fastapi.responses import StreamingResponse, Response
# import torch
# import numpy as np
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# import wavio
# import os
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain_core.messages import SystemMessage, AIMessage, ChatMessage, ToolMessage
# from langchain_huggingface import ChatHuggingFace
# from typing import Literal
# from langgraph.graph import END
# import uuid
# from pydantic import BaseModel
# from langchain.output_parsers import PydanticOutputParser
# from langchain_core.runnables.graph import MermaidDrawMethod
# from typing import Optional
# import requests
# import json
# from langchain import LLMChain, PromptTemplate
# import re
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM

# load_dotenv()

# llm_url = os.getenv("llama3.1_URL")

# app = FastAPI()

# ## VARIABLES
# TOTAL_QUESTIONS = 10

# # token for pocketbase
# auth_token = None

# model = OllamaLLM(model="llama3.1", temperature=0.75)

# class AskHuman(BaseModel):
#     """Ask the human a question"""
#     question: str

# parser = PydanticOutputParser(pydantic_object=AskHuman)

# # Your job is to interview the candidate based on their resume and the provided job description.
# # Ensure that the questions are relevant to both the candidate's background and the job requirements.

# template = """Your job is to interview the user based on the background information about the candidate and provided job description.

# - Job Description: {job_description}
# - Resume Summary: {resume_summary}

# First, greet the candidate by their name and ask them to introduce themselves.
# After receiving the introduction, ask 3 questions related to their latest project from the resume, considering its relevance to the job description.
# Then, ask 2 questions related to their experience or internship.
# Next, ask technical questions related to the skills mentioned in both the resume and job description, gradually increasing the difficulty level.
# Give follow-up questions taking the candidate's answers into consideration.

# Do not provide feedback, suggestions, or justifications. Your output should consist solely of the next question.
# Ask brief and concise questions.
# Make sure to keep the interview focused but dive for clarification of knowledge when necessary.
# If asked to repeat a question or if the candidate doesn't understand, rephrase the question in a more descriptive or understandable way.

# If you are not able to discern specific information, ask the candidate to clarify. Do not attempt to guess wildly.

# Output the next question to be asked.
# """

# # ###################################################################### To Score #############################################################

# prompt_template2 = """
# Question: {question}
# Candidate's Answer: {answer}

# Evaluate the candidate's answer based on the question.

# The range of score is between 0 and 10, with 0 being the lowest and increasing to 10 which is the highest score.

# For answers like "repeat question," "ok," "I don't know," give the lowest score, which is 0.

# If the answer responded is contextual but incorrect, based on your own judgment give a score between 4 to 7.

# If the answer is contextual and correct based on your own judgement give a score between 8 to 10

# If the answer provided by the candidate is inappropriate, off-topic or irrelevant to the question, give a score based on your own judgement in the range 0 to 3.

# Your response should strictly contain only the score as a single integer
# """


# prompt2 = ChatPromptTemplate.from_template(prompt_template2)

# # And a query intended to prompt a language model to populate the data structure.
# chain2 = prompt2 | model

# # function to evaluate the answer given by the candidate
# def evaluate_answer(question: str, answer: str) -> int:
#     result  = chain2.invoke({"question": question, "answer": answer})
#     print(f"this is response from llm:- {result}")
#     try:
#         score = int(result.strip())
#         print(f"Score: {score}")
#     except ValueError:
#         print("The response did not return a valid integer score.")
#         score = 0
#     print(f"from inside score function:- {score}")
#     return score


# def get_messages_info(messages: list):
#     return [SystemMessage(content=template)] + messages

# chain = get_messages_info | model


# # This code snippet checks whether the last message in the messages list is a ChatMessage. 
# # If it’s not, the function returns "ask_human", indicating that the system needs to ask the human for input again. If it is a ChatMessage, it returns "info", which likely means that the system is processing user-provided information.
# def get_state(messages: list) -> Literal["bye", "ask_human"]:
#     if not isinstance(messages[-1], ChatMessage):
#         return "ask_human"
#     return "info"

# def end_result(messages: list):
#     return "Thank you for your time."

# from langgraph.graph import MessageGraph
# from langgraph.checkpoint.sqlite import SqliteSaver

# # we are telling that use the in memory to store the states of the graph
# memory = SqliteSaver.from_conn_string(":memory:")      # :memory: represents to use in memory
# workflow = MessageGraph()

# def ask_human(state: list):
#     pass

# # Correct way to invoke `chain`
# def ask_human2(state: list):
#     # Get the messages for the chain
#     messages = get_messages_info(state)
    
#     # Invoke the chain with the messages
#     response = chain.invoke(messages)
    
#     # Process the response
#     if isinstance(response, str):
#         return response
#     else:
#         # Handle cases where response is not a string
#         return "Something went wrong."


# # Workflow and graph setup
# workflow.add_node("info", ask_human2)              # info => means ask_human2 => is basically invokes the chain
# workflow.add_node("bye", end_result)
# workflow.add_node("ask_human", ask_human)

# workflow.set_entry_point("info")
# workflow.add_conditional_edges("info", get_state)
# workflow.add_edge("ask_human", "info")
# workflow.add_edge("bye", END)

# # workflow.compile() compiles the message graph(workflow) into a complete flow.
# graph = workflow.compile(checkpointer=memory, interrupt_before=["ask_human"])         #checkpointer=momory, compiled workflow will use the in-memory SQLite database to store checkpoints or save its state. and This specifies that the workflow should be interrupted before reaching a certain node called "ask_human". In other words, the execution of the graph will pause or break before it hits the "ask_human" node.
# config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# # to process the answer and generate the next question
# def graph_stream(ans: str, msg_num: int, msg_info: list, uid: str, name: str, JD:str):
#     msg = None       
#     for output in graph.stream([ChatMessage(content=ans, role='user')], config=config, stream_mode='updates'):         # This part of the code streams responses from the message graph (graph), using the graph.stream() function. The function takes two main inputs:A list of messages, in this case, the user’s answer (ans) wrapped as a ChatMessage object with the role set as 'user'.A configuration (config), which is likely related to how the workflow is set up (e.g., thread_id for tracking sessions).
#         print(f"######################################################Output received: {output}")  # Debugging line
#         last_message = next(iter(output.values()))      #The next(iter(...)) syntax gets the first value from the output dictionary.
#         print(f"####################################################Last message content: {last_message}")  # Debugging line

#         if msg_num >= TOTAL_QUESTIONS * 2:              
#             msg = end_result([])
#             yield msg
#             break

#         if "output" in output.keys():
#             last_message = output["info"]
        
#         # This block of code checks the type of last_message. If it is a string, it directly assigns it to msg. 
#         # If it is an object (e.g., a ChatMessage object), it extracts the content attribute.
#         # The msg variable will now hold the message that the system intends to send as a response.
#         if isinstance(last_message, str):
#             msg = last_message
#         elif hasattr(last_message, 'content'):
#             msg = last_message.content
        
#         print(f"Message assigned: {msg}")  # Debugging line

#         if msg:
#             store_interview_data(uid, name, msg_info)
#             yield msg
#             print(msg_info)


# class AnswerModel(BaseModel):
#     ans: str
#     msg_num: Optional[int] = 0
#     msg_info: list
#     uid: str
#     name: str
#     job_description: Optional[str] = None

# @app.post("/langgraph")
# async def main(answer: AnswerModel):
#     # If job description is provided, update the template
#     # if answer.job_description:
#     #     global template
#     #     template = template.format(
#     #         job_description=answer.job_description,
#     #         resume_summary=get_resume_summary(answer.msg_info)
#     #     )
    
#     return StreamingResponse(graph_stream(answer.ans, answer.msg_num, answer.msg_info, answer.uid, answer.name, answer.job_description), media_type="application/json")

# @app.get("/", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
# async def check_langgraph():
#     image_bytes = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
#     return Response(content=image_bytes, media_type="image/png")

# def get_resume_summary(msg_info: list) -> str:
#     # Extract resume information from the message history
#     resume_summary = ""
#     for message in msg_info:
#         if message['role'] == 'user' and "resume" in message['content'].lower():
#             resume_summary = message['content']
#             break
#     return resume_summary

# # PocketBase integration
# def get_auth_token():
#     global auth_token
#     if auth_token is None:
#         data={
#         "identity": "belallshaikh@gmail.com",
#         "password": "Password1#"
#         }
#         auth_end_point= "https://dev-pocketbase3.huhoka.com/api/admins/auth-with-password"
#         response= requests.post(auth_end_point, json=data)
#         decoded_string = response.content.decode('utf-8')
#         json_data = json.loads(decoded_string)
#         auth_token= json_data["token"]
#     return auth_token

# def store_interview_data(uid: str, name: str, msg_info: list):
#     base_url = 'https://dev-pocketbase3.huhoka.com'
#     collection = 'interview_bot'
#     endpoint = f'{base_url}/api/collections/{collection}/records'
#     score = 0

#     latest_ai_message = None
#     latest_user_message = None

#     for message in reversed(msg_info):
#         if message['role'] == 'ai' and latest_ai_message is None:
#             latest_ai_message = message['content']
#         elif message['role'] == 'user' and latest_user_message is None:
#             latest_user_message = message['content']
        
#         if latest_ai_message and latest_user_message:
#             score = evaluate_answer(latest_ai_message, latest_user_message)
#             # print(f"Score is by another outside score llm:- {score}")
#             break
    
#     data = {
#         'uId': uid,
#         'candidateName': name,
#         'question': latest_ai_message,
#         'answer': latest_user_message,
#         'score': score
#     }

#     headers = {
#         'Authorization': get_auth_token(),
#         'Content-Type': 'application/json'
#     }

#     response = requests.post(endpoint, headers=headers, json=data)

#     if response.status_code == 201:
#         print('Record created:', response.json())
#     else:
#         print('Error:', response.status_code, response.text)

# # WebSocket for real-time transcription
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# model_id = "openai/whisper-base"
# # this is a base model that we are using in here, because of no GPU, when we test it in server then 
# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
# )
# model.to(device)

# processor = AutoProcessor.from_pretrained(model_id)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     max_new_tokens=128,
#     chunk_length_s=30,
#     batch_size=16,
#     return_timestamps=True,
#     torch_dtype=torch_dtype,
#     device=device,
# )

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("WebSocket connection established.")
#     transcription_file = open("transcription.txt", "a+")
#     chunk_counter = 0

#     try:
#         while True:
#             data = await websocket.receive_bytes()

#             if data == b"STOP":
#                 print("Received stop signal from client.")
#                 break

#             chunk_counter += 1
#             print(f"Received audio chunk {chunk_counter} from client.")
#             audio_filename = "recorded_audio.wav"

#             # Save audio chunk
#             audio_data = np.frombuffer(data, dtype=np.int16)
#             wavio.write(audio_filename, audio_data, rate=16000, sampwidth=2)
#             print("audio saved in .wav file")

#             # Process audio for transcription
#             audio_data_float = audio_data.astype(np.float32) / 32768.0
#             result = pipe(audio_data_float, return_timestamps=True, generate_kwargs={"language": "english"})
#             transcription_text = result["text"]
#             print(f"Transcription for chunk {chunk_counter}: {transcription_text}")

#             transcription_file.writelines(transcription_text)
#             print(f"Transcription for chunk {chunk_counter} saved to file.")

#             # Send transcription back to client
#             await websocket.send_text(transcription_text)
#             print(f"Transcription for chunk {chunk_counter} sent back to client.")
#     except Exception as e:
#         print(f"Connection closed: {e}")
#     finally:
#         transcription_file.close()
#         print("Transcription file closed.")




from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, Response
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import wavio
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, AIMessage, ChatMessage, ToolMessage
from langchain_huggingface import ChatHuggingFace
from typing import Literal
from langgraph.graph import END
import uuid
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables.graph import MermaidDrawMethod
from typing import Optional
import requests
import json
from langchain import LLMChain, PromptTemplate
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

load_dotenv()

llm_url = os.getenv("llama3.1_URL")

app = FastAPI()

## VARIABLES
TOTAL_QUESTIONS = 7

# token for pocketbase
auth_token = None

model = OllamaLLM(model="llama3.1", temperature=0.75)

class AskHuman(BaseModel):
    """Ask the human a question"""
    question: str

parser = PydanticOutputParser(pydantic_object=AskHuman)

# Ensure that the questions are relevant to both the candidate's background and the job requirements.

# template = """Your job is to interview the user based on the background information about the candidate and job description.

# job description: {job_description}

# First greet the candidate by their name and ask them to introduce themselves.
# After receiving the introduction answer from the candidate, next ask 3 questions related to their latest project from the resume.
# After that, from resume ask 2 questions related to their experience or internship .
# Next, ask difficult questions related to the technical skills mentioned in the resume, gradually increasing the difficulty level of the questions.
# Give follow-up questions taking the candidate's answers into consideration.
# Do not provide feedback, suggestions, or justifications. Your output should consist solely of the next question.
# Don't ask lengthy questions  just ask it in short and brief
# Make sure to make interview breif but dive for clarification of knowledge.
# If you have been asked to repeat the question or they did not understand, then repeat the question in a descriptive or understandable way.
 
 
# if last answer by candidate is repeat the question then repeat the question in a descriptive or understandable way
 
# If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.
 
# Output the next question to be asked.
 

# """

template = """Your task is to interview the candidate based on their background information and Job description provided. Asking only 7 questions in total.

job description : 
{job_description}

Make sure the questions are short and straightforward, without any lengthy explanations. 
Do not provide feedback, suggestions, or justifications. 
Only output the next question based on the candidate's responses.
 
1. Greet the candidate by their name and ask them to introduce themselves.
 
2. Ask the candidate to explain one of their projects briefly.
 
3. Ask the technical questions based on their resume and in context of job description.
 
4. Ask a situational-based question relevant to the candidate's role and job description.
 
5. Ask if the candidate has any questions about Centralogic. Use the following information to give the response:
   Centralogic was founded in 2010 by Mr. Ajay Navgale, Founder, Director & CEO, and Mr. Sanjay Navgale, Founder & Director. Its services include Project Management, DevOps, Cloud Migration, Cyber Security, and others. Centralogic offers a salary of 3.45 LPA for freshers, with a 2-year bond requirement.
   Casual Leaves:
    6 casual leaves will be allocated to you right at the start of the year.
    You can only take a maximum of one leave per month.
    Any unused casual leaves won't carry over to the next year.
    You need to submit your application for casual leave through the HuHoKa portal, if 
    your leave is not approved on the portal, it will be considered as an unpaid leave.
    Special leaves (Birthdays, Anniversary) fall within this category.
 
6. Finally, end the interview by saying "Thank you"

"""
# ###################################################################### To Score #############################################################

prompt_template2 = """
Question: {question}
Candidate's Answer: {answer}

Evaluate the candidate's answer based on the question.

The range of score is between 0 and 10, with 0 being the lowest and increasing to 10 which is the highest score.

For answers like "repeat question," "ok," "I don't know," give the lowest score, which is 0.

If the answer responded is contextual but incorrect, based on your own judgment give a score between 4 to 7.

If the answer is contextual and correct based on your own judgement give a score between 8 to 10

If the answer provided by the candidate is inappropriate, off-topic or irrelevant to the question, give a score based on your own judgement in the range 0 to 3.

Your response should strictly contain only the score as a single integer
"""


prompt2 = ChatPromptTemplate.from_template(prompt_template2)

# And a query intended to prompt a language model to populate the data structure.
chain2 = prompt2 | model

# function to evaluate the answer given by the candidate
def evaluate_answer(question: str, answer: str) -> int:
    result  = chain2.invoke({"question": question, "answer": answer})
    print(f"this is response from llm:- {result}")
    try:
        score = int(result.strip())
        print(f"Score: {score}")
    except ValueError:
        print("The response did not return a valid integer score.")
        score = 0
    print(f"from inside score function:- {score}")
    return score


def get_messages_info(messages: list):
    return [SystemMessage(content=template)] + messages

chain = get_messages_info | model


# This code snippet checks whether the last message in the messages list is a ChatMessage. 
# If it’s not, the function returns "ask_human", indicating that the system needs to ask the human for input again. If it is a ChatMessage, it returns "info", which likely means that the system is processing user-provided information.
def get_state(messages: list) -> Literal["bye", "ask_human"]:
    if not isinstance(messages[-1], ChatMessage):
        return "ask_human"
    return "info"

def end_result(messages: list):
    return "Thank you for your time."

from langgraph.graph import MessageGraph
from langgraph.checkpoint.sqlite import SqliteSaver

# we are telling that use the in memory to store the states of the graph
memory = SqliteSaver.from_conn_string(":memory:")      # :memory: represents to use in memory
workflow = MessageGraph()

def ask_human(state: list):
    pass

# Correct way to invoke `chain`
def ask_human2(state: list):
    # Get the messages for the chain
    messages = get_messages_info(state)
    
    # Invoke the chain with the messages
    response = chain.invoke(messages)
    
    # Process the response
    if isinstance(response, str):
        return response
    else:
        # Handle cases where response is not a string
        return "Something went wrong."


# Workflow and graph setup
workflow.add_node("info", ask_human2)              # info => means ask_human2 => is basically invokes the chain
workflow.add_node("bye", end_result)
workflow.add_node("ask_human", ask_human)

workflow.set_entry_point("info")
workflow.add_conditional_edges("info", get_state)
workflow.add_edge("ask_human", "info")
workflow.add_edge("bye", END)

# workflow.compile() compiles the message graph(workflow) into a complete flow.
graph = workflow.compile(checkpointer=memory, interrupt_before=["ask_human"])         #checkpointer=momory, compiled workflow will use the in-memory SQLite database to store checkpoints or save its state. and This specifies that the workflow should be interrupted before reaching a certain node called "ask_human". In other words, the execution of the graph will pause or break before it hits the "ask_human" node.
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# to process the answer and generate the next question
def graph_stream(ans: str, msg_num: int, msg_info: list, uid: str, name: str):
    msg = None       
    for output in graph.stream([ChatMessage(content=ans, role='user')], config=config, stream_mode='updates'):         # This part of the code streams responses from the message graph (graph), using the graph.stream() function. The function takes two main inputs:A list of messages, in this case, the user’s answer (ans) wrapped as a ChatMessage object with the role set as 'user'.A configuration (config), which is likely related to how the workflow is set up (e.g., thread_id for tracking sessions).
        print(f"######################################################Output received: {output}")  # Debugging line
        last_message = next(iter(output.values()))      #The next(iter(...)) syntax gets the first value from the output dictionary.
        print(f"####################################################Last message content: {last_message}")  # Debugging line

        if msg_num >= TOTAL_QUESTIONS * 2:              
            msg = end_result([])
            yield msg
            break

        if "output" in output.keys():
            last_message = output["info"]
        
        # This block of code checks the type of last_message. If it is a string, it directly assigns it to msg. 
        # If it is an object (e.g., a ChatMessage object), it extracts the content attribute.
        # The msg variable will now hold the message that the system intends to send as a response.
        if isinstance(last_message, str):
            msg = last_message
        elif hasattr(last_message, 'content'):
            msg = last_message.content
        
        print(f"Message assigned: {msg}")  # Debugging line

        if msg:
            store_interview_data(uid, name, msg_info)
            yield msg
            print(msg_info)


class AnswerModel(BaseModel):
    ans: str
    msg_num: Optional[int] = 0
    msg_info: list
    uid: str
    name: str
    job_description: Optional[str] = None

@app.post("/langgraph")
async def main(answer: AnswerModel):
    # If job description is provided, update the template
    if answer.job_description:
        global template
        template = template.format(
            job_description=answer.job_description,
        )
    
    return StreamingResponse(graph_stream(answer.ans, answer.msg_num, answer.msg_info, answer.uid, answer.name), media_type="application/json")

@app.get("/", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def check_langgraph():
    image_bytes = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    return Response(content=image_bytes, media_type="image/png")

# def get_resume_summary(msg_info: list) -> str:
#     # Extract resume information from the message history
#     resume_summary = ""
#     for message in msg_info:
#         if message['role'] == 'user' and "resume" in message['content'].lower():
#             resume_summary = message['content']
#             break
#     return resume_summary

# PocketBase integration
def get_auth_token():
    global auth_token
    if auth_token is None:
        data={
        "identity": "belallshaikh@gmail.com",
        "password": "Password1#"
        }
        auth_end_point= "https://dev-pocketbase3.huhoka.com/api/admins/auth-with-password"
        response= requests.post(auth_end_point, json=data)
        decoded_string = response.content.decode('utf-8')
        json_data = json.loads(decoded_string)
        auth_token= json_data["token"]
    return auth_token

def store_interview_data(uid: str, name: str, msg_info: list):
    base_url = 'https://dev-pocketbase3.huhoka.com'
    collection = 'interview_bot'
    endpoint = f'{base_url}/api/collections/{collection}/records'
    score = 0

    latest_ai_message = None
    latest_user_message = None

    for message in reversed(msg_info):
        if message['role'] == 'ai' and latest_ai_message is None:
            latest_ai_message = message['content']
        elif message['role'] == 'user' and latest_user_message is None:
            latest_user_message = message['content']
        
        if latest_ai_message and latest_user_message:
            score = evaluate_answer(latest_ai_message, latest_user_message)
            # print(f"Score is by another outside score llm:- {score}")
            break
    
    data = {
        'uId': uid,
        'candidateName': name,
        'question': latest_ai_message,
        'answer': latest_user_message,
        'score': score
    }

    headers = {
        'Authorization': get_auth_token(),
        'Content-Type': 'application/json'
    }

    response = requests.post(endpoint, headers=headers, json=data)

    if response.status_code == 201:
        print('Record created:', response.json())
    else:
        print('Error:', response.status_code, response.text)

# WebSocket for real-time transcription
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-base"
# this is a base model that we are using in here, because of no GPU, when we test it in server then 
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")
    transcription_file = open("transcription.txt", "a+")
    chunk_counter = 0

    try:
        while True:
            data = await websocket.receive_bytes()

            if data == b"STOP":
                print("Received stop signal from client.")
                break

            chunk_counter += 1
            print(f"Received audio chunk {chunk_counter} from client.")
            audio_filename = "recorded_audio.wav"

            # Save audio chunk
            audio_data = np.frombuffer(data, dtype=np.int16)
            wavio.write(audio_filename, audio_data, rate=16000, sampwidth=2)
            print("audio saved in .wav file")

            # Process audio for transcription
            audio_data_float = audio_data.astype(np.float32) / 32768.0
            result = pipe(audio_data_float, return_timestamps=True, generate_kwargs={"language": "english"})
            transcription_text = result["text"]
            print(f"Transcription for chunk {chunk_counter}: {transcription_text}")

            transcription_file.writelines(transcription_text)
            print(f"Transcription for chunk {chunk_counter} saved to file.")

            # Send transcription back to client
            await websocket.send_text(transcription_text)
            print(f"Transcription for chunk {chunk_counter} sent back to client.")
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        transcription_file.close()
        print("Transcription file closed.")

