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

load_dotenv()

app = FastAPI()

## VARIABLES
TOTAL_QUESTIONS = 10

# token for pocketbase
auth_token = None

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0.75,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    streaming=True
)

class AskHuman(BaseModel):
    """Ask the human a question"""
    question: str

parser = PydanticOutputParser(pydantic_object=AskHuman)

# template = """
# You are a professional technical interviewer for a software engineering company.
# Your job is to interview the candidate based on the background information.
# First greet the candidate by their name and ask them to introduce themselves.
# Give follow-up questions taking the candidate's answers into consideration.
# Only output the question to be asked.

# If you have been asked to repeat the question or they did not understand, then repeat the question in descriptive way or understandable way.
# If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

# Explore all the professional experience of the candidate including their skills, work experience and any projects they have worked on.
# Ensure to keep the interview brief in the sense that a shallow dive is made on each topic of questions.
# After receiving the answer from the candidate, ask a maximum two follow-up questions related to it. After that, take
# a different topic from the resume to ask the next series of questions,
# Capability of a candidate can be measured by the correctness of his answers and the performance of question answering.
# based on the capability of the candidate, gradually increasing the difficulty level of the questions.

# Output the next question to be asked.
# """


template = """Your job is to interview the user based on the background information about the candidate.
First greet the candidate by their name and ask them to introduce themselves.
After receiving the introduction answer from the candidate, next ask 3 questions related to their latest project from the resume.
After that, from resume ask 2 questions related to their experience or internship .
Next, ask difficult questions related to the technical skills mentioned in the resume, gradually increasing the difficulty level of the questions.
Give follow-up questions taking the candidate's answers into consideration.
Only output the question to be asked.
Make sure to make interview breif but dive for clarification of knowledge.
If you have been asked to repeat the question or they did not understand, then repeat the question in a descriptive or understandable way.
 
 
if last answer by candidate is repeat the question then repeat the question in a descriptive or understandable way
 
If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.
 
Output the next question to be asked.
 

"""


# ###################################################################### To Score #############################################################
# prompt_template = """
# Evaluate the following answer based on relevance, clarity, and completeness together. Provide a integer score from 1 to 10 and nothing else.
#  Give the evaluation score in first line of response.
# for example Candidate_Score:8
# Provide an integer score from 1 to 10, with 1 being the lowest and 10 being the highest, based on the overall performance
# in all three areas combined.
 
# for answers like repeat question , ok give lowest score, give Candidate_Score: 0
 
 
#  If the answer or suggestion provided by the candidate is inappropriate, off-topic, or irrelevant, give Candidate_Score: 0.
#  Question: {question}
#  Answer: {answer}
 
# """
 
prompt_template = """
Evaluate the following answer based on relevance, clarity, and completeness together. Provide a integer score from 1 to 10 and nothing else.
 Give the evaluation score in first line of response.
for example Candidate_Score:8
Provide an integer score from 1 to 10, with 1 being the lowest and 10 being the highest, based on the overall performance
in all three areas combined.
 
for answers like repeat question , ok give lowest score, give Candidate_Score: 0
 
 
 If the answer or suggestion provided by the candidate is inappropriate, off-topic, or irrelevant, give Candidate_Score: 0.
 Question: {question}
 Answer: {answer}
 
"""


# Create a prompt template
prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template=prompt_template
)
 
 
evaluation_chain = LLMChain(
    llm=llm,
    prompt=prompt
)
 
def evaluate_answer(question: str, answer: str) -> int:
    result = evaluation_chain.run(question=question, answer=answer)
    # print(f"Result from evaluate_answer function is :- {result}")
    score = 0
    for line in result.splitlines():
        if "Candidate_Score" in line:
            # Extract and print the score
            score = line.split(':')[1].strip()
            # print(f"before returning:- {score}")
            return score
            # break
    print(f"returnign from function:- {score}")
    # return score


def get_messages_info(messages: list):
    return [SystemMessage(content=template)] + messages

chat_model = ChatHuggingFace(llm=llm)
chain = get_messages_info | chat_model

def get_state(messages: list) -> Literal["bye", "ask_human"]:
    if not isinstance(messages[-1], ChatMessage):
        return "ask_human"

def end_result(messages: list):
    return "Thank you for your time."

from langgraph.graph import MessageGraph
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
workflow = MessageGraph()

def ask_human(state: list):
    pass

workflow.add_node("info", chain)
workflow.add_node("bye", end_result)
workflow.add_node("ask_human", ask_human)

workflow.set_entry_point("info")
workflow.add_conditional_edges("info", get_state)
workflow.add_edge("ask_human", "info")
workflow.add_edge("bye", END)

graph = workflow.compile(checkpointer=memory, interrupt_before=["ask_human"])
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

def graph_stream(ans: str, msg_num: int, msg_info: list, uid: str, name: str):
    for output in graph.stream([ChatMessage(content=ans, role='user')], config=config, stream_mode='updates'):
        last_message = next(iter(output.values()))
        if msg_num >= TOTAL_QUESTIONS * 2:
            msg = end_result([])
            yield msg
            break
        if "output" in output.keys():
            last_message = output["info"]
        if type(last_message) != str:
            msg = last_message.content
        if msg:
            store_interview_data(uid, name, msg_info)
            yield msg
            print(msg_info)

class AnswerModel(BaseModel):
    ans: str
    msg_num: Optional[int] = 0
    msg_info: list
    uid : str
    name : str

@app.post("/langgraph")
async def main(answer: AnswerModel):
    return StreamingResponse(graph_stream(answer.ans, answer.msg_num, answer.msg_info, answer.uid, answer.name), media_type="application/json")

@app.get("/", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def check_langgraph():
    image_bytes = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    return Response(content=image_bytes, media_type="image/png")

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