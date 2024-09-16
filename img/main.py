# this is a final main file for interview bot

from fastapi import FastAPI, WebSocket, UploadFile, Form, File
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
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.llms import OllamaLLM
import PyPDF2
import io
from fastapi.responses import JSONResponse

load_dotenv()

llm_url = os.getenv("llama3.1_URL")

app = FastAPI()

## VARIABLES
TOTAL_QUESTIONS = 7

resume_data_global = None
jd_data_global = None

# token for pocketbase
auth_token = None

model = OllamaLLM(model="llama3.1", temperature=0.75 , base_url=llm_url)

class AskHuman(BaseModel):
    """Ask the human a question"""
    question: str


template = """Your task is to interview the candidate based on their resume data and Job description provided. 
Ask only 7 questions as described below.

resume data:
    {resume_data}

job description: 
    {jd_data}

Make sure the questions are short and straightforward, without any explanations. 
Only output the next question to be asked.

1. Greet the candidate by their name and ask them to introduce themselves.
2. Ask a question related to their latest project from the resume.
3. Ask a technical question based on their resume and job description.
4. Ask another technical question based on their resume and job description.
5. Ask a situational-based question relevant to the candidate's role and job description.
6. Ask if the candidate has any questions about Centralogic. Use this info: Centralogic was founded in 2010 by Mr. Ajay Navgale and Mr. Sanjay Navgale. Services include Project Management, DevOps, Cloud Migration, Cyber Security. 3.45 LPA for freshers, 2-year bond.
7. Thank the candidate for their time.

Do not give any summary or additional comments.
"""

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

def evaluate_answer(question: str, answer: str) -> int:
    result = chain2.invoke({"question": question, "answer": answer})
    try:
        return int(result.strip())
    except ValueError:
        return 0
    
def get_messages_info(messages: list):
    global resume_data_global, jd_data_global
    formatted_template = template.format(
        resume_data = resume_data_global,
        jd_data = jd_data_global
    )
    return [SystemMessage(content=formatted_template)] + messages

chain = get_messages_info | model

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

workflow.add_node("info", chain)
workflow.add_node("bye", end_result)
workflow.add_node("ask_human", ask_human)

workflow.set_entry_point("info")
workflow.add_conditional_edges("info", get_state)
workflow.add_edge("ask_human", "info")
workflow.add_edge("bye", END)

graph = workflow.compile(checkpointer=memory, interrupt_before=["ask_human"])        
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

def graph_stream(ans: str, msg_num: int, msg_info: list, uid: str):
    msg = None       
    for output in graph.stream([ChatMessage(content=ans, role='user')], config=config, stream_mode='updates'):         
        last_message = next(iter(output.values()))      #The next(iter(...)) syntax gets the first value from the output dictionary.

        if msg_num >= TOTAL_QUESTIONS * 2:              
            msg = end_result([])
            yield msg
            break

        if "output" in output.keys():
            last_message = output["info"]
        
        if isinstance(last_message, str):
            msg = last_message
        elif hasattr(last_message, 'content'):
            msg = last_message.content

        if msg:
            store_interview_data(uid, msg_info)
            yield msg
            print(msg_info)


class AnswerModel(BaseModel):
    ans: str
    msg_num: Optional[int] = 0
    msg_info: list
    uid: str
    # name: str
    # job_description: Optional[str] = None

@app.post("/langgraph")
async def main(answer: AnswerModel):    
    return StreamingResponse(graph_stream(answer.ans, answer.msg_num, answer.msg_info, answer.uid), media_type="application/json")


@app.post("/submit_resume")
async def process_resume(resume: UploadFile = File(...) ):
    global resume_data_global

    resume_data = await resume.read()
    
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(resume_data))
    extracted_resume_text = ""
    for page_num in range(len(pdf_reader.pages)):
        extracted_resume_text += pdf_reader.pages[page_num].extract_text()
    resume_data_global = extracted_resume_text
    return JSONResponse(content={"resume_text": resume_data_global})


class JDRequest(BaseModel):
    jd: str

@app.post("/submit_JD")
async def process_jd(request:JDRequest):
    global jd_data_global
    jd_data_global = request.jd
    # print(f"JD data:- {jd_data_global}")
    return {"message": "JD received"}

@app.get("/", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def check_langgraph():
    image_bytes = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    return Response(content=image_bytes, media_type="image/png")

def get_name(text):
    lines = text.split('\n')
    candidate_name = lines[0].strip()
    return candidate_name

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

def store_interview_data(uid: str, msg_info: list):
    global resume_data_global
    base_url = 'https://dev-pocketbase3.huhoka.com'
    collection = 'interview_bot'
    endpoint = f'{base_url}/api/collections/{collection}/records'
    score = 0
    name = get_name(resume_data_global)
    latest_ai_message = None
    latest_user_message = None

    for message in reversed(msg_info):
        if message['role'] == 'ai' and latest_ai_message is None:
            latest_ai_message = message['content']
        elif message['role'] == 'user' and latest_user_message is None:
            latest_user_message = message['content']
        
        if latest_ai_message and latest_user_message:
            score = evaluate_answer(latest_ai_message, latest_user_message)
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

    try:
        while True:
            data = await websocket.receive_bytes()

            if data == b"STOP":
                print("Received stop signal from client.")
                break

            audio_filename = "recorded_audio.wav"

            # Save audio chunk
            audio_data = np.frombuffer(data, dtype=np.int16)
            wavio.write(audio_filename, audio_data, rate=16000, sampwidth=2)

            # Process audio for transcription
            audio_data_float = audio_data.astype(np.float32) / 32768.0
            result = pipe(audio_data_float, return_timestamps=True, generate_kwargs={"language": "english"})
            transcription_text = result["text"]
            print(f"{transcription_text}")

            transcription_file.writelines(transcription_text)

            # Send transcription back to client
            await websocket.send_text(transcription_text)
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        transcription_file.close()
        print("Transcription file closed.")
