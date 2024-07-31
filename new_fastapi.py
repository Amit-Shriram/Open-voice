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

load_dotenv()

app = FastAPI()

## VARIABLES
TOTAL_QUESTIONS = 2

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

template = """Your job is to interview the user based on the background information about the candidate.
Give follow up questions taking the candidate's answers into consideration.
Only output the question to be asked.
If you have been asked to repeat the question or they did not understand then repeat the question.
If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.
After you are able to discern all the information, call the relevant tool.
"""

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

def graph_stream(ans: str, msg_num: int):
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
            yield msg

class AnswerModel(BaseModel):
    ans: str
    msg_num: Optional[int] = 0

@app.post("/langgraph")
async def main(answer: AnswerModel):
    return StreamingResponse(graph_stream(answer.ans, answer.msg_num), media_type="application/json")

@app.get("/", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def check_langgraph():
    image_bytes = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    return Response(content=image_bytes, media_type="image/png")

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
