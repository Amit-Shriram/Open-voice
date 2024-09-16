import streamlit as st
import requests
import asyncio
import websockets
import time
import sounddevice as sd
from dotenv import load_dotenv
from pypdf import PdfReader
import pyaudio
import wave
import torch
from TTS.api import TTS
import uuid

## VARIABLES
TOTAL_QUESTIONS = 7

## Initialize session state attributes if they don't exist
if "pdf_flag" not in st.session_state:
    st.session_state.pdf_flag = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "running" not in st.session_state:
    st.session_state.running = False
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "uid" not in st.session_state:
    st.session_state.uid = str(uuid.uuid4())
if "tts_model" not in st.session_state:
    st.session_state.tts_model = None
 

if st.session_state.tts_model is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.session_state.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
tts = st.session_state.tts_model
 
# Function to generate and play audio using xTTS
def generate_and_play_audio(text, speaker_wav, language="en", file_path="output.wav"):
    try:
        # wav = tts.tts(text=text, speaker_wav=speaker_wav, language=language)
        tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=file_path)
        play_audio(file_path)
    except Exception as e:
        st.error(f"Error during audio generation: {e}")

# Function to play audio using PyAudio
def play_audio(file_path):
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    stream.stop_stream()
    stream.close()
    p.terminate()

# Function to start the WebSocket client and send audio chunks
async def send_audio(uri):
    async with websockets.connect(uri) as websocket:
        while st.session_state.running:
            audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='int16')
            sd.wait()

            await websocket.send(audio.tobytes())
            transcription = await websocket.recv()
            print(transcription)
            if transcription.strip() == "you":
                st.session_state.running = False
                break

            st.session_state.transcription += transcription
                
        await websocket.send(b"STOP")

# Function to start the WebSocket connection
async def start_websocket():
    st.session_state.running = True
    await send_audio('ws://localhost:8000/ws')

# Streamlit UI for uploading PDF and displaying chat
response = requests.get('http://localhost:8000/')
if response.status_code == 200:
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", type=["pdf"])
        
        # New section for job description
        st.subheader("Job Description")
        job_description = st.text_area("Enter the job description here:", height=200)
        
        if st.button("Submit & Process"):
            if pdf_docs and job_description:
                with st.spinner("Processing..."):
                    files = {"resume": pdf_docs}
                    data = {"jd": job_description}
                    upload_response = requests.post("http://localhost:8000/submit_resume", files=files)
                    upload_response2 = requests.post("http://localhost:8000/submit_JD", json=data)

                    if upload_response.status_code == 200 and upload_response2.status_code==200:
                        st.session_state.raw_text = upload_response.json().get("resume_text", "")
                        st.session_state.pdf_flag = True
                        if st.session_state.raw_text:
                            st.success("Done")
                    else:
                        st.error("Failed to upload files.")
    
    def display_chat():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    display_chat()

    l = len(st.session_state.messages)
    asked = l >= (2 * (TOTAL_QUESTIONS))

    ans = st.session_state.raw_text if st.session_state.pdf_flag else ""

    if ans := st.chat_input(placeholder="Answer: ", disabled=asked, key="node") or st.session_state.raw_text:
        if not st.session_state.raw_text:
            with st.chat_message("user"):
                st.markdown(ans)
                st.session_state.messages.append({"role": "user", "content": ans})  
       
        payload = {
            "ans": ans,
            "msg_num": len(st.session_state.messages),
            "msg_info": st.session_state.messages,
            "uid": st.session_state.uid,
        }

        response = requests.post('http://localhost:8000/langgraph', json=payload)
        if response.status_code == 200:
            with st.chat_message("ai"):
                st.markdown(response.text)
                st.session_state.messages.append({"role": "ai", "content": response.text})
                generate_and_play_audio(response.text, "prajwal.wav", language="en", file_path="outputs/ai_audio.wav")

            st.session_state.raw_text = ""  
            asyncio.run(start_websocket())
            st.rerun()

    if st.session_state.transcription:
        with st.chat_message("user"):
            st.markdown(st.session_state.transcription)
            st.session_state.messages.append({"role": "user", "content": st.session_state.transcription})

        payload = {
            "ans": st.session_state.transcription,
            "msg_num": len(st.session_state.messages),
            "msg_info": st.session_state.messages,
            "uid": st.session_state.uid,
        }

        response = requests.post('http://localhost:8000/langgraph', json=payload)
        if response.status_code == 200:
            st.session_state.messages.append({"role": "ai", "content": response.text})
            with st.chat_message("ai"):
                st.markdown(response.text)
                generate_and_play_audio(response.text, "prajwal.wav", language="en", file_path="outputs/ai_audio.wav")
                st.session_state.transcription = ""
                asyncio.run(start_websocket())
                st.rerun()
        else:
            st.error("Failed to get response from the server.")
else:
    st.error('Failed to retrieve data')
