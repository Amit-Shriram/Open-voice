#this is proper code to automate the process of start and stop recoding
import requests
import streamlit as st
import uuid
from pypdf import PdfReader
import os
from dotenv import load_dotenv
import asyncio
import websockets
import sounddevice as sd
import numpy as np
import time
import torch
import pyaudio
import wave

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

## VARIABLES
TOTAL_QUESTIONS = 10

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
if "candidate_name" not in st.session_state:
    st.session_state.candidate_name = ""
if "uid" not in st.session_state:
    st.session_state.uid = str(uuid.uuid4())

## Read the pdf
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() 
    return text

def get_name(text):
    lines = text.split('\n')
    candidate_name = lines[0].strip()
    return candidate_name

## OPENVOICE ##
from OpenVoice.OpenVoice.api import BaseSpeakerTTS

## Model and device setup
en_ckpt_base = 'OpenVoice/checkpoints/base_speakers/EN' 
ckpt_converter = 'OpenVoice/checkpoints/converter'
device = 'cpu'  # device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

## Load models
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
sampling_rate = en_base_speaker_tts.hps.data.sampling_rate
mark = en_base_speaker_tts.language_marks.get("english", None)

## Load speaker embeddings for English
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)

## Function to play audio using PyAudio
def play_audio(file_path):
    ## Open the audio file
    wf = wave.open(file_path, 'rb')

    ## Create a PyAudio instance
    p = pyaudio.PyAudio()

    ## Open a stream to play audio
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    ## Read and play audio data
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    ## Stop and close the stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()

## Main processing function
def process_and_play(prompt, style, audio_file_path):
    tts_model = en_base_speaker_tts 
    ## Process text and generate audio
    try:
        tts_model.tts(prompt, audio_file_path, speaker=style, language='English')
        play_audio(audio_file_path)

    except Exception as e:
        print(f"Error during audio generation: {e}")

## STREAMLIT UI ##
response = requests.get('http://localhost:8000/')
if response.status_code == 200:
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                st.session_state.raw_text = get_pdf_text(pdf_docs)
                st.session_state.candidate_name = get_name(st.session_state.raw_text)
                if st.session_state.raw_text:
                    st.success("Done")
                    st.session_state.pdf_flag = True

    def display_chat():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def check_silence(audio_chunk):
        return np.max(np.abs(audio_chunk)) < 500  # Adjust the threshold as needed

    async def send_audio(uri):
        async with websockets.connect(uri) as websocket:
            silence_start = time.time()
            while st.session_state.running:
                audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='int16')
                sd.wait()

                if check_silence(audio):
                    if time.time() - silence_start > 5:  # 5 seconds of silence
                        st.session_state.running = False
                        break            
                else:
                    silence_start = time.time()

                await websocket.send(audio.tobytes())
                transcription = await websocket.recv()
                st.session_state.transcription += transcription

            # Send a stop signal to the server
            await websocket.send(b"STOP")

    async def start_websocket():
        st.session_state.running = True
        await send_audio('ws://localhost:8000/ws')

    # Display previous chat messages
    display_chat()

    l = len(st.session_state.messages)

    ## CHECKING NUMBER OF QUESTIONS
    if l >= (2 * (TOTAL_QUESTIONS)): 
        asked = True
    else:
        asked = False

    # Displaying the question
    ans = st.session_state.raw_text if st.session_state.pdf_flag else ""

    if ans := st.chat_input(placeholder="Answer: ", disabled=asked, key="node") or st.session_state.raw_text:
        if not st.session_state.raw_text:
            with st.chat_message("user"):
                st.markdown(ans)
                st.session_state.messages.append({"role": "user", "content": ans})  
        
        output = None
        ## Using LangGraph
        response = requests.post('http://localhost:8000/langgraph',
                                 json={
                                     "ans": ans, 
                                     "msg_num": len(st.session_state.messages),
                                     "msg_info": st.session_state.messages,
                                     "uid" : st.session_state.uid,
                                     "name": st.session_state.candidate_name
                                 })
        with st.chat_message("ai"):
            st.markdown(response.text)
            st.session_state.messages.append({"role": "ai", "content": response.text})
            st.session_state.question_asked = True
            # process_and_play(response.text,"friendly","outputs/ai_audio.wav") 

        st.session_state.raw_text = ""  # Clear raw text after processing
        # await start_websocket()  # Start recording automatically after the AI asks a question
        asyncio.run(start_websocket())
        st.rerun()

    if st.session_state.transcription:
        with st.chat_message("user"):
            st.markdown(st.session_state.transcription)
            st.session_state.messages.append({"role": "user", "content": st.session_state.transcription})
        response = requests.post('http://localhost:8000/langgraph',
                                json={
                                    "ans": st.session_state.transcription, 
                                    "msg_num": len(st.session_state.messages),
                                    "msg_info": st.session_state.messages,
                                    "uid" : st.session_state.uid,
                                    "name": st.session_state.candidate_name
                                })
        if response.status_code == 200:
            st.session_state.messages.append({"role": "ai", "content": response.text})
            st.session_state.question_asked = True
            with st.chat_message("ai"):
                st.markdown(response.text)
                # process_and_play(response.text,"friendly","outputs/ai_audio.wav") 
                asyncio.run(start_websocket())
                st.rerun()
        else:
            st.error("Failed to get response from the server.")
        
        st.session_state.transcription = ""

else:
    st.error('Failed to retrieve data')
