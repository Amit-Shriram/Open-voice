import streamlit as st
import requests
import asyncio
import websockets
import time
import sounddevice as sd
from dotenv import load_dotenv
import uuid
import numpy as np
from gtts import gTTS
import pygame
from io import BytesIO

## VARIABLES
TOTAL_QUESTIONS = 15

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


# ... [rest of the imports and function definitions remain the same] ...

def generate_and_play_audio(text):
    try:
        mp3_fp = BytesIO()
        tts = gTTS(text, lang='en')
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Load and play audio directly from memory
        pygame.mixer.music.load(mp3_fp, 'mp3')
        pygame.mixer.music.play()

        # Wait until the audio is finished playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except Exception as e:
        st.error(f"Error during audio generation: {e}")

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

                    if upload_response.status_code == 200 and upload_response2.status_code == 200:
                        st.session_state.pdf_flag = True
                        st.success("Resume and Job Description submitted successfully!")
                    else:
                        st.error("Failed to upload files.")

    if st.session_state.pdf_flag:
        start_response = requests.post("http://localhost:8000/start_interview")
        print("1st")
        if start_response.status_code == 200:
            first_question = start_response.text
            with st.chat_message("ai"):
                st.markdown(first_question)
                st.session_state.messages.append({"role": "ai", "content": first_question})
            generate_and_play_audio(first_question)
            st.session_state.pdf_flag = False
            asyncio.run(start_websocket())
            st.rerun()

    def display_chat():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    display_chat()

    l = len(st.session_state.messages)
    asked = l >= (2 * (TOTAL_QUESTIONS))

    if ans := st.chat_input(placeholder="Answer: ", disabled=asked):
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
        print("2nd")
        if response.status_code == 200:
            response_data = response.json()
            with st.chat_message("ai"):
                st.markdown(response.text)
                st.session_state.messages.append({"role": "ai", "content": response.text})
                generate_and_play_audio(response.text)
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
        print(f"answer going to server:-{st.session_state.transcription}")

        response = requests.post('http://localhost:8000/langgraph', json=payload)
        print("3rd")
        if response.status_code == 200:
            response_data = response.json()
            st.session_state.messages.append({"role": "ai", "content": response.text})
            with st.chat_message("ai"):
                st.markdown(response.text)
                generate_and_play_audio(response.text)
            st.session_state.transcription = ""
            asyncio.run(start_websocket())
            st.rerun()
        else:
            st.error("Failed to get response from the server.")

else:
    st.error('Failed to retrieve data')
