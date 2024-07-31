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

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

## VARIABLES
TOTAL_QUESTIONS = 5
pdf_flag = False

## Read the pdf
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

## STREAMLIT UI ##
response = requests.get('http://localhost:8000/')
if response.status_code == 200:
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
    if "running" not in st.session_state:
        st.session_state.running = False
    if "transcription" not in st.session_state:
        st.session_state.transcription = ""

    l = len(st.session_state.messages)

    ## CHECKING NUMBER OF QUESTIONS
    if l >= (2 * (TOTAL_QUESTIONS)): 
        asked = True
    else:
        asked = False

    # Displaying the question
    st.write("### Question:")
    if l > 0:
        with st.chat_message(st.session_state.messages[-1]["role"]):
            st.markdown(st.session_state.messages[-1]["content"])

    if pdf_flag:
        ans = raw_text
    else:
        raw_text = ""
        ans = ""

    def update_value(input):
        ans = input
    
    if ans := st.chat_input(placeholder="Answer: ", disabled=asked, key="node", on_submit=update_value, args=(ans,)) or raw_text:
        if not raw_text:
            with st.chat_message("user"):
                st.markdown(ans)
                st.session_state.messages.append({"role": "user", "content": ans})  
        
        output = None
        ## Using LangGraph
        response = requests.post('http://localhost:8000/langgraph',
                                 json={
                                     "ans": ans, 
                                     "msg_num": len(st.session_state.messages)
                                 })
        with st.chat_message("ai"):
            st.markdown(response.text)
            st.session_state.messages.append({"role": "ai", "content": response.text})
           
            st.rerun()

    async def send_audio(uri):
        async with websockets.connect(uri) as websocket:
            st.write("WebSocket connection established.")
            
            while st.session_state.running:
                st.write("Recording 5 seconds of audio...")
                audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='int16')
                sd.wait()

                if not st.session_state.running:
                    break

                st.write("Sending audio chunk to server...")
                await websocket.send(audio.tobytes())

                transcription = await websocket.recv()
                st.session_state.transcription = transcription

                st.write("Received transcription from server:", transcription)

            # Send a stop signal to the server
            await websocket.send(b"STOP")
            st.write("Sent stop signal to server.")

    async def start_websocket():
        await send_audio('ws://localhost:8000/ws')

    
    # Display the recording buttons
    st.write("### Real-Time Speech-to-Text")
    col1, col2 = st.columns([2, 1])  # Adjust the column width ratios

    with col1:
        if st.button('Start Recording'):
            st.session_state.running = True
            asyncio.run(start_websocket())
            st.write("Recording started...")

    with col2:
        if st.button('Stop Recording'):
            st.session_state.running = False
            st.write("Recording stopped.")
            if st.session_state.transcription:
                response = requests.post('http://localhost:8000/langgraph',
                                         json={
                                             "ans": st.session_state.transcription, 
                                             "msg_num": len(st.session_state.messages)
                                         })
                if response.status_code == 200:
                    for msg in response.iter_lines():
                        st.session_state.messages.append({"role": "assistant", "content": msg.decode()})
                        with st.chat_message("assistant"):
                            st.markdown(msg.decode())
                else:
                    st.error("Failed to get response from the server.")

    

    # async def start_websocket():
    #     await send_audio('ws://localhost:8000/ws')

else:
    st.error('Failed to retrieve data')
