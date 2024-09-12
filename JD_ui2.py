import streamlit as st
import requests
import asyncio
import websockets
import sounddevice as sd
from pypdf import PdfReader
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
if "candidate_name" not in st.session_state:
    st.session_state.candidate_name = ""
if "uid" not in st.session_state:
    st.session_state.uid = str(uuid.uuid4())
if "job_description" not in st.session_state:
    st.session_state.job_description = ""
if "jd_sent" not in st.session_state:
    st.session_state.jd_sent = False
if "question_asked" not in st.session_state:
    st.session_state.question_asked = False
    
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
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        # New section for job description
        st.subheader("Job Description")
        job_description = st.text_area("Enter the job description here:", height=200)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                st.session_state.raw_text = get_pdf_text(pdf_docs)
                st.session_state.candidate_name = get_name(st.session_state.raw_text)
                st.session_state.job_description = job_description
                if st.session_state.raw_text:
                    st.success("Done")
                    st.session_state.pdf_flag = True

    def display_chat():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # def display_buttons():
    #     if st.session_state.question_asked:
    #         if st.button('Start Recording'):
    #             st.session_state.running = True      
    #             asyncio.run(start_websocket())
    #             # st.write("Recording started...")

    #         if st.button('Stop Recording'):
    #             st.session_state.running = False
    #             # st.write("Recording stopped.")


    display_chat()

    l = len(st.session_state.messages)

    if l >= (2 * (TOTAL_QUESTIONS)):
        asked = True
    else:
        asked = False

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
            "name": st.session_state.candidate_name,
        }

        # Send job description only once
        if not st.session_state.jd_sent:
            payload["job_description"] = st.session_state.job_description
            st.session_state.jd_sent = True

        response = requests.post('http://localhost:8000/langgraph', json=payload)
        with st.chat_message("ai"):
            st.markdown(response.text)
            st.session_state.messages.append({"role": "ai", "content": response.text})
            st.session_state.question_asked = True

        st.session_state.raw_text = ""  
        asyncio.run(start_websocket())
        st.rerun()

    @st.cache_data
    def cached_get_pdf_text(pdf_docs):
        return get_pdf_text(pdf_docs)

    if st.session_state.transcription:
        with st.chat_message("user"):
            st.markdown(st.session_state.transcription)
            st.session_state.messages.append({"role": "user", "content": st.session_state.transcription})
            print(f"######################## Transcription sending to server:- {st.session_state.transcription}")
        
        payload = {
            "ans": st.session_state.transcription,
            "msg_num": len(st.session_state.messages),
            "msg_info": st.session_state.messages,
            "uid": st.session_state.uid,
            "name": st.session_state.candidate_name,
        }

        # Send job description only once
        if not st.session_state.jd_sent:
            payload["job_description"] = st.session_state.job_description
            st.session_state.jd_sent = True

        response = requests.post('http://localhost:8000/langgraph', json=payload)
        if response.status_code == 200:
            st.session_state.messages.append({"role": "ai", "content": response.text})
            st.session_state.question_asked = True
            with st.chat_message("ai"):
                st.markdown(response.text)
                st.session_state.transcription = ""
                asyncio.run(start_websocket())
                st.rerun()
        else:
            st.error("Failed to get response from the server.")
    # display_buttons()
       
else:
    st.error('Failed to retrieve data')