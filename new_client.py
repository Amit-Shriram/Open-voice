# import streamlit as st
# import asyncio
# import websockets
# import sounddevice as sd
# import numpy as np
# import requests
# import uuid
# from pypdf import PdfReader
# import os
# from dotenv import load_dotenv

# load_dotenv()

# hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# ## VARIABLES
# TOTAL_QUESTIONS = 5
# pdf_flag = False

# ## Read the pdf
# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Initialize session state
# if "running" not in st.session_state:
#     st.session_state.running = False
# if "transcription" not in st.session_state:
#     st.session_state.transcription = ""

# async def send_audio(uri):
#     async with websockets.connect(uri) as websocket:
#         st.write("WebSocket connection established.")
        
#         while st.session_state.running:
#             st.write("Recording 5 seconds of audio...")
#             audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='int16')
#             sd.wait()

#             if not st.session_state.running:
#                 break

#             st.write("Sending audio chunk to server...")
#             await websocket.send(audio.tobytes())

#             transcription = await websocket.recv()
#             st.session_state.transcription = transcription

#             st.write("Received transcription from server:", transcription)

#         # Send a stop signal to the server
#         await websocket.send(b"STOP")
#         st.write("Sent stop signal to server.")

# async def start_websocket():
#     await send_audio('ws://localhost:8000/ws')

# ## STREAMLIT UI ##
# response = requests.get('http://localhost:8000/')
# if response.status_code == 200:
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 if raw_text:
#                     st.success("Done")
#                     ans = raw_text
#                     pdf_flag = True

#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     l = len(st.session_state.messages)

#     ## CHECKING NUMBER OF QUESTIONS
#     if l >= (2 * TOTAL_QUESTIONS): 
#         asked = True
#     else:
#         asked = False

#     if l == 1:
#         with st.chat_message(st.session_state.messages[0]["role"]):
#             st.markdown(st.session_state.messages[0]["content"])

#     if "recording" not in st.session_state:
#         st.session_state.recording = False

#     if pdf_flag:
#         st.title("📄 Chat with LLM based on your Resume!")
#         st.text_area("Transcript will appear here:", value=st.session_state.transcription, height=200)
#         if st.button("Start Interview"):
#             st.session_state.recording = True
#             st.session_state.running = True
#             asyncio.run(start_websocket())

#         if st.button("Stop Interview"):
#             st.session_state.recording = False
#             st.session_state.running = False
#             st.success("Interview stopped.")
        
#         # Display messages and handle chat
#         if st.session_state.messages:
#             for message in st.session_state.messages:
#                 with st.chat_message(message["role"]):
#                     st.markdown(message["content"])

#         if asked:
#             st.session_state.running = False
#             st.session_state.recording = False
#             st.success("Interview process completed. Thank you for participating!")

#         if st.session_state.recording:
#             if st.session_state.transcription:
#                 st.chat_message("user", content=st.session_state.transcription)
#                 st.session_state.messages.append({"role": "user", "content": st.session_state.transcription})

#                 # Send transcription to FastAPI for processing
#                 response = requests.post("http://localhost:8000/langgraph", json={"ans": st.session_state.transcription, "msg_num": len(st.session_state.messages)})
#                 if response.status_code == 200:
#                     for msg in response.iter_lines():
#                         st.session_state.messages.append({"role": "assistant", "content": msg.decode()})
#                         with st.chat_message("assistant"):
#                             st.markdown(msg.decode())
#                 else:
#                     st.error("Failed to get response from the server.")








# import requests
# import streamlit as st
# import uuid
# from pypdf import PdfReader
# import os
# from dotenv import load_dotenv
# import asyncio
# import websockets
# import sounddevice as sd
# import numpy as np

# load_dotenv()

# hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

# ## VARIABLES
# TOTAL_QUESTIONS = 5
# pdf_flag = False

# ## Read the pdf
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# ## STREAMLIT UI ##
# response = requests.get('http://localhost:8000/')
# if response.status_code == 200:
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 if raw_text:
#                     st.success("Done")
#                     ans = raw_text
#                     pdf_flag = True

#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "running" not in st.session_state:
#         st.session_state.running = False
#     if "transcription" not in st.session_state:
#         st.session_state.transcription = ""

#     l = len(st.session_state.messages)

#     ## CHECKING NUMBER OF QUESTIONS
#     if l >= (2 * (TOTAL_QUESTIONS)): 
#         asked = True
#     else:
#         asked = False

#     if l == 1:
#         with st.chat_message(st.session_state.messages[0]["role"]):
#             st.markdown(st.session_state.messages[0]["content"])
#     elif l > 1:
#         with st.chat_message(st.session_state.messages[0]["role"]):
#             st.markdown(st.session_state.messages[0]["content"])
#         for i in range(1, l):
#             with st.chat_message(st.session_state.messages[i]["role"]):
#                 st.markdown(st.session_state.messages[i]["content"])

#     config = {"configurable": {"thread_id": str(uuid.uuid4())}}

#     if pdf_flag:
#         ans = raw_text
#     else:
#         raw_text = ""
#         ans = ""

#     def update_value(input):
#         ans = input
    
#     if ans := st.chat_input(placeholder="Answer: ", disabled=asked, key="node", on_submit=update_value, args=(ans,)) or raw_text:
#         if not raw_text:
#             with st.chat_message("user"):
#                 st.markdown(ans)
#                 st.session_state.messages.append({"role": "user", "content": ans})  
        
#         output = None
#         ## Using LangGraph
#         response = requests.post('http://localhost:8000/langgraph',
#                                  json={
#                                      "ans": ans, 
#                                      "msg_num": len(st.session_state.messages)
#                                  })
#         with st.chat_message("ai"):
#             st.markdown(response.text)
#             st.session_state.messages.append({"role": "ai", "content": response.text})
           
#             st.rerun()

#     st.title("Real-Time Speech-to-Text")

#     # col1, col2 = st.columns(2)
#     # col1, col2, col3 = st.columns(3)

#     async def send_audio(uri):
#         async with websockets.connect(uri) as websocket:
#             st.write("WebSocket connection established.")
            
#             while st.session_state.running:
#                 st.write("Recording 5 seconds of audio...")
#                 audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='int16')
#                 sd.wait()

#                 if not st.session_state.running:
#                     break

#                 st.write("Sending audio chunk to server...")
#                 await websocket.send(audio.tobytes())

#                 transcription = await websocket.recv()
#                 st.session_state.transcription = transcription

#                 st.write("Received transcription from server:", transcription)

#             # Send a stop signal to the server
#             await websocket.send(b"STOP")
#             st.write("Sent stop signal to server.")

#     async def start_websocket():
#         await send_audio('ws://localhost:8000/ws')

#     # with col1:
#     if st.button('Start Recording'):
#             st.session_state.running = True
#             asyncio.run(start_websocket())
#             st.write("Recording started...")

#     # with col2:
#     if st.button('Stop Recording'):
#             st.session_state.running = False
#             st.write("Recording stopped.")
#             # st.write("Sending Recording..")
#             if st.session_state.transcription:
#                 response = requests.post('http://localhost:8000/langgraph',
#                                          json={
#                                              "ans": st.session_state.transcription, 
#                                              "msg_num": len(st.session_state.messages)
#                                          })
#                 if response.status_code == 200:
#                     for msg in response.iter_lines():
#                         st.session_state.messages.append({"role": "assistant", "content": msg.decode()})
#                         with st.chat_message("assistant"):
#                             st.markdown(msg.decode())
#                 else:
#                     st.error("Failed to get response from the server.")
# else:
#     st.error('Failed to retrieve data')











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

    if l == 1:
        with st.chat_message(st.session_state.messages[0]["role"]):
            st.markdown(st.session_state.messages[0]["content"])
    elif l > 1:
        with st.chat_message(st.session_state.messages[0]["role"]):
            st.markdown(st.session_state.messages[0]["content"])
        for i in range(1, l):
            with st.chat_message(st.session_state.messages[i]["role"]):
                st.markdown(st.session_state.messages[i]["content"])

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

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

    st.title("Real-Time Speech-to-Text")

    col1, col2 = st.columns(2)

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

    with col1:
        if st.button('Start Recording', disabled=st.session_state.running):
            st.session_state.running = True
            asyncio.run(start_websocket())
            st.write("Recording started...")

    with col2:
        if st.button('Stop Recording', disabled=st.session_state.running):
            st.session_state.running = True
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
else:
    st.error('Failed to retrieve data')
