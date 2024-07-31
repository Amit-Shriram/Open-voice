import requests
import streamlit as st
import uuid
from pypdf import PdfReader
import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

## VARIABLES
TOTAL_QUESTIONS = 2
pdf_flag = False

## Read the pdf
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

## OPENVOICE ##
from OpenVoice.openvoice.api import BaseSpeakerTTS
import torch
import pyaudio
import wave
## Model and device setup
en_ckpt_base = 'OpenVoice/checkpoints/base_speakers/EN' 
ckpt_converter = 'OpenVoice/checkpoints/converter'
device = 'cpu' # device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
# @app.post("/tts")
def process_and_play(prompt, style, audio_file_path):
    tts_model = en_base_speaker_tts 
    ## Process text and generate audio
    try:
        tts_model.tts(prompt, audio_file_path, speaker=style, language='English')
        print("Audio generated successfully.")
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
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    st.success("Done")
                    ans = raw_text
                    pdf_flag = True

    if "messages" not in st.session_state:
        st.session_state.messages = []
    l = len(st.session_state.messages)

    ## CHECKING NUMBER OF QUESTIONS
    if l >= (2*(TOTAL_QUESTIONS)): 
        asked = True
    else:
        asked = False

    if l == 1:
        with st.chat_message(st.session_state.messages[0]["role"]):
            st.markdown(st.session_state.messages[0]["content"])
    elif l>1:
        with st.chat_message(st.session_state.messages[0]["role"]):
            st.markdown(st.session_state.messages[0]["content"])
        for i in range(1,l):
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
    
    if ans := st.chat_input(placeholder="Answer: ",disabled=asked, key="node", on_submit=update_value, args=(ans,)) or raw_text:
        if not(raw_text):
            with st.chat_message("user"):
                st.markdown(ans)
                st.session_state.messages.append({"role": "user", "content": ans})  
        
        output = None
        ## Using LangGraph
        response = requests.post('http://localhost:8000/langgraph',
                            json = {
                                    "ans":ans, 
                                    "msg_num":len(st.session_state.messages)
                                }
                        )
        with st.chat_message("ai"):
            st.markdown(response.text)
            st.session_state.messages.append({"role": "ai", "content": response.text})
           
            process_and_play(response.text,"friendly","outputs/ai_audio.wav")
            st.rerun()
else:
    st.error('Failed to retrieve data')
