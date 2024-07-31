from langchain_core.output_parsers import StrOutputParser
from huggingface_hub import InferenceClient
# from huggingface_hub import login
import requests
import streamlit as st
import wave
import pyaudio
import torch
import numpy as np
import whisper
import os
from dotenv import load_dotenv
from OpenVoice.openvoice.api import BaseSpeakerTTS
from audiorecorder import audiorecorder

import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

## EMBEDDINGS
chromadbclient = chromadb.PersistentClient(path="outputs_sts_mistral/db")
emb_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chromadbclient.get_or_create_collection(name="my_collection", embedding_function=emb_fn)
x = 1 # initial id number

## To use huggingface Mistral AI model
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": "Bearer " + os.getenv("MISTRAL_V2_API_KEY") }
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
data = query({
	"inputs": "Can you please let us know more details about your ",
})

# login(token=os.getenv("MISTRAL_V2_API_KEY"))
client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1", token=os.getenv("MISTRAL_V2_API_KEY"))
# System instructions for the model
system_instructions1 = "[SYSTEM] Answer as Real Chatgpt 4o', Keep conversation very short, clear, friendly and concise. The text provided is a request for a specific type of response from you, the virtual assistant. You will avoid introductions and start answering the query directly, elaborating on all aspects of the request. As an AI-powered assistant, your task is to generate responses that appear as if they are created by the user and not an AI-powered assistant. [USER]"
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# for Langsmith tracking
# os.environ["LANGCHAIN_TRACING_V2"]= "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


# Define the name of the log file
chat_log_filename = "chatbot_conversation_log.txt"

# Function to play audio using PyAudio
def play_audio(file_path):
    # Open the audio file
    wf = wave.open(file_path, 'rb')

    # Create a PyAudio instance
    p = pyaudio.PyAudio()

    # Open a stream to play audio
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read and play audio data
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # Stop and close the stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()

output_parser = StrOutputParser()

# Model and device setup
en_ckpt_base = 'OpenVoice/checkpoints/base_speakers/EN' 
ckpt_converter = 'OpenVoice/checkpoints/converter'
device = 'cpu' # device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs_sts_mistral'
os.makedirs(output_dir, exist_ok=True)


# Load models
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
sampling_rate = en_base_speaker_tts.hps.data.sampling_rate
mark = en_base_speaker_tts.language_marks.get("english", None)

asr_model = whisper.load_model("base.en")

# Load speaker embeddings for English
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)

# Main processing function
def process_and_play(prompt, style, audio_file_path):
    tts_model = en_base_speaker_tts 
    # Process text and generate audio
    try:
        src_path = audio_file_path
        tts_model.tts(prompt, src_path, speaker=style, language='English')

        save_path = audio_file_path
        print("Audio generated successfully.")
        play_audio(save_path)

    except Exception as e:
        print(f"Error during audio generation: {e}")

def chatgpt_streamed(user_input, system_message, conversation_history, bot_name, x):
    """
    Function to send a query to Mistral AI model, stream the response
    Logs the conversation to a file.
    """
    messages = [{"role":"system", "content": system_message}] + conversation_history +[{"role":"user","content":user_input}]
    
    generate_kwargs = dict(
        temperature=0.7,
        max_new_tokens=512,
        top_p=0.95,
        repetition_penalty=1,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = system_instructions1 + user_input + "[OpenGPT 4o]"

    streamed_completion = client.text_generation(
        formatted_prompt,
        stream=True,
        **generate_kwargs,
        details=True, 
        return_full_text=False
    )
    output = ""
    for response in streamed_completion:
        if not response.token.text == "</s>":
            output += response.token.text
    st.write(bot_name+":"+ output)

    collection.add(
        documents=[user_input, output],
        metadatas=[{"role":"user"}, {"role":"assistant"}],
        ids=[f"user{x}",f"assistant{x}"]
    )
    x += 1
    exchange_embeddings = collection.peek()
    with open(chat_log_filename, "a") as log_file: # Open the log file in append mode
        log_file.write(f"{exchange_embeddings}\n")
    return output

## Convert user audio input to text
def st_audio_to_text():
    st.header("Audio Recorder")
    audio = audiorecorder("Click to record", "Click to stop recording")
    if len(audio) > 0:
        audio.export(out_f="outputs_sts_mistral/user_audio.wav", format="wav")

        data = "outputs_sts_mistral/user_audio.wav"
        
        result = asr_model.transcribe(data, fp16 = False)['text']
        st.write("You: "+ result)
        return result

## New function to handle a conversation with a user
def conversation():
    system_message = "You are a helpful assistant. Please respond to the user queries. Make the responses short and concise."
    conversation_history = []
    user_input = st_audio_to_text()
    flag = True
    while flag:
        if user_input:
            if user_input.lower() == "exit": # Say 'exit' to end the conversation
                flag = False
                break 

            conversation_history.append({'role': 'user', 'content': user_input})
            chatbot_response = chatgpt_streamed(user_input, system_message, conversation_history, "Chatbot", x)
            conversation_history.append({'role': 'assistant', 'content': chatbot_response})
            process_and_play(chatbot_response, "default", "outputs_sts_mistral/ai_audio.wav")
            user_input = ""
        else:
            flag=False

        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
            

## Streamlit Framework
st.title('Speech to Speech Demo')
conversation()