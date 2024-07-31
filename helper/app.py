from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import base64
import nltk
from nltk.tokenize import sent_tokenize

app = FastAPI()

html = open("client/index.html").read()

nltk.download('punkt')

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    transcript = []
    while True:
        data = await websocket.receive_text()
        if data.startswith("data:audio/wav;base64,"):
            audio_data = data.split(",")[1]
            with open("audio_message.wav", "wb") as f:
                f.write(base64.b64decode(audio_data))
            await websocket.send_text("Audio message received")
        else:
            transcript.append(data)
            with open("transcript.txt", "w") as f:
                f.write(" ".join(transcript))
            
            # Combine the sentences semantically
            combined_transcript = " ".join(transcript)
            sentences = sent_tokenize(combined_transcript)
            semantic_transcript = " ".join(sentences)
            
            await websocket.send_text(f"Message text was: {semantic_transcript}")
