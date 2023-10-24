import os
import asyncio
import requests
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status, Request
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from download import download_model_if_not_cached

app = FastAPI()

# url http://127.0.0.1:8000/v1/audio/transcriptions \
#  -H "Authorization: Bearer $OPENAI_API_KEY" \
#  -H "Content-Type: multipart/form-data" \
#  -F model="whisper-ch" \
#  -F file="@/path/to/file/openai.mp3"

# {
#  "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger..."
# }

MODEL_DATA_DIR = "/data/cache"
MEDIA_DIR = os.getenv("MEDIA_ROOT")


@lru_cache(maxsize=1)
def get_whisper_model(whisper_model: str, quantization: str) -> WhisperModel:
    """Get a whisper model from the cache or download it if it doesn't exist"""

    model_folder = download_model_if_not_cached(
        model_data_dir=MODEL_DATA_DIR,
        whisper_model_name=whisper_model,
        quantization=quantization,
    )

    model = WhisperModel(str(model_folder), compute_type=quantization)
    return model


def transcribe(
    audio_path: str, whisper_model: str, quantization: str, **whisper_args
) -> Iterable[Segment]:
    """Transcribe the audio file using whisper"""

    # Get whisper model
    # NOTE: If mulitple models are selected, this may keep all of them in memory depending on the cache size
    transcriber = get_whisper_model(whisper_model, quantization)

    segments, _ = transcriber.transcribe(
        audio=audio_path,
        **whisper_args,
    )

    return segments


WHISPER_DEFAULT_SETTINGS = {
    "whisper_model": os.getenv("MODEL"),
    "quantization": os.getenv("QUANTIZATION"),
    "task": "transcribe",
    "language": "de",
    "beam_size": 5,
}

async def transcribe_post(postback_uri: str, audio_path: str):
    print(f"{audio_path}: Starting transcription")
    segments = transcribe(audio_path, **WHISPER_DEFAULT_SETTINGS)
    
    segment_dicts = []

    for segment in segments:
        segment_dicts.append(
            {
                "transcript": segment.text,
                "start": segment.start,
                "end": segment.end,
            }
        )

    data = {"content": segment_dicts}
    r = requests.post(url=postback_uri, json=data)
    print(r.status_code, r.reason)

@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    model: str = Form(...),
    file: str = Form(...)
):
    
    print(f"Received request for {file}")
    postback_uri = request.headers.get("LanguageServicePostbackUri")

    assert model == "whisper-ch"
    loop = asyncio.get_running_loop()
    loop.create_task(transcribe_post(postback_uri, audio_path=str(file)))

    return "Transcription started"
