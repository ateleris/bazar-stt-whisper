import os
import asyncio
import uuid

import requests
import tempfile

from functools import lru_cache
from pathlib import Path
from typing import Iterable

from fastapi import FastAPI, Form, HTTPException, status, Request
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from azure.storage.blob import BlobClient

from download import download_model_if_not_cached
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# url http://127.0.0.1:8000/v1/audio/transcriptions \
#  -H "Authorization: Bearer $OPENAI_API_KEY" \
#  -H "Content-Type: multipart/form-data" \
#  -F model="whisper-ch" \
#  -F file="@/path/to/file/openai.mp3"

# {
#  "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger..."
# }

MODEL_DATA_DIR = os.getenv("MODEL_DATA_DIR")
MEDIA_DIR = os.getenv("MEDIA_ROOT")

AZURE_BLOB_STORAGE_CONTAINER_NAME = os.getenv('AZURE_BLOB_STORAGE_CONTAINER_NAME')
AZURE_BLOB_STORAGE_CONNECTION_STRING = os.getenv('AZURE_BLOB_STORAGE_CONNECTION_STRING')

WHISPER_DEFAULT_SETTINGS = {
    "whisper_model": os.getenv("MODEL"),
    "quantization": os.getenv("QUANTIZATION"),
    "task": "transcribe",
    "language": "de",
    "beam_size": 5,
}


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


async def transcribe_post(postback_uri: str, audio_path: str):
    try:
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            r = requests.post(url=postback_uri, json=None)
            r.raise_for_status()
            return

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
        print(f"{audio_path}: Posting transcription to {postback_uri}")
        r = requests.post(url=postback_uri, json=data)
        r.raise_for_status()
        print(f"Deleting {audio_path}")
        os.remove(audio_path)

    except Exception as e:
        print(e)
        os.remove(audio_path)


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


@app.post("/v1/audio/transcriptions/azure-file")
async def transcriptions_azure_file(
        request: Request,
        model: str = Form(...),
        content_file_url: str = Form(...),
):
    print(f"Received request for azure file {content_file_url}")

    postback_uri = request.headers.get("LanguageServicePostbackUri")

    random_file_name = uuid.uuid4().hex
    path = Path(f"{MEDIA_DIR}/{random_file_name}")

    try:
        blob_client = BlobClient.from_connection_string(
            AZURE_BLOB_STORAGE_CONNECTION_STRING,
            container_name=AZURE_BLOB_STORAGE_CONTAINER_NAME,
            blob_name=content_file_url
        )
        with open(file=path, mode="wb") as fs:
            download_stream = blob_client.download_blob()
            fs.write(download_stream.readall())

        assert model == "whisper-ch"
        loop = asyncio.get_running_loop()
        loop.create_task(transcribe_post(postback_uri, audio_path=str(path)))
    except Exception as e:
        print(e)
        os.remove(path)


@app.post("/v1/audio/transcriptions/url")
async def transcriptions_url(
        request: Request,
        model: str = Form(...),
        url: str = Form(...)
):
    print(f"Received request for {url}")
    postback_uri = request.headers.get("LanguageServicePostbackUri")

    random_file_name = uuid.uuid4().hex
    path = Path(f"{MEDIA_DIR}/{random_file_name}")

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        assert model == "whisper-ch"
        loop = asyncio.get_running_loop()
        loop.create_task(transcribe_post(postback_uri, audio_path=str(path)))
    except Exception as e:
        print(e)
        os.remove(path)


@app.get("/healthz", status_code=200)
async def health() -> str:
    return "OK"
