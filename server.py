"""The local web server: the page, and one WebSocket per conversation.

    uv run server.py               # downloads the models on the first start, then opens the page
    uv run server.py --no-browser
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from conversation import Conversation
from models import Models

WEB = Path(__file__).resolve().parent / "web"
PORT = int(os.environ.get("VOICE_CHAT_PORT", "8791"))
LOCAL_ORIGINS = (f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}")
models = Models()


@asynccontextmanager
async def lifespan(app):
    models.submit(models.load)  # in the background; the page shows the progress
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def revalidate(request, call_next):
    """Browsers must check for a newer page and script instead of running a cached old version."""
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")
    return response


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/status")
def status():
    return {"ready": models.ready, "status": models.status, "error": models.error}


@app.websocket("/ws")
async def conversation(ws: WebSocket):
    """Microphone PCM (binary messages) in; events (JSON) and reply audio (binary, 24 kHz PCM) out.

    Commands: {"type": "listen", "language": "de" | "en" | "auto"}, {"type": "mute"},
    {"type": "playback", "active": bool} (whether the assistant is audible), {"type": "reset"}.
    """
    if ws.headers.get("origin", LOCAL_ORIGINS[0]) not in LOCAL_ORIGINS:
        await ws.close(code=1008)  # another website must not use this computer's microphone stream
        return
    await ws.accept()
    events = asyncio.Queue()

    async def send_events():
        while True:
            event = await events.get()
            if "pcm" in event:
                await ws.send_bytes(event["pcm"])
            else:
                await ws.send_json(event)

    sender = asyncio.create_task(send_events())
    talk = Conversation(models, events.put_nowait)
    try:
        while (message := await ws.receive())["type"] != "websocket.disconnect":
            if message.get("bytes") is not None:
                await talk.hear(message["bytes"])
                continue
            try:
                command = json.loads(message.get("text") or "{}")
            except ValueError:
                continue
            action = command.get("type") if isinstance(command, dict) else None
            if action == "listen":
                language = command.get("language")
                if models.ready and language in ("de", "en", "auto"):
                    talk.listen(language)
                    events.put_nowait({"type": "listening"})
                else:
                    events.put_nowait({"type": "error", "message": models.error or models.status})
            elif action == "mute":
                talk.mute()
            elif action == "playback":
                talk.playback(bool(command.get("active")))
            elif action == "reset":
                await talk.reset()
    finally:
        talk.mute()
        sender.cancel()


app.mount("/static", StaticFiles(directory=WEB), name="static")


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Local hands-free voice chat")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    if not parser.parse_args().no_browser:
        threading.Timer(1.5, webbrowser.open, [f"http://127.0.0.1:{PORT}/"]).start()
    print(f"Voice chat on http://127.0.0.1:{PORT}/  (Ctrl+C stops it)", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT, ws_max_size=4 * 1024 * 1024, log_level="warning")
