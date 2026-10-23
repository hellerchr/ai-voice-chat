"""The WebSocket conversation with fake models: turns, replies, interruptions and history."""
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import server
from models import Cancelled

START, END = 1, 2  # the first PCM sample tells the fake detector what happened
SPEECH = np.ones(8000, np.float32) * .1


class FakeDetector:
    assistant_speaking = False

    def feed(self, samples):
        return [{START: ("start",), END: ("end", SPEECH)}[round(samples[0] * 32768)]]


def pcm(code):
    return np.array([code] + [0] * 511, dtype="<i2").tobytes()


class FakeModels:
    ready, status, error = True, "Ready", None

    def __init__(self):
        self.worker = ThreadPoolExecutor(1)
        self.turns = SimpleNamespace(detector=FakeDetector)
        self.histories = []
        self.speaking = threading.Event()  # set: speech blocks until the test releases it
        self.release = threading.Event()

    def submit(self, function, *args):
        return self.worker.submit(function, *args)

    def load(self):
        pass

    def transcribe(self, samples, language):
        return "Frage"

    def reply(self, text, history, language, cancel=None):
        self.histories.append([m["content"] for m in history])
        yield "Erster Satz. "
        yield "Zweiter Satz."

    def speak(self, text, language, cancel=None):
        if self.speaking.is_set():
            self.release.wait(5)
            if cancel.is_set():
                raise Cancelled()
        yield b"\x01\x00" * 100


@pytest.fixture
def client(monkeypatch):
    models = FakeModels()
    monkeypatch.setattr(server, "models", models)
    with TestClient(server.app) as test:
        yield test, models


@contextmanager
def connect(test):
    with test.websocket_connect("/ws", headers={"origin": "http://127.0.0.1:8791"}) as ws:
        ws.send_json({"type": "listen", "language": "de"})
        assert ws.receive_json() == {"type": "listening"}
        yield ws


def receive_until(ws, kind):
    """Events up to `kind`, with each audio message as 'audio'."""
    events = []
    while True:
        message = ws.receive()
        event = "audio" if message.get("bytes") is not None else server.json.loads(message["text"])
        events.append(event)
        if event != "audio" and event["type"] == kind:
            return events


def test_a_turn_is_transcribed_answered_and_spoken_sentence_by_sentence(client):
    test, models = client
    with connect(test) as ws:
        ws.send_bytes(pcm(START))
        assert ws.receive_json() == {"type": "speech"}
        ws.send_bytes(pcm(END))
        events = receive_until(ws, "done")
        assert events[:3] == [{"type": "thinking"}, {"type": "transcript", "text": "Frage"},
                              {"type": "reply", "text": "Erster Satz. "}]
        assert events.count("audio") == 4  # two sentences, each followed by a short pause
        ws.send_bytes(pcm(END))
        receive_until(ws, "done")
        assert models.histories[1] == ["Frage", "Erster Satz. Zweiter Satz."]
        ws.send_json({"type": "reset"})
        ws.send_bytes(pcm(END))
        receive_until(ws, "done")
        assert models.histories[2] == []


def test_speaking_interrupts_the_reply_and_the_history_says_so(client):
    test, models = client
    with connect(test) as ws:
        models.speaking.set()
        ws.send_bytes(pcm(END))
        assert receive_until(ws, "reply")[-1] == {"type": "reply", "text": "Erster Satz. "}
        ws.send_bytes(pcm(START))
        assert receive_until(ws, "speech")[-1] == {"type": "speech"}
        models.release.set()
        models.speaking.clear()
        ws.send_bytes(pcm(END))
        events = receive_until(ws, "done")
        assert "audio" in events and {"type": "reply", "text": "Zweiter Satz."} in events
        assert models.histories[1] == ["Frage", "… [interrupted by the user]"]


def test_audio_is_ignored_until_listening(client):
    test, _ = client
    with test.websocket_connect("/ws", headers={"origin": "http://127.0.0.1:8791"}) as ws:
        ws.send_bytes(pcm(START))
        ws.send_json({"type": "listen", "language": "xx"})
        assert ws.receive_json()["type"] == "error"


def test_another_website_cannot_connect(client):
    test, _ = client
    with pytest.raises(WebSocketDisconnect):
        with test.websocket_connect("/ws", headers={"origin": "https://example.com"}) as ws:
            ws.receive_json()
