"""One conversation: turns from the detector, replies spoken sentence by sentence, and the history.

When the user starts speaking, the running reply stops. When their turn ends,
the reply runs on the models' thread: transcribe, write the reply, and speak
each sentence as soon as it is complete.
"""
from __future__ import annotations
import asyncio
import re
import threading

import numpy as np

from models import HISTORY, SAMPLE_RATE, Cancelled

SENTENCE_PAUSE = 0.12  # seconds of silence between sentences, as a reader would make


class Conversation:
    def __init__(self, models, send):
        self.models, self.send = models, send  # send(event) queues one event for the browser
        self.loop = asyncio.get_running_loop()
        self.detector, self.language = None, "de"
        self.history: list[dict] = []  # only touched on the models' thread
        self.cancel = None  # of the current reply

    def listen(self, language):
        self.language = language
        self.detector = self.detector or self.models.turns.detector()

    def mute(self):
        self.interrupt()
        self.detector = None

    def playback(self, active):
        if self.detector:
            self.detector.assistant_speaking = active

    def interrupt(self):
        if self.cancel:
            self.cancel.set()
            self.cancel = None

    async def reset(self):
        self.interrupt()
        await asyncio.wrap_future(self.models.submit(self.history.clear))

    async def hear(self, pcm: bytes):
        """Feed microphone PCM (16 kHz, 16 bit) and act on the turn events it produces."""
        if self.detector is None or len(pcm) % 2:
            return
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        for kind, *audio in await asyncio.to_thread(self.detector.feed, samples):
            self.interrupt()
            if kind == "start":
                self.send({"type": "speech"})
            else:
                self.respond(audio[0])

    def respond(self, samples):
        cancel = self.cancel = threading.Event()
        self.send({"type": "thinking"})

        def emit(event):  # called on the models' thread; drops events of a reply that was interrupted
            self.loop.call_soon_threadsafe(lambda: None if cancel.is_set() else self.send(event))

        self.models.submit(run_turn, self.models, samples, self.history, self.language, cancel, emit)


def run_turn(models, samples, history, language, cancel, emit):
    """Transcribe, reply and speak one turn; then add it to the history (on the models' thread)."""
    if cancel.is_set():
        return
    if len(samples) < 0.3 * 16000 or float(np.sqrt(np.mean(samples ** 2))) < 0.002:
        return emit({"type": "empty"})
    transcript = models.transcribe(samples, language)
    if not transcript or cancel.is_set():
        return emit({"type": "empty"})
    emit({"type": "transcript", "text": transcript})
    spoken, sentences = [], Sentences()

    def say(sentence):
        for pcm in models.speak(sentence, language, cancel):
            emit({"type": "audio", "pcm": pcm})
        emit({"type": "audio", "pcm": bytes(2 * round(SENTENCE_PAUSE * SAMPLE_RATE))})
        spoken.append(sentence)

    try:
        for chunk in models.reply(transcript, history, language, cancel):
            emit({"type": "reply", "text": chunk})
            for sentence in sentences.add(chunk):
                say(sentence)
        for sentence in sentences.finish():
            say(sentence)
        emit({"type": "done"})
    except Cancelled:
        spoken.append("… [interrupted by the user]")
    except Exception as error:  # noqa: BLE001 - shown in the page, the conversation goes on
        emit({"type": "error", "message": f"{type(error).__name__}: {error}"})
    if spoken:
        history += [{"role": "user", "content": transcript}, {"role": "assistant", "content": " ".join(spoken)}]
        if len(history) > HISTORY:
            del history[:len(history) - HISTORY // 2]


class Sentences:
    """Cuts streamed text into complete sentences, so each can be spoken while the next is written."""

    # A period after a number, a single letter or a common abbreviation does not end a sentence.
    NOT_AN_END = re.compile(r"(?:\b(?:Dr|Prof|Mr|Mrs|Ms|bzw|ca|usw|etc|z\.\s?B|e\.g|i\.e)|\b\d+|\b[a-zA-Z])\.$", re.IGNORECASE)

    def __init__(self, longest=200):
        self.pending, self.longest = "", longest

    def add(self, text) -> list[str]:
        self.pending += text
        sentences = []
        while True:
            end = next((m.end() for m in re.finditer(r"[.!?…][\"”’']?(?=\s)", self.pending)
                        if not self.NOT_AN_END.search(self.pending[:m.end()])), 0)
            if not end and len(self.pending) >= self.longest:
                end = self.pending.rfind(",", 25, self.longest) + 1  # only a clause may split a long sentence
            if not end:
                return sentences
            sentence, self.pending = self.pending[:end].strip(), self.pending[end:].lstrip()
            if sentence:
                sentences.append(sentence)

    def finish(self) -> list[str]:
        sentence, self.pending = self.pending.strip(), ""
        return [sentence] if sentence else []
