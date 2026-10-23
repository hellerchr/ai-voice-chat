"""Hands-free turn taking: Silero VAD hears speech, Smart Turn judges whether a sentence is finished.

Audio arrives as a continuous 16 kHz stream. START seconds of speech start a
turn (and interrupt the assistant). After PAUSE seconds of silence Smart Turn
looks at the last eight seconds: if the user sounds finished the turn ends
right away, otherwise after GIVE_UP seconds of silence.

Events: ("start",) and ("end", audio).
"""
from __future__ import annotations
from collections import deque

import numpy as np

FRAME = 512  # samples per VAD step at 16 kHz (32 ms)
RATE = 16000
SPEECH, SILENCE = 0.5, 0.35  # VAD probabilities for speech and for silence (hysteresis)
START = 0.2  # seconds of speech that start a turn
START_WHILE_SPEAKING = 0.4  # longer while the assistant talks, so a cough does not interrupt it
PAUSE = 0.3  # seconds of silence before asking Smart Turn
GIVE_UP = 1.0  # seconds of silence that end a turn even if it sounds unfinished
PRE_ROLL = 0.3  # seconds kept before the detected start, so the first syllable is not lost
LONGEST = 30  # seconds; a longer turn is ended


class TurnDetector:
    """Feed 16 kHz float32 samples; returns the events listed in the module docstring."""

    def __init__(self, speech_probability, turn_complete):
        self.speech_probability = speech_probability  # 512 samples -> probability of speech
        self.turn_complete = turn_complete  # the turn's audio -> probability that it is finished
        self.assistant_speaking = False
        self.pending = np.zeros(0, np.float32)
        self.pre_roll = deque(maxlen=round(PRE_ROLL * RATE / FRAME))
        self.turn: list[np.ndarray] | None = None
        self.speech = self.silence = 0.0
        self.asked = False  # Smart Turn already judged this pause

    def feed(self, samples: np.ndarray) -> list[tuple]:
        self.pending = np.concatenate([self.pending, samples])
        events = []
        while len(self.pending) >= FRAME:
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            events += self._step(frame, self.speech_probability(frame))
        return events

    def _step(self, frame, probability) -> list[tuple]:
        step = FRAME / RATE
        if self.turn is None:
            self.pre_roll.append(frame)
            self.speech = self.speech + step if probability >= SPEECH else 0.0
            if self.speech < (START_WHILE_SPEAKING if self.assistant_speaking else START):
                return []
            self.turn, self.silence, self.asked = list(self.pre_roll), 0.0, False
            self.pre_roll.clear()
            return [("start",)]
        self.turn.append(frame)
        if probability >= SPEECH:
            self.silence, self.asked = 0.0, False
        elif probability < SILENCE:
            self.silence += step
        done = self.silence >= GIVE_UP or len(self.turn) * step >= LONGEST
        if not done and self.silence >= PAUSE and not self.asked:
            self.asked = True
            done = self.turn_complete(np.concatenate(self.turn)[-8 * RATE:]) >= 0.5
        if not done:
            return []
        audio, self.turn, self.speech = np.concatenate(self.turn), None, 0.0
        return [("end", audio)]


class TurnModels:
    """Silero VAD and Smart Turn on onnxruntime, shared by all conversations (each detector keeps its own VAD state).

    They run on the CPU, independent of MLX, so a reply on the GPU never delays them.
    """

    def __init__(self, vad_path, turn_path):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.vad = ort.InferenceSession(str(vad_path), options, providers=["CPUExecutionProvider"])
        self.smart_turn = ort.InferenceSession(str(turn_path), options, providers=["CPUExecutionProvider"])
        self.features = WhisperFeatureExtractor(chunk_length=8)

    def detector(self) -> TurnDetector:
        state, context = np.zeros((2, 1, 128), np.float32), np.zeros(64, np.float32)

        def speech_probability(frame):
            nonlocal state, context
            window = np.concatenate([context, frame])[None]  # Silero 5 sees the previous frame's last 64 samples
            probability, state = self.vad.run(None, {"input": window, "state": state, "sr": np.array(RATE)})
            context = frame[-64:]
            return float(probability[0, 0])

        return TurnDetector(speech_probability, self.turn_complete)

    def turn_complete(self, audio: np.ndarray) -> float:
        audio = np.pad(audio, (8 * RATE - len(audio), 0))  # Smart Turn pads at the front
        features = self.features(audio, sampling_rate=RATE, return_tensors="np", padding="max_length",
                                 max_length=8 * RATE, truncation=True, do_normalize=True).input_features
        return float(self.smart_turn.run(None, {"input_features": features.astype(np.float32)})[0].item())
