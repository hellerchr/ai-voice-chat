import numpy as np

from turns import FRAME, GIVE_UP, PAUSE, RATE, START, START_WHILE_SPEAKING, TurnDetector

STEP = FRAME / RATE


def frames(seconds, level):
    """Audio whose VAD probability (the fake below) is `level`."""
    return np.full(round(seconds / STEP) * FRAME, level, np.float32)


def detector(complete=1.0):
    asked = []
    turns = TurnDetector(lambda frame: float(frame[0]), lambda audio: asked.append(len(audio)) or complete)
    return turns, asked


def test_short_noise_does_not_start_a_turn():
    turns, _ = detector()
    assert turns.feed(frames(START - 2 * STEP, 0.9)) == []
    assert turns.feed(frames(1, 0.0)) == []


def test_finished_sentence_ends_after_a_short_pause():
    turns, asked = detector(complete=0.9)
    assert turns.feed(frames(1.0, 0.9)) == [("start",)]
    events = turns.feed(frames(PAUSE + 2 * STEP, 0.0))
    assert [e[0] for e in events] == ["end"] and len(asked) == 1
    assert len(events[0][1]) > RATE  # the whole turn, with the pre-roll before its start


def test_unfinished_sentence_waits_for_the_long_pause():
    turns, asked = detector(complete=0.1)
    turns.feed(frames(1.0, 0.9))
    assert turns.feed(frames(GIVE_UP - 3 * STEP, 0.0)) == [] and len(asked) == 1
    assert [e[0] for e in turns.feed(frames(4 * STEP, 0.0))] == ["end"]


def test_speaking_again_after_a_pause_asks_again():
    turns, asked = detector(complete=0.1)
    turns.feed(frames(1.0, 0.9))
    turns.feed(frames(PAUSE + 2 * STEP, 0.0))
    turns.feed(frames(0.5, 0.9))
    turns.feed(frames(PAUSE + 2 * STEP, 0.0))
    assert len(asked) == 2


def test_interrupting_the_assistant_needs_longer_speech():
    turns, _ = detector()
    turns.assistant_speaking = True
    assert turns.feed(frames(START + STEP, 0.9)) == []
    assert turns.feed(frames(START_WHILE_SPEAKING - START, 0.9)) == [("start",)]
