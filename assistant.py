"""The assistant: its system prompt and the functions it may call. Edit freely.

Every @tool function here is offered to the model (see tools.py for the format).
The two functions below return made-up demo data.
"""
from __future__ import annotations
import hashlib
from datetime import date, timedelta

from tools import tool

SYSTEM_PROMPT = """You are a helpful personal voice assistant. Your replies are read aloud.
Give a direct, concise answer, usually one or two short sentences.
Start with a complete short sentence of at most twelve words.
End every sentence with punctuation.
Use plain spoken language, without markdown, bullets, emojis, or reasoning tags.
Do not repeat the question. When a function can answer, call it right away, before saying anything;
never ask for permission or for details you can assume, like the day (today).
In German, address the user informally (du).
Speak warmly and simply. You are an AI with a synthetic voice, not a real person."""


WEEKDAYS = [("monday", "montag"), ("tuesday", "dienstag"), ("wednesday", "mittwoch"), ("thursday", "donnerstag"),
            ("friday", "freitag"), ("saturday", "samstag", "weekend", "wochenende"), ("sunday", "sonntag")]


def _day(day: str) -> date:
    """"today", "tomorrow", a weekday (its next occurrence) or an ISO date; English or German."""
    today, word = date.today(), day.lower().removeprefix("this ").removeprefix("next ").strip()
    offsets = {"today": 0, "heute": 0, "tomorrow": 1, "morgen": 1, "übermorgen": 2}
    if word in offsets:
        return today + timedelta(days=offsets[word])
    for weekday, names in enumerate(WEEKDAYS):
        if word in names:
            return today + timedelta(days=(weekday - today.weekday()) % 7)
    try:
        return date.fromisoformat(day)
    except ValueError:
        raise ValueError(f'Unknown day {day!r}: use "today", "tomorrow", a weekday or a date like 2026-10-16') from None


@tool(say="Ich schaue nach dem Wetter.")
def get_weather(city: str = "Berlin", day: str = "today") -> dict:
    """Weather forecast for a place and day.

    Args:
        city: Name of the city or place.
        day: "today", "tomorrow", a weekday like "saturday", or a date like 2026-10-16.
    """
    when = _day(day)
    seed = hashlib.sha256(f"{city.lower()}{when}".encode()).digest()  # the same answer for the same question
    low = 2 + seed[0] % 12
    return {"place": city, "day": when.isoformat(),
            "weather": ["sonnig", "leicht bewölkt", "bewölkt", "regnerisch", "windig"][seed[1] % 5],
            "min_celsius": low, "max_celsius": low + 3 + seed[2] % 8,
            "rain_probability_percent": seed[3] % 10 * 10}


APPOINTMENTS = {  # days from today: (start, end, title, place)
    0: [("09:30", "10:00", "Team-Meeting", "Videocall"), ("13:00", "14:00", "Mittagessen mit Anna", "Café Kranz"),
        ("17:30", "18:30", "Zahnarzt", "Praxis Dr. Weber")],
    1: [("08:00", "09:00", "Joggen", "Stadtpark"), ("15:00", "16:30", "Projektbesprechung", "Büro, Raum 3")],
    2: [("19:00", "22:00", "Geburtstag von Max", "Bei Max zu Hause")],
}


@tool(say="Ich schaue in deinen Kalender.")
def get_calendar(day: str = "today", days: int = 1) -> dict:
    """The user's appointments, starting on a day.

    Args:
        day: "today", "tomorrow", a weekday like "saturday", or a date like 2026-10-16.
        days: How many days to include, starting with that day (1 to 14).
    """
    first = _day(day)
    events = []
    for offset in range(max(1, min(days, 14))):
        when = first + timedelta(days=offset)
        for start, end, title, place in APPOINTMENTS.get((when - date.today()).days, []):
            events.append({"date": when.isoformat(), "start": start, "end": end, "title": title, "location": place})
    return {"from": first.isoformat(), "days": max(1, min(days, 14)), "events": events}
