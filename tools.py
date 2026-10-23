"""Expose Python functions to the language model with @tool.

Write a function with type hints and a docstring: its first paragraph says what
it does, an "Args:" section has one "name: description" line per parameter.
Decorate it with @tool and the model can call it:

    @tool(say="Ich schaue nach.")
    def get_weather(city: str, day: str = "today") -> dict:
        '''Weather forecast for a place.

        Args:
            city: Name of the city, e.g. "Berlin".
            day: "today", "tomorrow" or a date like 2026-10-16.
        '''

`say` is spoken while the function runs, unless the model already said
something. `timeout` (seconds) bounds a slow function. The return value
(anything JSON can represent) goes back to the model.
"""
from __future__ import annotations
import concurrent.futures
import inspect
import re
import typing
from dataclasses import dataclass
from collections.abc import Callable

TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


@dataclass
class Tool:
    function: Callable
    declaration: dict
    say: str | None = None
    timeout: float = 10


TOOLS: dict[str, Tool] = {}


def tool(function=None, *, say: str | None = None, timeout: float = 10):
    """Register a function as a tool (options: see the module docstring)."""
    def register(fn):
        TOOLS[fn.__name__] = Tool(fn, _declaration(fn), say, timeout)
        return fn
    return register(function) if function else register


def _declaration(fn) -> dict:
    doc = inspect.getdoc(fn) or ""
    summary, _, rest = doc.partition("\nArgs:")
    parameters = inspect.signature(fn).parameters
    described, current = {}, None
    for line in rest.splitlines():
        match = re.match(r"\s*(\w+)\s*:\s*(.*)", line)
        if match and match.group(1) in parameters:
            current = match.group(1)
            described[current] = match.group(2).strip()
        elif current and line.strip():
            described[current] += " " + line.strip()
    hints = typing.get_type_hints(fn)
    properties, required = {}, []
    for name, parameter in parameters.items():
        annotation = hints.get(name, str)
        schema = {"description": described.get(name, "")}
        if typing.get_origin(annotation) is typing.Literal:
            schema |= {"type": "string", "enum": list(typing.get_args(annotation))}
        else:
            schema["type"] = TYPES.get(typing.get_origin(annotation) or annotation, "string")
        properties[name] = schema
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "function", "function": {
        "name": fn.__name__, "description": " ".join(summary.split()),
        "parameters": {"type": "object", "properties": properties, "required": required}}}


def declarations() -> list[dict]:
    return [t.declaration for t in TOOLS.values()]


def run(name: str, arguments: dict):
    """Call a tool; failures and timeouts become {"error": ...} so the model can explain them."""
    if name not in TOOLS:
        return {"error": f"Unknown function {name}"}
    worker = concurrent.futures.ThreadPoolExecutor(1)
    try:
        return worker.submit(TOOLS[name].function, **arguments).result(timeout=TOOLS[name].timeout)
    except concurrent.futures.TimeoutError:
        return {"error": f"{name} did not answer within {TOOLS[name].timeout:g} seconds"}
    except Exception as error:  # noqa: BLE001 - reported to the model, not raised
        return {"error": f"{type(error).__name__}: {error}"}
    finally:
        worker.shutdown(wait=False)  # a stuck function keeps its thread, not the reply
