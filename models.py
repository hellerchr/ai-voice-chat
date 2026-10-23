"""The models, all on MLX: the Apple GPU on a Mac, CUDA on Linux.

Speech recognition, the reply model and the voice share one worker thread
(`Models.submit`): MLX work must never run on two threads at once. Turn
detection (turns.py) runs on onnxruntime instead, so the user can always
interrupt, even while the GPU is busy.
"""
from __future__ import annotations
import hashlib
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import partial
from pathlib import Path
from collections.abc import Iterator

import numpy as np

import assistant
import tools

# Hugging Face repos at pinned revisions: about 8.7 GB to download, 8.5 GB of memory at most.
MODELS = {
    "asr": ("mlx-community/Qwen3-ASR-0.6B-8bit", "89e96d92ba34aca20b3e29fb10cc284097d1219f"),
    # Google's quantization-aware-trained Gemma 4 E4B on its own 4-bit grid (groups of 32), as MLX weights.
    "llm": ("NeoRoth/gemma-4-e4b-it-qat-q4_0-mlx", "b121896218fc040a25f0e6d38e93c74a332e7ec0"),
    "tts": ("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit", "50f45ef0047cde7e84c2ef04326acb8ada2436a7"),
    "vad": ("onnx-community/silero-vad", "e71cae966052b992a7eca6b17738916ce0eca4ec", "onnx/model.onnx"),
    "turn": ("pipecat-ai/smart-turn-v3", "f766f81d3cfdf7737ac64aad813d91bbfd56bf93", "smart-turn-v3.2-cpu.onnx"),
}
# The voice: Qwen3-TTS clones a few seconds of reference speech (reference.flac, its words in reference.txt).
VOICE = Path(__file__).resolve().parent / "voice"
SAMPLE_RATE = 24000  # of the voice
LANGUAGES = {  # (speech recognition, reply instruction, voice)
    "de": ("German", "Antworte auf Deutsch.", "german"),
    "en": ("English", "Reply in English.", "english"),
    "auto": (None, "Reply in the same language as the user.", "auto"),
}
TOOL_ROUNDS = 3  # function calls per reply
HISTORY = 24  # messages kept; older ones are dropped in one step, which keeps the prompt cache useful


class Cancelled(Exception):
    pass


def download(repo: str, revision: str, file: str | None = None) -> Path:
    """The pinned model (one file of it, if given), downloaded on first use and read offline afterwards."""
    from huggingface_hub import hf_hub_download, snapshot_download
    fetch = partial(hf_hub_download, repo, file) if file else partial(snapshot_download, repo)
    try:
        return Path(fetch(revision=revision, local_files_only=True))
    except Exception:  # noqa: BLE001 - not downloaded yet
        return Path(fetch(revision=revision))


class Models:
    def __init__(self):
        self.ready, self.error, self.status = False, None, "Starting"
        self.gpu = ThreadPoolExecutor(1, thread_name_prefix="mlx")

    def submit(self, function, *args):
        """Run on the models' thread; returns a concurrent.futures.Future."""
        return self.gpu.submit(function, *args)

    def load(self):
        """Download (first start only), load and warm up all models. Runs on the models' thread."""
        try:
            self._load()
            print("Ready.", flush=True)
        except Exception as error:  # noqa: BLE001 - shown in the page
            traceback.print_exc()
            self.error, self.status = f"{type(error).__name__}: {error}", "Failed to load"

    def _load(self):
        from mlx_audio.stt import load as load_stt
        from mlx_audio.tts import load as load_tts
        from turns import TurnModels

        paths = {}
        for name, label in (("vad", "turn detection"), ("turn", "turn detection"), ("asr", "speech recognition"),
                            ("llm", "Gemma"), ("tts", "the voice")):
            self.status = f"Downloading {label}"
            paths[name] = download(*MODELS[name])
        self.status = "Loading"
        self.turns = TurnModels(paths["vad"], paths["turn"])
        self.asr = load_stt(str(paths["asr"]))
        self.chat = Chat(paths["llm"])
        self.tts = load_tts(str(paths["tts"]))
        self._load_voice()
        # The first run of each model compiles its GPU kernels (on CUDA once per machine, then cached).
        self.status = "Warming up"
        speech = b"".join(self.speak("Hallo, ich bin bereit.", "de"))
        samples = np.frombuffer(speech, "<i2").astype(np.float32) / 32768
        self.transcribe(samples[::3].copy(), "de")  # 24 kHz to 16 kHz, good enough for a warm-up
        list(self.reply("Sag Hallo.", [], "de", max_tokens=4))
        self.ready, self.status = True, "Ready"

    def _load_voice(self):
        """The reference speech, and its codec codes, encoded once and kept next to it.

        Encoding takes several GB of memory for a few seconds, more than the whole conversation,
        so it runs on the CPU and only when the reference is new.
        """
        import mlx.core as mx
        from mlx_audio.utils import load_audio
        audio = VOICE / "reference.flac"
        self.reference = load_audio(str(audio), sample_rate=SAMPLE_RATE)
        self.reference_text = (VOICE / "reference.txt").read_text().strip()
        codes_file = VOICE / f"codes-{hashlib.sha256(audio.read_bytes()).hexdigest()[:12]}.npy"
        if not codes_file.exists():
            self.status = "Encoding the voice"
            with mx.stream(mx.cpu):
                codes = self.tts.speech_tokenizer.encode(self.reference[None, None])
                np.save(codes_file, np.array(codes))
        codes = mx.array(np.load(codes_file))
        tokenizer = self.tts.speech_tokenizer
        tokenizer.encode = lambda audio: codes  # only ever asked to encode the reference
        tokenizer.decoder = PrimedDecoder(tokenizer.decoder, codes)

    def transcribe(self, samples: np.ndarray, language: str) -> str:
        """16 kHz mono float32 audio to text."""
        return self.asr.generate(samples, language=LANGUAGES[language][0]).text.strip()

    def reply(self, text: str, history: list[dict], language: str, cancel=None, max_tokens=200) -> Iterator[str]:
        """The reply as text chunks. Function calls run in between and their results go back to the model."""
        system = (f"{assistant.SYSTEM_PROMPT}\nToday is {date.today():%A, %Y-%m-%d}. "  # stable all day: cacheable
                  + LANGUAGES[language][1])
        messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": text}]
        tokenizer = self.chat.tokenizer
        for round in range(TOOL_ROUNDS + 1):
            offer = tools.declarations() if round < TOOL_ROUNDS else None
            said, call, thinking = "", None, False
            if round:
                yield " "  # the continuation after a function call starts a new sentence
            for token, piece in self.chat.generate(messages, offer, max_tokens):
                if cancel and cancel.is_set():
                    raise Cancelled()
                if thinking:  # the model's hidden reasoning is never spoken
                    thinking = token != self.chat.think_end
                elif token == self.chat.think_start:
                    thinking = True
                elif call is not None:  # a function call: collected, not spoken
                    if token == self.chat.call_end:
                        break
                    call += piece
                elif token == self.chat.call_start:
                    call = ""
                else:
                    said += piece
                    yield piece
            if call is None:
                return
            try:
                parsed = tokenizer.tool_parser(call, offer)
                name, arguments = parsed["name"], parsed.get("arguments") or {}
            except Exception:  # noqa: BLE001 - a garbled call ends the reply with what was said
                return
            tool = tools.TOOLS.get(name)
            if tool and tool.say and not said.strip():
                said = tool.say + " "
                yield said
            result = tools.run(name, arguments)
            messages += [
                {"role": "assistant", "content": said,
                 "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]},
                {"role": "tool", "name": name, "content": json.dumps(result, ensure_ascii=False)},
            ]

    def speak(self, text: str, language: str, cancel=None) -> Iterator[bytes]:
        """16-bit PCM pieces (24 kHz) of one sentence, each as soon as it is decoded."""
        pieces = self.tts.generate(text, ref_audio=self.reference, ref_text=self.reference_text,
                                   lang_code=LANGUAGES[language][2], stream=True, streaming_interval=0.32)
        for count, piece in enumerate(pieces):
            if cancel and cancel.is_set():
                raise Cancelled()
            audio = np.array(piece.audio, dtype=np.float32)
            if count == 0:  # a 10 ms fade-in keeps the first sound from clicking
                fade = min(240, len(audio))
                audio[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            yield (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


class PrimedDecoder:
    """The voice's codec decoder, but each streamed sentence is decoded after the reference speech.

    mlx-audio decodes a whole cloned sentence after the reference, yet starts a streamed one from
    silence, so streamed sentences often began lower and rougher. Primed, both decode the same.
    Decoding the reference takes longer than the sentence itself, so it runs once and each
    sentence starts from a copy of the state it left.
    """

    STATE = ("_buffer", "_overflow")  # what the decoder's streaming layers carry from step to step

    def __init__(self, decoder, reference_codes):
        import mlx.core as mx
        self.decoder = decoder
        self.module = module = decoder.streaming_step.__self__  # under mx.compile's wrapper, which holds no state
        module.reset_streaming_state()
        mx.eval(module.streaming_step(reference_codes))  # its audio is dropped; the state carries over
        # Cut to their length, the next step writes into new arrays and never into these.
        self.cache = [(c.keys[..., :c.offset, :], c.values[..., :c.offset, :], c.offset)
                      for c in module._transformer_cache]
        self.layers = [(layer, {name: getattr(layer, name) for name in self.STATE if hasattr(layer, name)})
                       for _, layer in module.named_modules() if hasattr(layer, "reset_state")]

    def __getattr__(self, name):
        return getattr(self.decoder, name)

    def __call__(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def reset_streaming_state(self):
        from mlx_audio.lm.models.cache import KVCache
        caches = []
        for keys, values, offset in self.cache:
            cache = KVCache()
            cache.keys, cache.values, cache.offset = keys, values, offset
            caches.append(cache)
        self.module._transformer_cache = caches
        for layer, state in self.layers:
            for name, value in state.items():
                setattr(layer, name, value)


class Chat:
    """mlx-lm with a prompt cache: a new turn only processes what changed since the last one."""

    def __init__(self, path: Path):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        self.model, self.tokenizer = load(str(path))
        self.sampler = make_sampler(temp=0.5, top_p=0.95, top_k=64)  # a little calmer than Gemma's default 1.0
        self.cache, self.cached = None, []  # the KV cache and the token ids it holds
        # The tokens that open and close the model's reasoning and its function calls (None if it has none).
        t = self.tokenizer
        self.think_start, self.think_end = (t.think_start_tokens[0], t.think_end_tokens[-1]) if t.has_thinking else (None, None)
        self.call_start, self.call_end = ((t.tool_call_start_tokens[0], t.tool_call_end_tokens[-1])
                                          if t.has_tool_calling else (None, None))

    def new_cache(self):
        """The model's KV cache, but growing for every layer: Gemma's sliding-window layers get their
        window from the attention mask, and unlike a rotating cache this one can be trimmed to the
        part of the prompt that did not change, also late in a long conversation."""
        from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
        return [KVCache() if isinstance(c, RotatingKVCache) else c for c in make_prompt_cache(self.model)]

    def generate(self, messages, offer, max_tokens) -> Iterator[tuple[int, str]]:
        """(token id, text) as the model writes the reply to `messages`."""
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
        ids = self.tokenizer.apply_chat_template(messages, tools=offer, add_generation_prompt=True, enable_thinking=False)
        common = 0
        for old, new in zip(self.cached, ids[:-1]):  # always process at least one token
            if old != new:
                break
            common += 1
        if self.cache is not None and can_trim_prompt_cache(self.cache):
            trim_prompt_cache(self.cache, self.cache[0].offset - common)
        else:
            self.cache, common = self.new_cache(), 0
        self.cached = list(ids)
        try:
            for response in stream_generate(self.model, self.tokenizer, ids[common:], max_tokens=max_tokens,
                                            sampler=self.sampler, prompt_cache=self.cache):
                self.cached.append(response.token)
                yield response.token, response.text
        finally:
            if can_trim_prompt_cache(self.cache):  # it may hold one token more or less than was yielded
                self.cached = self.cached[:self.cache[0].offset]
                trim_prompt_cache(self.cache, self.cache[0].offset - len(self.cached))
            else:
                self.cache = None
