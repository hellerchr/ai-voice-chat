# Voice chat

A demo by Christian Heller.

A hands-free voice assistant that runs entirely on your computer. Open the page, start the
conversation and talk. Pause when you are done; talk over the assistant to interrupt it.

## Start

You need [uv](https://docs.astral.sh/uv/) and one of:

- a Mac with Apple Silicon, macOS 14 or newer, 16 GB of memory
- Linux with an NVIDIA GPU (12 GB or more) and a driver for CUDA 13
- Windows with such a GPU, inside [WSL 2](https://learn.microsoft.com/windows/wsl/install) (untested);
  MLX has no GPU support on native Windows

```bash
brew install uv            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/hellerchr/ai-voice-chat.git
cd ai-voice-chat
uv run server.py
```

The page opens at http://127.0.0.1:8791/. Allow the microphone, press **Start conversation** and
talk. **Esc** ends the conversation; **Ctrl+C** stops the server.

The first start downloads the models (8.7 GB, into the Hugging Face cache) and, on Linux, compiles
the GPU kernels once (a few minutes). Later starts take about 10 seconds and work offline.

## Models

| Step | Model | Size |
| --- | --- | --- |
| Hear speech | [Silero VAD](https://huggingface.co/onnx-community/silero-vad) (CPU) | 2 MB |
| Detect the end of a turn | [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) (CPU) | 9 MB |
| Transcribe | [Qwen3-ASR 0.6B](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit), 8-bit | 1.0 GB |
| Reply | [Gemma 4 E4B](https://huggingface.co/NeoRoth/gemma-4-e4b-it-qat-q4_0-mlx), Google's quantization-aware 4-bit weights | 5.7 GB |
| Speak | [Qwen3-TTS 0.6B Base](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit), 8-bit | 2.0 GB |

All three GPU models run on [MLX](https://github.com/ml-explore/mlx): Metal on a Mac, CUDA on Linux.
They use about 8.5 GB of memory together. Each model is pinned to a fixed revision in `models.py`.

## The voice

Qwen3-TTS has no built-in voice here: it copies the voice in `voice/reference.flac`, given its words
in `voice/reference.txt`. For each sentence the model continues the reference recording with the
new text, so it keeps that speaker's timbre and accent.

The reference is a synthetic, accent-free High German voice designed with Qwen3-TTS VoiceDesign. To use another voice, replace both files with 5 to 10 seconds of clear speech
you have the right to use and its exact words. The next start encodes the new reference once
(about 20 seconds) and stores the result as `voice/codes-<hash>.npy`.

## Make it yours

- **Behavior and functions:** `assistant.py` holds the system prompt and the functions the assistant
  may call: a demo weather forecast and a demo calendar, both with made-up data. A function is plain
  Python with type hints and a docstring; `@tool` offers it to the model:

  ```python
  @tool(say="Ich schaue nach dem Wetter.")   # optional: spoken while it runs
  def get_weather(city: str = "Berlin", day: str = "today") -> dict:
      """Weather forecast for a place and day.

      Args:
          city: Name of the city or place.
          day: "today", "tomorrow", a weekday like "saturday", or a date like 2026-10-16.
      """
  ```

  The return value goes back to the model, which continues its answer (up to three calls per reply).
- **Language:** German, English or automatic, chosen on the page.
- **Port:** `VOICE_CHAT_PORT=8800 uv run server.py`. `--no-browser` does not open the page.

## How it works

1. The browser streams the microphone (with echo cancellation) to the server over a WebSocket.
2. Silero VAD hears speech. 0.2 s of speech starts a turn and stops the assistant (0.4 s while it
   speaks, so a cough does not interrupt it).
3. After a 0.3 s pause, Smart Turn judges whether the sentence sounds finished. If it does not, the
   turn ends after 1 s of silence.
4. Qwen3-ASR transcribes the turn. Gemma writes the answer, and each sentence is spoken as soon as it
   is complete, in pieces of 0.32 s.

The answer starts about 0.3 s after you stop talking (measured on an RTX 5080). Gemma keeps the
conversation in its cache and only reads what is new. If you interrupt, the history keeps only the
sentences that were spoken.

## Troubleshooting

- **No microphone:** allow it in the browser's site settings. On a Mac, also allow the browser under
  System Settings › Privacy & Security › Microphone.
- **The assistant interrupts itself:** your speakers are loud enough to get past echo cancellation.
  Use headphones or turn the volume down.
- **Slow or out of memory on a 16 GB Mac:** close other large apps; the models need about 8.5 GB.
- **Port in use:** set `VOICE_CHAT_PORT`.

## Files

| File | Purpose |
| --- | --- |
| `server.py` | Starts everything; serves the page and one WebSocket per conversation |
| `conversation.py` | One conversation: turns, the spoken reply, the history |
| `turns.py` | When a turn starts and ends |
| `models.py` | Downloads, loads and runs the models; the voice |
| `assistant.py` | System prompt and functions |
| `tools.py` | `@tool` |
| `voice/` | The reference voice |
| `web/` | The page: microphone, playback, interruptions |

Tests: `uv run pytest`.

## Licenses

Gemma 4, Qwen3-ASR and Qwen3-TTS are Apache 2.0, Silero VAD is MIT and Smart Turn is BSD-2-Clause.
