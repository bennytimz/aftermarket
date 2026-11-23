"""Voice-over with edge-tts: one MP3 per scene line plus word timings (for the burned-in captions).

    python video/voice.py samples        # first line in the three candidate voices -> video/build/sample_*.mp3
    python video/voice.py full [VOICE]   # every line -> video/build/vo_<i>.mp3 + vo_<i>.json (word timings)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import edge_tts

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
CANDIDATES = ["en-US-AriaNeural", "en-US-JennyNeural", "en-GB-SoniaNeural"]
VOICE = "en-GB-SoniaNeural"
RATE = "+3%"  # -5% was asked for, but no voice then fits 60 s with pauses (speech alone 59.9 s with Sonia)

# one entry per scene; numbers spoken here are checked against paper/derived/numbers.json in build.py
LINES = [
    "Every AI safety fix is tested on the day it's released. But that's not how models live.",
    "People fine-tune them, compress them, strip their templates. "
    "So we asked: do public safety fixes survive what happens next?",
    "Aftermarket puts six public fixes for Tiny Aya through six everyday changes, "
    "scored by two automatic judges and checked by humans.",
    "Most fixes break. One cheap fine-tune pushes harmful replies to around ninety percent. "
    "And the attacked models stay just as capable: unsafe, not broken.",
    "The tools that measure safety can mislead too. Llama Guard barely agrees with human labels. "
    "And through back-translation, Yoruba looks far less safe, but a native speaker confirms only four in twenty.",
    "So: test the forms people actually run, and check your judges against humans, in every language.",
    "Aftermarket. One command. Under a dollar per audit.",
]


async def speak(text: str, voice: str, mp3: Path, words_json: Path | None = None) -> None:
    com = edge_tts.Communicate(text, voice, rate=RATE, boundary="WordBoundary")
    words = []
    with mp3.open("wb") as f:
        async for chunk in com.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":  # offsets in 100-ns ticks
                words.append({"text": chunk["text"], "start": chunk["offset"] / 1e7,
                              "end": (chunk["offset"] + chunk["duration"]) / 1e7})
    if words_json is not None:
        words_json.write_text(json.dumps(words, indent=1), encoding="utf-8")


async def main(argv: list[str]) -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    mode = argv[0] if argv else "full"
    if mode == "samples":
        for v in CANDIDATES:
            await speak(LINES[0], v, BUILD / f"sample_{v}.mp3")
            print("wrote", BUILD / f"sample_{v}.mp3")
        return
    voice = argv[1] if len(argv) > 1 else VOICE
    for i, line in enumerate(LINES):
        await speak(line, voice, BUILD / f"vo_{i}.mp3", BUILD / f"vo_{i}.json")
    (BUILD / "vo_voice.txt").write_text(voice, encoding="utf-8")
    print(f"wrote {len(LINES)} lines with {voice}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
