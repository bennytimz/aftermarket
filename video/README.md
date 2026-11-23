# Aftermarket: 60-second video

`make video` (or `python video/build.py`) writes `video/aftermarket_60s.mp4` (1920x1080, 30 fps, H.264 + AAC,
under 60 s) and `video/contact_sheet.png` (a frame every 3 s). Neither is committed.

- `voice.py`: the voice-over lines and edge-tts settings (voice, rate). `python video/voice.py samples` renders the
  first line in the three candidate voices; `python video/build.py --revoice` re-renders the full voice-over.
- `data.py`: every number and figure series, read from the paper's outputs (`paper/derived/numbers.json`, the
  Fig. 2/5/6 sources). It checks the numbers the script uses against the paper and fails if a figure disagrees.
- `music.py`: original background music synthesised in numpy (no samples or downloads).
- `scene.html`: the scenes; `render(t)` draws the frame at time t, so the render is deterministic.
- `build.py`: timeline from the real voice timings, captions, mix (music ~22 dB under the voice, ducked, faded,
  -14 LUFS), frames via Playwright, encode, contact sheet. `--preview 1.5,12,30 DIR` writes stills only.
- `fonts/`: Inter (SIL Open Font License, `fonts/LICENSE.txt`).

Setup: `pip install -r video/requirements.txt && python -m playwright install chromium`.
