"""Build video/aftermarket_60s.mp4 (1920x1080, 30 fps, H.264 + AAC, <= 60 s) and video/contact_sheet.png.

    python video/build.py            # voice (cached), data checks, music + mix, frames, encode, contact sheet
    python video/build.py --revoice  # re-render the voice-over first

Steps: voice.py (edge-tts, word timings) -> timeline from the real audio -> data.py (every number from the paper)
-> music.py (original synth) mixed ~22 dB under the voice with ducking, faded, normalised to -14 LUFS -> scene.html
rendered frame by frame with Playwright (deterministic: render(t) per frame) -> ffmpeg.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
OUT = HERE / "aftermarket_60s.mp4"
SHEET = HERE / "contact_sheet.png"
sys.path.insert(0, str(HERE))

import data as vdata  # noqa: E402
import music  # noqa: E402
import voice  # noqa: E402

FPS, W, H = 30, 1920, 1080
SR = music.SR
LEAD, GAP, END_HOLD, MAX_LEN = 0.45, 0.36, 1.45, 59.8
EXTRA_GAP = {1: 0.75}  # after line 1: a beat for the AFTERMARKET title slam
MUSIC_BELOW_DB, DUCK_DB = 22.0, 4.0
MAX_WORDS = 7  # per caption


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


# ----------------------------------------------------------------------------- voice + timeline

def load_voice() -> list[dict]:
    lines = []
    for i, text in enumerate(voice.LINES):
        words = json.loads((BUILD / f"vo_{i}.json").read_text(encoding="utf-8"))
        wav = BUILD / f"vo_{i}.wav"
        run(["ffmpeg", "-y", "-i", str(BUILD / f"vo_{i}.mp3"), "-ac", "1", "-ar", str(SR), "-c:a", "pcm_f32le", str(wav)])
        sr, x = wavfile.read(wav)
        lines.append({"text": text, "words": words, "audio": x.astype(np.float64)})
    return lines


def timeline(lines: list[dict]) -> dict:
    """Place each line so its first word starts GAP after the previous line's last word; shrink gaps if needed."""
    speech = [ln["words"][-1]["end"] - ln["words"][0]["start"] for ln in lines]
    gap = GAP
    extra = sum(EXTRA_GAP.values())
    while LEAD + sum(speech) + gap * (len(lines) - 1) + extra + END_HOLD > MAX_LEN and gap > 0.2:
        gap -= 0.02
    t, starts, offsets = LEAD, [], []
    for i, (ln, sp) in enumerate(zip(lines, speech)):
        starts.append(t)
        offsets.append(t - ln["words"][0]["start"])  # where the line's audio file begins
        t += sp + (gap + EXTRA_GAP.get(i, 0) if i < len(lines) - 1 else 0)
    total = round(t + END_HOLD, 3)
    if total > 60.0:
        raise SystemExit(f"voice-over too long: {total:.2f} s")
    scene = [0.0] + [s - 0.3 for s in starts[1:]]
    return {"starts": starts, "offsets": offsets, "scene": scene, "total": total, "gap": round(gap, 2)}


def tokens(line: str, words: list[dict], offset: float) -> list[dict]:
    """Attach the line's own spelling and punctuation to each timed word (edge-tts strips punctuation)."""
    out, cur = [], 0
    low = line.lower()
    for w in words:
        i = low.find(w["text"].lower(), cur)
        if i < 0:
            out.append({"a": cur, "b": cur, "start": offset + w["start"], "end": offset + w["end"], "t": w["text"]})
            continue
        j = i + len(w["text"])
        while j < len(line) and line[j] in ".,?:;!'’\"":
            j += 1
        out.append({"a": i, "b": j, "start": offset + w["start"], "end": offset + w["end"], "t": line[i:j]})
        cur = j
    return out


def captions(lines: list[dict], tl: dict) -> tuple[list[dict], list[list[dict]]]:
    caps, words_all = [], []
    for k, ln in enumerate(lines):
        tk = tokens(ln["text"], ln["words"], tl["offsets"][k])
        words_all.append([{"t": x["t"], "start": round(x["start"], 3), "end": round(x["end"], 3)} for x in tk])
        # clauses at . ? : (and at commas once a clause has 4+ words), then long clauses split into balanced parts
        clauses: list[list[dict]] = [[]]
        for x in tk:
            clauses[-1].append(x)
            if x["t"][-1:] in ".?:" or (x["t"][-1:] == "," and len(clauses[-1]) >= 4):
                clauses.append([])
        clauses = [c for c in clauses if c]
        merged: list[list[dict]] = []
        for c in clauses:  # a one-word clause ("So:") joins the next one
            if merged and len(merged[-1]) == 1:
                merged[-1] += c
            else:
                merged.append(c)
        clauses = merged
        chunks = []
        for c in clauses:
            parts = -(-len(c) // MAX_WORDS)
            size = -(-len(c) // parts)
            chunks += [c[i:i + size] for i in range(0, len(c), size)]
        for i, c in enumerate(chunks):
            end = chunks[i + 1][0]["start"] if i + 1 < len(chunks) else c[-1]["end"] + 0.35
            caps.append({"text": ln["text"][c[0]["a"]:c[-1]["b"]], "start": round(c[0]["start"] - 0.05, 3),
                         "end": round(end, 3)})
    return caps, words_all


# ----------------------------------------------------------------------------- audio

def smooth_env(x: np.ndarray, attack: float = 0.02, release: float = 0.3) -> np.ndarray:
    a, r = np.exp(-1 / (attack * SR)), np.exp(-1 / (release * SR))
    env = np.empty_like(x)
    e = 0.0
    for i, v in enumerate(np.abs(x)):  # one-pole follower; callers pass the signal at SR/8
        c = a if v > e else r
        e = c * e + (1 - c) * v
        env[i] = e
    return env


def mix_audio(lines: list[dict], tl: dict) -> Path:
    n = int(tl["total"] * SR) + 1
    vo = np.zeros(n)
    for ln, off in zip(lines, tl["offsets"]):
        i0 = int(round(off * SR))
        x = ln["audio"][: n - i0]
        vo[i0:i0 + len(x)] += x
    m = music.compose(tl["total"], tl["scene"])
    m = m[:n]
    # sidechain ducking from the voice envelope (computed at SR/8, then upsampled)
    duck_env = smooth_env(vo[::8], 0.03, 0.25)
    duck_env = np.interp(np.arange(n) / 8, np.arange(len(duck_env)), duck_env)
    speaking = np.zeros(n, bool)  # inside a spoken word, from the edge-tts word timings
    for ln, off in zip(lines, tl["offsets"]):
        for w in ln["words"]:
            speaking[int((off + w["start"]) * SR):int((off + w["end"]) * SR)] = True
    amount = np.clip(duck_env / (np.percentile(duck_env[speaking], 50) + 1e-9), 0, 1)
    m *= (10 ** (-DUCK_DB * amount / 20))[:, None]
    # level: the (ducked) music under speech sits MUSIC_BELOW_DB under the speech, RMS over the spoken words
    v_rms = np.sqrt(np.mean(vo[speaking] ** 2))
    m_rms = np.sqrt(np.mean(m[speaking] ** 2))
    m *= (v_rms / m_rms) * 10 ** (-MUSIC_BELOW_DB / 20)
    print(f"music under speech: {MUSIC_BELOW_DB:.0f} dB below the voice; ducking up to {DUCK_DB:.0f} dB")
    # fades on the music: in over 1 s, out over the last 2 s
    t = np.arange(n) / SR
    m *= np.clip(t / 1.0, 0, 1)[:, None] * np.clip((tl["total"] - t) / 2.0, 0, 1)[:, None]
    mix = m + vo[:, None]
    raw = BUILD / "mix_raw.wav"
    wavfile.write(raw, SR, (mix / max(1.0, np.abs(mix).max() / 0.98)).astype(np.float32))
    # two-pass loudnorm to -14 LUFS
    p1 = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(raw), "-af",
                         "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                        capture_output=True, text=True)
    meas = json.loads(re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p1.stderr, re.S).group(0))
    out = BUILD / "mix.wav"
    af = ("loudnorm=I=-14:TP=-1.5:LRA=11:linear=true:"
          f"measured_I={meas['input_i']}:measured_TP={meas['input_tp']}:measured_LRA={meas['input_lra']}:"
          f"measured_thresh={meas['input_thresh']}:offset={meas['target_offset']}")
    run(["ffmpeg", "-y", "-i", str(raw), "-af", af, "-ar", str(SR), "-c:a", "pcm_s16le", str(out)])
    return out


# ----------------------------------------------------------------------------- frames

async def render_frames(total: float, audio: Path) -> None:
    from playwright.async_api import async_playwright

    nframes = int(round(total * FPS))
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg", "-i", "-",
         "-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
         "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", str(SR),
         "-movflags", "+faststart", "-t", f"{nframes / FPS:.3f}", str(OUT)], stdin=subprocess.PIPE)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        await page.goto((HERE / "scene.html").as_uri())
        await page.evaluate("document.fonts.ready")
        await page.wait_for_function("window.READY === true")
        for f in range(nframes):
            await page.evaluate(f"render({f / FPS:.4f})")
            enc.stdin.write(await page.screenshot(type="jpeg", quality=95))
            if f % 300 == 0:
                print(f"  frame {f}/{nframes}", flush=True)
        await browser.close()
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit("ffmpeg encode failed")


async def preview(times: list[float], out_dir: Path) -> None:
    """Stills at chosen times (layout checks without a full render)."""
    from playwright.async_api import async_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": W, "height": H})
        msgs: list[str] = []
        page.on("console", lambda m: msgs.append(m.text))
        page.on("pageerror", lambda e: msgs.append(f"PAGE ERROR: {e}"))
        await page.goto((HERE / "scene.html").as_uri())
        await page.wait_for_function("window.READY === true")
        for t in times:
            await page.evaluate(f"render({t:.3f})")
            await page.screenshot(path=str(out_dir / f"t{t:05.2f}.png"))
        await browser.close()
    print("\n".join(msgs) or "no console messages", f"-> {len(times)} stills in {out_dir}")


def contact_sheet(total: float) -> None:
    """A frame every 3 s, 4 per row, each labelled with its time."""
    from PIL import Image, ImageDraw, ImageFont

    times = [round(x, 2) for x in np.arange(1.5, total, 3.0)]
    tw, th, cols = 480, 270, 4
    rows = (len(times) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), "black")
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for k, t in enumerate(times):
        f = BUILD / f"sheet_{k:02d}.png"
        run(["ffmpeg", "-y", "-ss", str(t), "-i", str(OUT), "-frames:v", "1", "-vf", f"scale={tw}:{th}", str(f)])
        im = Image.open(f).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 70, 28], fill=(0, 0, 0))
        d.text((8, 4), f"{t:.1f}s", fill="white", font=font)
        sheet.paste(im, ((k % cols) * tw, (k // cols) * th))
    sheet.save(SHEET)
    print(f"contact sheet: {len(times)} frames ({cols}x{rows}) -> {SHEET}")


def main(argv: list[str]) -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    if "--revoice" in argv or not (BUILD / "vo_6.json").exists():
        asyncio.run(voice.main(["full"]))
    lines = load_voice()
    tl = timeline(lines)
    caps, words = captions(lines, tl)
    d = vdata.build()
    if d["diffs"]:
        print("numbers differ from the brief (paper values used):", d["diffs"])
    d.update({"timeline": tl, "captions": caps, "words": words,
              "voice": (BUILD / "vo_voice.txt").read_text(encoding="utf-8").strip()})
    (BUILD / "data.js").write_text("window.DATA = " + json.dumps(d) + ";\n", encoding="utf-8")
    print(f"timeline: total {tl['total']} s, gap {tl['gap']} s, scenes at {[round(s, 2) for s in tl['scene']]}")
    if "--no-render" in argv:
        return
    if "--sheet-only" in argv:
        contact_sheet(float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(OUT)]).stdout))
        return
    if "--preview" in argv:
        times = [float(x) for x in argv[argv.index("--preview") + 1].split(",")]
        asyncio.run(preview(times, Path(argv[argv.index("--preview") + 2])))
        return
    audio = mix_audio(lines, tl)
    print("audio mixed ->", audio)
    asyncio.run(render_frames(tl["total"], audio))
    dur = float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(OUT)]).stdout)
    print(f"video: {OUT} ({dur:.2f} s)")
    if dur > 60.0:
        raise SystemExit("over 60 s")
    contact_sheet(dur)


if __name__ == "__main__":
    main(sys.argv[1:])
