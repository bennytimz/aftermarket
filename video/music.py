"""Original background music, synthesised in numpy (no samples, nothing downloaded).

A calm ambient pad (soft additive voices, slow envelopes, a little reverb) with a gentle arpeggiated pulse. It
lifts at the findings (brighter pad, louder pulse) and resolves to the tonic at the end card.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 48000


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


# D major colour: Dmaj9, Bm9, Gmaj9, A6/9sus, Em9, and a plain D to resolve
CHORDS = {
    "D": [50, 57, 61, 64, 66], "Bm": [47, 54, 57, 61, 62], "G": [43, 50, 54, 57, 59],
    "A": [45, 52, 59, 62, 64], "Em": [52, 55, 59, 62, 66], "Dend": [38, 50, 57, 62, 66, 69],
}


def pad_voice(f: float, n: int, bright: float, rng: np.random.Generator) -> np.ndarray:
    """Soft additive tone: harmonics falling as 1/k^p (p lower = brighter), two detuned copies, slow vibrato."""
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    p = 2.2 - 0.8 * bright
    for ch, det in ((0, -0.004), (1, 0.004)):
        ph0 = rng.uniform(0, 2 * np.pi)
        vib = 1 + 0.0015 * np.sin(2 * np.pi * 0.21 * t + ph0)
        for k in range(1, 9):
            fk = f * k * (1 + det) * vib
            if f * k > 9000:
                break
            out[:, ch] += np.sin(2 * np.pi * np.cumsum(fk) / SR + ph0 * k) / k ** p
    return out


def envelope(n: int, a: float, r: float) -> np.ndarray:
    e = np.ones(n)
    na, nr = int(a * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na) ** 2
    e[-nr:] *= np.linspace(1, 0, nr) ** 2
    return e


def pluck(f: float, dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)) * np.exp(-t / 0.22)
    s[: int(0.004 * SR)] *= np.linspace(0, 1, int(0.004 * SR))
    return s


def reverb(x: np.ndarray, seconds: float = 2.2, wet: float = 0.28, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.empty_like(x)
    for ch in range(2):
        ir = rng.standard_normal(n) * np.exp(-t / (seconds / 5))
        ir /= np.sqrt(np.sum(ir ** 2))
        out[:, ch] = fftconvolve(x[:, ch], ir)[: len(x)]
    return (1 - wet) * x + wet * out


def compose(total: float, scene_starts: list[float], seed: int = 1) -> np.ndarray:
    """scene_starts: start time of each of the 7 scenes. Returns stereo float array, peak about 0.5."""
    rng = np.random.default_rng(seed)
    n = int(total * SR) + 1
    pad = np.zeros((n, 2))
    s = scene_starts + [total]
    # (chord, start, end, brightness, level): two chords per scene, lifting through scenes 3-4
    plan = []
    names = [("D", "Bm"), ("G", "A"), ("D", "Bm"), ("G", "A"), ("Em", "A"), ("G", "A"), ("Dend", "Dend")]
    bright = [0.15, 0.25, 0.35, 0.75, 0.85, 0.5, 0.3]
    level = [0.8, 0.85, 0.9, 1.1, 1.15, 0.95, 1.0]
    for i, (c1, c2) in enumerate(names):
        a, b = s[i], s[i + 1]
        if i == 6:
            plan.append((c1, a - 0.3, total, bright[i], level[i]))
        else:
            mid = (a + b) / 2
            plan.append((c1, a - 0.3, mid + 0.8, bright[i], level[i]))
            plan.append((c2, mid - 0.3, b + 0.8, bright[i], level[i]))
    for name, a, b, br, lv in plan:
        i0, i1 = int(max(a, 0) * SR), min(int(b * SR), n)
        m = i1 - i0
        if m <= 0:
            continue
        seg = sum(pad_voice(midi_hz(x), m, br, rng) for x in CHORDS[name]) / len(CHORDS[name])
        rel = 3.0 if name == "Dend" else 1.4
        pad[i0:i1] += lv * seg * envelope(m, 1.2, min(rel, m / SR / 2))[:, None]
    sos = butter(2, 2600, "low", fs=SR, output="sos")
    pad = sosfilt(sos, pad, axis=0)
    # gentle pulse: eighth-note arpeggio at 96 bpm from scene 1 to the start of the end card
    pulse = np.zeros((n, 2))
    eighth = 60 / 96 / 2
    plev = [0.0, 0.10, 0.13, 0.22, 0.24, 0.15, 0.0]
    t = s[1] - 0.2
    k = 0
    while t < s[6] - 0.1:
        sc = max(i for i in range(7) if s[i] <= t)
        chord = next(c for c, a, b, _, _ in reversed(plan) if a <= t)
        notes = CHORDS[chord]
        f = midi_hz(notes[[1, 2, 3, 4, 3, 2][k % 6]] + 12)
        sig = pluck(f, 0.6) * plev[sc] * (0.8 if k % 2 else 1.0)
        i0 = int(t * SR)
        m = min(len(sig), n - i0)
        pan = 0.5 + 0.25 * np.sin(k * 0.9)
        pulse[i0:i0 + m, 0] += sig[:m] * (1 - pan)
        pulse[i0:i0 + m, 1] += sig[:m] * pan
        t += eighth
        k += 1
    mix = reverb(pad + pulse)
    mix /= np.max(np.abs(mix)) + 1e-9
    return 0.5 * mix
