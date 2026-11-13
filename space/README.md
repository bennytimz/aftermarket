---
title: Aftermarket leaderboard
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Do Tiny Aya safety fixes survive fine-tuning, compression and abliteration?
---

# Aftermarket leaderboard

Post-release robustness of safety-fixed Tiny Aya models (Lexsi AI Safety & Alignment Hackathon). Every public
fix, plus the base model, faces the same battery: benign and harmful fine-tunes, abliteration, 4/3/2-bit
compression, a removed chat template and best-of-8 perturbations. Same prompts, seed, settings and judges for
every model, 95% bootstrap intervals, held-out test prompts only.

This Space shows aggregate metrics only (`data/*.csv`): no prompts, no model outputs and no attack-modified
weights. Method, code and credits for every team: https://github.com/bennytimz/aftermarket
