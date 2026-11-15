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

A one-page **report card** per model (Report cards tab, `cards/`): attack success per attack with 95% CIs against
the leaderboard range, retention and rank, over-refusal and MMLU. Make one for your own Tiny Aya derivative with
`aftermarket audit --model <hf id or path>` (one T4; see the GitHub README).

This Space shows aggregate metrics only (`data/*.csv`, `cards/`): no prompts, no model outputs and no
attack-modified weights. Method, code and credits for every team: https://github.com/bennytimz/aftermarket
