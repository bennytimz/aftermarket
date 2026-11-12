# Aftermarket

**Does a safety fix survive the next person who downloads the model?**
Post-release robustness of safety-fixed Tiny Aya models: Lexsi AI Safety & Alignment Hackathon.

> **Status: in progress.**

This hackathon has produced many safety-fixed Tiny Aya models. Every one is tested only on the checkpoint its authors released. But nobody runs the checkpoint as released: people fine-tune it, compress it to fit a phone, or strip its refusals on purpose. **Aftermarket** puts the public safety fixes through these everyday changes and reports which ones survive, as a leaderboard and a reusable test suite. Then it builds the combined defense that tops that leaderboard.
