"""Fetch BibTeX for every reference from authoritative sources and verify it.

    python paper/fetch_bib.py      # -> paper/custom.bib + paper/bib_check.json

DBLP blocks scripted access (bot check, 2026-11-22), so:
  1. Semantic Scholar `paper/search/match` finds the paper by its exact title (normalized title must match).
  2. ACL Anthology papers: the Anthology's own .bib (aclanthology.org/<id>.bib).
     Everything else: arXiv's own BibTeX export (arxiv.org/bibtex/<id>), the authors and title as the authors gave them.
  3. For peer-reviewed non-ACL venues (NeurIPS, ICLR, ICML, AAAI, SOSP, AIES) the arXiv entry is turned into an
     @inproceedings with the venue. The venue name must appear in Semantic Scholar's venue field (checked); the venue
     year is the one listed below and must be >= the arXiv year.
Nothing is typed by hand except the venue year of a verified venue. Any mismatch is reported and the entry dropped.

custom.bib has since been polished by hand (braces that keep capitals, arXiv-only papers as @article with an
"arXiv preprint" journal) and later entries were appended to it from bib_cache/. A full rerun would undo that
polish: to add a reference, add it to REFS, fetch it into bib_cache/, and append that entry to custom.bib by hand.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "aftermarket-paper-bib/0.2"}

VENUES = {  # short name -> (booktitle, string that must occur in Semantic Scholar's venue field)
    "NeurIPS": ("Advances in Neural Information Processing Systems", "Neural Information Processing Systems"),
    "ICLR": ("International Conference on Learning Representations", "International Conference on Learning Representations"),
    "ICML": ("International Conference on Machine Learning", "International Conference on Machine Learning"),
    "AAAI": ("Proceedings of the AAAI Conference on Artificial Intelligence", "AAAI"),
    "SOSP": ("Proceedings of the 29th Symposium on Operating Systems Principles", "Symposium on Operating Systems Principles"),
    "AIES": ("Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society", "Ethics, and Society"),
}

# key -> (exact title, (venue, venue year) or None for arXiv / ACL Anthology)
REFS = {
    "qi2024finetuning": ("Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!", ("ICLR", 2024)),
    "yang2023shadow": ("Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models", None),
    "zhan2024removing": ("Removing RLHF Protections in GPT-4 via Fine-Tuning", None),
    "lermen2023lora": ("LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B", None),
    "arditi2024refusal": ("Refusal in Language Models Is Mediated by a Single Direction", ("NeurIPS", 2024)),
    "huang2024lisa": ("Lisa: Lazy Safety Alignment for Large Language Models against Harmful Fine-tuning Attack", ("NeurIPS", 2024)),
    "huang2024survey": ("Harmful Fine-tuning Attacks and Defenses for Large Language Models: A Survey", None),
    "tamirisa2025tamper": ("Tamper-Resistant Safeguards for Open-Weight LLMs", ("ICLR", 2025)),
    "rosati2024representation": ("Representation Noising: A Defence Mechanism Against Harmful Finetuning", ("NeurIPS", 2024)),
    "qi2025shallow": ("Safety Alignment Should Be Made More Than Just a Few Tokens Deep", ("ICLR", 2025)),
    "wei2024brittleness": ("Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications", ("ICML", 2024)),
    "henderson2023selfdestructing": ("Self-Destructing Models: Increasing the Costs of Harmful Dual Uses of Foundation Models", ("AIES", 2023)),
    "egashira2024exploiting": ("Exploiting LLM Quantization", ("NeurIPS", 2024)),
    "hong2024decoding": ("Decoding Compressed Trust: Scrutinizing the Trustworthiness of Efficient LLMs Under Compression", ("ICML", 2024)),
    "inan2023llamaguard": ("Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations", None),
    "han2024wildguard": ("WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs", ("NeurIPS", 2024)),
    "souly2024strongreject": ("A StrongREJECT for Empty Jailbreaks", ("NeurIPS", 2024)),
    "mazeika2024harmbench": ("HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal", ("ICML", 2024)),
    "zou2023universal": ("Universal and Transferable Adversarial Attacks on Aligned Language Models", None),
    "rottger2024xstest": ("XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models", None),
    "deng2024multilingual": ("Multilingual Jailbreak Challenges in Large Language Models", ("ICLR", 2024)),
    "yong2023lowresource": ("Low-Resource Languages Jailbreak GPT-4", None),
    "nllb2022": ("No Language Left Behind: Scaling Human-Centered Machine Translation", None),
    "kargaran2023glotlid": ("GlotLID: Language Identification for Low-Resource Languages", None),
    "biderman2024lessons": ("Lessons from the Trenches on Reproducible Evaluation of Language Models", None),
    "kwon2023vllm": ("Efficient Memory Management for Large Language Model Serving with PagedAttention", ("SOSP", 2023)),
    "hendrycks2021mmlu": ("Measuring Massive Multitask Language Understanding", ("ICLR", 2021)),
    "clark2018arc": ("Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge", None),
    "zellers2019hellaswag": ("HellaSwag: Can a Machine Really Finish Your Sentence?", None),
    "sakaguchi2020winogrande": ("WinoGrande: An Adversarial Winograd Schema Challenge at Scale", ("AAAI", 2020)),
    "cobbe2021gsm8k": ("Training Verifiers to Solve Math Word Problems", None),
    "zhou2023ifeval": ("Instruction-Following Evaluation for Large Language Models", None),
    "hughes2024bon": ("Best-of-N Jailbreaking", None),
    "dettmers2023qlora": ("QLoRA: Efficient Finetuning of Quantized LLMs", ("NeurIPS", 2023)),
    "hu2022lora": ("LoRA: Low-Rank Adaptation of Large Language Models", ("ICLR", 2022)),
    "ustun2024aya": ("Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model", None),
    "yi2025safegrad": ("Gradient Surgery for Safe LLM Fine-Tuning", None),
    "zheng2023judging": ("Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", ("NeurIPS", 2023)),
    "chao2024jailbreakbench": ("JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models", ("NeurIPS", 2024)),
    # bib_cache/salamanca2026tinyaya.bib holds the entry as first added to custom.bib (arXiv 2603.11510).
    "salamanca2026tinyaya": ("Tiny Aya: Bridging Scale and Multilingual Depth", None),
    # Tools and data the paper uses (added 2026-11-26).
    "reimers2019sbert": ("Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", None),
    "wang2020minilm": ("MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers", ("NeurIPS", 2020)),
    "cui2024orbench": ("OR-Bench: An Over-Refusal Benchmark for Large Language Models", ("ICML", 2025)),
    "sheshadri2025lat": ("Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs", None),
    "wolf2020transformers": ("Transformers: State-of-the-Art Natural Language Processing", None),
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def get(url: str, tries: int = 6) -> str:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read().decode("utf-8")
        except Exception:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def s2_match(title: str) -> dict | None:
    q = urllib.parse.quote(title)
    url = (f"https://api.semanticscholar.org/graph/v1/paper/search/match?query={q}"
           "&fields=title,year,venue,externalIds,authors,publicationVenue")
    try:
        d = json.loads(get(url))
    except RuntimeError:
        return None
    hits = d.get("data") or []
    return hits[0] if hits and norm(hits[0]["title"]) == norm(title) else None


def rekey(bib: str, key: str) -> str:
    return re.sub(r"@(\w+)\{[^,]+,", lambda m: f"@{m.group(1)}{{{key},", bib.strip(), count=1)


def field(bib: str, name: str) -> str | None:
    m = re.search(rf"\b{name}\s*=\s*[{{\"](.+?)[}}\"]\s*,?\s*$", bib, flags=re.M | re.I)
    return m.group(1).strip() if m else None


def to_inproceedings(arxiv_bib: str, key: str, booktitle: str, year: int, arxiv_id: str) -> str:
    title, author = field(arxiv_bib, "title"), field(arxiv_bib, "author")
    return (f"@inproceedings{{{key},\n  title = {{{title}}},\n  author = {{{author}}},\n"
            f"  booktitle = {{{booktitle}}},\n  year = {{{year}}},\n  note = {{arXiv:{arxiv_id}}},\n"
            f"  url = {{https://arxiv.org/abs/{arxiv_id}}}\n}}")


def fetch(key: str, title: str, venue) -> tuple[str | None, dict]:
    m = s2_match(title)
    if m is None:
        return None, {"status": "NOT FOUND on Semantic Scholar", "title": title}
    ext = m.get("externalIds") or {}
    rep = {"s2_title": m["title"], "s2_year": m.get("year"), "s2_venue": m.get("venue"), "externalIds": ext}
    if ext.get("ACL"):
        bib = get(f"https://aclanthology.org/{ext['ACL']}.bib")
        if norm(field(bib, "title") or "") != norm(title):
            return None, {**rep, "status": "ACL title mismatch"}
        return rekey(bib, key), {**rep, "status": "ok", "source": f"ACL Anthology {ext['ACL']}"}
    if not ext.get("ArXiv"):
        return None, {**rep, "status": "no ACL / arXiv id"}
    abib = get(f"https://arxiv.org/bibtex/{ext['ArXiv']}")
    if norm(field(abib, "title") or "") != norm(title):
        return None, {**rep, "status": "arXiv title mismatch", "arxiv_title": field(abib, "title")}
    if venue is None:
        return rekey(abib, key), {**rep, "status": "ok", "source": f"arXiv {ext['ArXiv']}"}
    short, vyear = venue
    booktitle, must = VENUES[short]
    s2v = f"{m.get('venue') or ''} {(m.get('publicationVenue') or {}).get('name', '')}"
    ayear = int(field(abib, "year") or 0)
    if must.lower() not in s2v.lower():
        return None, {**rep, "status": f"venue not confirmed: wanted {short}, S2 says {s2v.strip()!r}"}
    if vyear < ayear:
        return None, {**rep, "status": f"venue year {vyear} < arXiv year {ayear}"}
    return to_inproceedings(abib, key, booktitle, vyear, ext["ArXiv"]), {
        **rep, "status": "ok", "source": f"arXiv {ext['ArXiv']} + venue {short} {vyear} (confirmed by S2)"}


CACHE = HERE / "bib_cache"


def main() -> None:
    """Entries already fetched and verified are cached in paper/bib_cache/ (one file each); only missing ones are
    fetched, so a rate-limited run never loses entries."""
    CACHE.mkdir(exist_ok=True)
    old = json.loads((HERE / "bib_check.json").read_text(encoding="utf-8")) if (HERE / "bib_check.json").exists() else {}
    entries, report = [], {}
    for key, (title, venue) in REFS.items():
        c = CACHE / f"{key}.bib"
        if c.exists():
            entries.append(c.read_text(encoding="utf-8"))
            report[key] = old.get(key, {"status": "ok (cached)"})
            continue
        bib, rep = fetch(key, title, venue)
        if bib:
            c.write_text(bib, encoding="utf-8")
        report[key] = rep
        print(f"{rep['status'][:60]:60s} {key}", flush=True)
        if bib:
            entries.append(bib)
        time.sleep(1.2)  # Semantic Scholar: ~1 request/s without a key
    (HERE / "custom.bib").write_text("% Generated by paper/fetch_bib.py (sources in paper/bib_check.json). "
                                     "Do not edit by hand.\n\n" + "\n\n".join(entries) + "\n", encoding="utf-8")
    (HERE / "bib_check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
