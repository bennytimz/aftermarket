"""Stage 2 prompt sets for vLLM generation (E3, E5, E6). Each set is a list of
{"id", "set", "lang", "kind" (harmful|xstest), "origin" (orig|mt|human), "prompt"}.

    ml_<lang>   E6: the fixed 100 harmful_test + 50 xstest items; en = original, others = NLLB MT
                (data/stage2/translations.jsonl, built by aftermarket.stage2.translate)
    mj_<lang>   E6 human-translation check: 100 MultiJail items (en, zh, sw), pinned revision
    sr          E3: 150 StrongREJECT prompts, pinned commit
    harmful_test  the 200 Stage 1 harmful_test prompts (A1/A2 re-run vs Stage 1; E5 seeds)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from aftermarket.stage2.subsets import REPO_ROOT, load_subsets, read_jsonl

TRANSLATIONS = REPO_ROOT / "data" / "stage2" / "translations.jsonl"
ML_LANGS = ["en", "es", "zh", "yo", "sw", "fr"]
MJ_LANGS = ["en", "zh", "sw"]


def _splits() -> dict[str, dict]:
    d = REPO_ROOT / "data" / "splits"
    rows = read_jsonl(d / "harmful_test.jsonl") + read_jsonl(d / "xstest_test.jsonl")
    return {r["id"]: r for r in rows}


def ml_items() -> list[dict]:
    """The 150 English source items of E6, in a fixed order (harmful first)."""
    sub = load_subsets()["multilingual"]
    by_id = _splits()
    return ([{"id": i, "kind": "harmful", "prompt": by_id[i]["prompt"]} for i in sub["harmful_ids"]]
            + [{"id": i, "kind": "xstest", "prompt": by_id[i]["prompt"]} for i in sub["xstest_ids"]])


def load_translations(path: Path = TRANSLATIONS) -> dict[tuple[str, str], str]:
    return {(r["id"], r["lang"]): r["text"] for r in read_jsonl(path)} if path.exists() else {}


def ml_set(lang: str) -> list[dict]:
    tr = load_translations()
    out = []
    for it in ml_items():
        if lang == "en":
            text, origin = it["prompt"], "orig"
        else:
            if (it["id"], lang) not in tr:
                raise FileNotFoundError(f"no {lang} translation of {it['id']}: run aftermarket.stage2.translate")
            text, origin = tr[(it["id"], lang)], "mt"
        out.append({"id": it["id"], "set": f"ml_{lang}", "lang": lang, "kind": it["kind"], "origin": origin,
                    "prompt": text})
    return out


@lru_cache(maxsize=1)
def _multijail():
    import pandas as pd
    from huggingface_hub import hf_hub_download

    s = load_subsets()["multijail"]
    df = pd.read_csv(hf_hub_download(s["hf_dataset"], s["file"], repo_type="dataset", revision=s["revision"]))
    return df.iloc[s["rows"]]


def mj_set(lang: str) -> list[dict]:
    df = _multijail()
    return [{"id": f"multijail:{int(r['id'])}", "set": f"mj_{lang}", "lang": lang, "kind": "harmful",
             "origin": "orig" if lang == "en" else "human", "prompt": str(r[lang])} for _, r in df.iterrows()]


def sr_set() -> list[dict]:
    import io
    import urllib.request

    import pandas as pd

    s = load_subsets()["strongreject"]
    df = pd.read_csv(io.BytesIO(urllib.request.urlopen(s["url"]).read()))
    assert len(df) == s["n_total"]
    return [{"id": f"strongreject:{i:03d}", "set": "sr", "lang": "en", "kind": "harmful", "origin": "orig",
             "category": df.iloc[i]["category"], "prompt": str(df.iloc[i][s["prompt_col"]])} for i in s["rows"]]


def harmful_test_set() -> list[dict]:
    rows = read_jsonl(REPO_ROOT / "data" / "splits" / "harmful_test.jsonl")
    return [{"id": r["id"], "set": "harmful_test", "lang": "en", "kind": "harmful", "origin": "orig",
             "prompt": r["prompt"]} for r in rows]


def build_set(name: str) -> list[dict]:
    if name.startswith("ml_"):
        return ml_set(name[3:])
    if name.startswith("mj_"):
        return mj_set(name[3:])
    if name == "sr":
        return sr_set()
    if name == "harmful_test":
        return harmful_test_set()
    raise KeyError(name)


def dump(rows: list[dict], path: Path) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
