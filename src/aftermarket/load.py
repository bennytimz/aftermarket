"""Unified loader for every model in configs/models.yaml (PLAN.md §3, §8, §13.2).

    model, tok = load_model("GC", precision="bf16")
    reply = generate(model, tok, [{"role": "user", "content": "Hi"}])

Formats:
  full    HF checkpoint (BASE, SDT, SP)
  lora    PEFT adapter on BASE @ base_revision (GC, IL); merged unless precision="4bit"
  deltas  ScarTissue rank-one patches applied in order to BASE (ST)
  template  BASE weights + the repo's chat template; the fix is template-only (RL)

Rules (CLAUDE.md): the tokenizer is always BASE's at the pinned revision, and every repo
that ships its own tokenizer must produce identical token IDs (`assert_token_ids_match`).
Each model's own chat template is used when its repo ships one (`template="native"`);
`template="base"` swaps in BASE's template (e.g. RL without its guard, attack A5).

torch / transformers / peft are imported lazily so configs and patch
validation can be unit-tested on a laptop without them.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_YAML = REPO_ROOT / "configs" / "models.yaml"

FORMATS = {"full", "lora", "deltas", "template"}
STATUSES = {"active", "dropped", "planned"}
PRECISIONS = {"bf16", "fp16", "4bit"}

# Multilingual probe strings for the token-ID check (scripts the models are evaluated in).
TOKEN_PROBES = [
    "How do I open a savings account?",
    "¿Cómo abro una cuenta de ahorros?",
    "Как открыть сберегательный счёт?",
    "كيف أفتح حساب توفير؟",
    "मैं बचत खाता कैसे खोलूं?",
    "Paano ako magbubukas ng savings account?",
    "আমি কীভাবে একটি সঞ্চয় অ্যাকাউন্ট খুলব?",
    "Ninawezaje kufungua akaunti ya akiba?",
    "Abeg, how I fit open savings account for bank?",
    "def f(x):\n    return x ** 2  # 🙂",
    "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>hi<|END_OF_TURN_TOKEN|>",
]


# ----------------------------------------------------------------------------- config

def load_registry(path: str | Path = MODELS_YAML) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_spec(model_id: str, registry: dict | None = None) -> dict:
    reg = registry or load_registry()
    spec = reg["models"].get(model_id) or reg.get("optional", {}).get(model_id)
    if spec is None:
        raise KeyError(f"unknown model id {model_id!r}")
    if spec.get("status") != "active":
        raise ValueError(f"{model_id} is {spec.get('status')}: {spec.get('drop_reason', spec.get('notes', ''))}")
    return spec


def active_model_ids(registry: dict | None = None) -> list[str]:
    reg = registry or load_registry()
    return [k for k, v in reg["models"].items() if v.get("status") == "active"]


# ----------------------------------------------------------------------------- tokenizer

@lru_cache(maxsize=1)
def load_base_tokenizer():
    from transformers import AutoTokenizer

    reg = load_registry()
    return AutoTokenizer.from_pretrained(reg["base_model"], revision=reg["base_revision"])


def assert_token_ids_match(base_tok, other_tok, probes: list[str] = TOKEN_PROBES, label: str = "") -> None:
    """Raise if `other_tok` encodes any probe differently from the pinned BASE tokenizer."""
    for text in probes:
        a = base_tok(text, add_special_tokens=False)["input_ids"]
        b = other_tok(text, add_special_tokens=False)["input_ids"]
        if a != b:
            raise AssertionError(f"{label}: token IDs differ from BASE on {text!r}\nbase={a}\nrepo={b}")


def _check_repo_tokenizer(spec: dict, base_tok) -> None:
    if "tokenizer.json" not in spec.get("files", []) or spec["hf_path"] == load_registry()["base_model"]:
        return
    from transformers import AutoTokenizer

    repo_tok = AutoTokenizer.from_pretrained(spec["hf_path"], revision=spec["revision"])
    assert_token_ids_match(base_tok, repo_tok, label=spec["hf_path"])


def _chat_template(spec: dict, template: str) -> str | None:
    """Model's own template text (template='native'), or None to use BASE's."""
    if template == "base" or not spec.get("chat_template_file"):
        return None
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(spec["hf_path"], spec["chat_template_file"], revision=spec["revision"])
    return Path(p).read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- deltas (ScarTissue)
# Reimplements amugoodbad229/scartissue artifacts/load_patch.py (validated loader):
# delta = alpha * outer(d, proj) in float64 on the stored columns, cast float32 -> param dtype.

def load_patch(raw: bytes) -> dict:
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        names = z["modules"]
        if names.ndim != 1 or names.dtype.kind != "U":
            raise ValueError("modules: expected a string vector")
        modules = names.tolist()
        patch = {
            "d": z["d"],
            "alpha": z["alpha"],
            "modules": modules,
            "proj": {m: z[f"proj__{i}"] for i, m in enumerate(modules)},
            "cols": {m: z[f"cols__{i}"] for i, m in enumerate(modules)},
        }
    validate_patch(patch)
    return {
        "d": patch["d"].astype(np.float64),
        "alpha": float(patch["alpha"]),
        "modules": modules,
        "proj": {m: p.astype(np.float64) for m, p in patch["proj"].items()},
        "cols": {m: c.astype(np.int64) for m, c in patch["cols"].items()},
    }


def validate_patch(patch: dict) -> None:
    d = np.asarray(patch["d"])
    if d.ndim != 1 or not d.size or not np.isfinite(d).all() or not np.isclose(np.linalg.norm(d), 1, atol=1e-6):
        raise ValueError("d: expected a finite nonempty unit vector")
    if np.asarray(patch["alpha"]).ndim != 0 or not np.isfinite(patch["alpha"]):
        raise ValueError("alpha: expected a finite scalar")
    mods = patch["modules"]
    if not mods or len(set(mods)) != len(mods) or set(patch["proj"]) != set(mods) or set(patch["cols"]) != set(mods):
        raise ValueError("modules / factor keys mismatch")
    for m in mods:
        cols, proj = np.asarray(patch["cols"][m]), np.asarray(patch["proj"][m])
        if cols.ndim != 1 or not cols.size or cols.dtype.kind not in "iu" or cols[0] < 0 or np.any(np.diff(cols) <= 0):
            raise ValueError(f"{m}: expected sorted unique nonnegative integer columns")
        if proj.shape != cols.shape or not np.isfinite(proj).all():
            raise ValueError(f"{m}: bad projection")


def patch_delta(patch: dict, module: str) -> np.ndarray:
    """Exact float32 update for one matrix on its stored columns."""
    return (patch["alpha"] * np.outer(patch["d"], patch["proj"][module])).astype(np.float32)


def apply_patches(model, patches: list[dict]) -> int:
    """Validate every target first, then add each patch in order. Returns edited scalar count."""
    import torch

    params = dict(model.named_parameters())
    for patch in patches:
        for m in patch["modules"]:
            p = params.get(m)
            if p is None or p.ndim != 2 or not p.is_floating_point():
                raise ValueError(f"{m}: missing or not a dense float matrix")
            if len(patch["d"]) != p.shape[0] or patch["cols"][m][-1] >= p.shape[1]:
                raise ValueError(f"{m}: shape mismatch")
    edited = 0
    with torch.no_grad():
        for patch in patches:
            for m in patch["modules"]:
                p, cols = params[m], torch.from_numpy(patch["cols"][m]).to(params[m].device)
                upd = torch.from_numpy(patch_delta(patch, m)).to(p.device, p.dtype)
                p[:, cols] += upd
                edited += upd.numel()
    return edited


# ----------------------------------------------------------------------------- HF loading

def _hf_kwargs(precision: str) -> dict:
    import torch

    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if precision == "4bit":
        from transformers import BitsAndBytesConfig

        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        return {"quantization_config": q, "device_map": {"": 0}}
    return {"dtype": torch.bfloat16 if precision == "bf16" else torch.float16, "device_map": {"": 0}}


def _load_base_hf(precision: str):
    from transformers import AutoModelForCausalLM

    reg = load_registry()
    return AutoModelForCausalLM.from_pretrained(reg["base_model"], revision=reg["base_revision"], **_hf_kwargs(precision))


def load_model(model_id: str, precision: str = "bf16", template: str = "native"):
    """Return (model, tokenizer) for `model_id` from configs/models.yaml.

    precision: "bf16" | "fp16" | "4bit" (bnb nf4).
    template:  "native" (model's own chat template) | "base" (BASE's template).
    The returned model carries `aftermarket_id`, `aftermarket_spec` and
    `aftermarket_chat_template` attributes used by `generate`.
    """
    from huggingface_hub import hf_hub_download

    if template not in {"native", "base"}:
        raise ValueError("template must be 'native' or 'base'")
    spec = get_spec(model_id)
    tok = load_base_tokenizer()
    _check_repo_tokenizer(spec, tok)
    fmt = spec["format"]

    if fmt == "template":
        model = _load_base_hf(precision)
    elif fmt == "full":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(spec["hf_path"], revision=spec["revision"], **_hf_kwargs(precision))
    elif fmt == "lora":
        from peft import PeftModel

        model = PeftModel.from_pretrained(_load_base_hf(precision), spec["hf_path"], revision=spec["revision"])
        if precision != "4bit":
            model = model.merge_and_unload()
    elif fmt == "deltas":
        if precision == "4bit":
            raise ValueError("deltas must be applied to dense weights: load with bf16/fp16")
        model = _load_base_hf(precision)
        patches = [load_patch(Path(hf_hub_download(spec["hf_path"], f, revision=spec["revision"])).read_bytes())
                   for f in spec["delta_files"]]
        apply_patches(model, patches)
    else:
        raise ValueError(f"unknown format {fmt!r}")

    model.eval()
    emb = model.get_input_embeddings().weight.shape[0]
    if emb < len(tok):
        raise AssertionError(f"{model_id}: embedding rows {emb} < BASE vocab {len(tok)}")

    model.aftermarket_id = model_id
    model.aftermarket_spec = spec
    model.aftermarket_chat_template = _chat_template(spec, template)
    return model, tok


def free_memory() -> None:
    """Release GPU memory. Call after `del model, tok` and before loading the next model."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


# ----------------------------------------------------------------------------- generation

def _stop_ids(tok) -> set[int]:
    ids = {tok.eos_token_id}
    for t in ("<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id:
            ids.add(i)
    return {i for i in ids if i is not None}


def build_prompt_ids(model, tok, messages: list[dict] | None = None, raw_text: str | None = None) -> list[int]:
    """Chat-format `messages` with the model's template, or tokenize `raw_text` (template stripped, A5)."""
    if raw_text is None:
        raw_text = tok.apply_chat_template(messages, chat_template=getattr(model, "aftermarket_chat_template", None),
                                           tokenize=False, add_generation_prompt=True)
    return tok(raw_text, add_special_tokens=False)["input_ids"]


def generate(model, tok, messages: list[dict] | None = None, *, raw_text: str | None = None,
             max_new_tokens: int = 256) -> str:
    """Greedy decode one reply (PLAN.md §5)."""
    import torch

    ids = build_prompt_ids(model, tok, messages, raw_text)
    x = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new_tokens,
                           do_sample=False, eos_token_id=sorted(_stop_ids(tok)), pad_token_id=tok.pad_token_id)
    out = y[0, len(ids):].tolist()
    return tok.decode(out, skip_special_tokens=True).strip()


# ----------------------------------------------------------------------------- model runs

def all_run_ids(registry: dict | None = None) -> list[str]:
    """Every generation run: each active model (own template) plus the extra `runs:` variants."""
    reg = registry or load_registry()
    return active_model_ids(reg) + list(reg.get("runs", {}))


def resolve_run(run_id: str, registry: dict | None = None) -> tuple[str, str]:
    """Run ID -> (model_id, template)."""
    reg = registry or load_registry()
    if run_id in reg.get("runs", {}):
        r = reg["runs"][run_id]
        return r["model"], r["template"]
    if reg["models"].get(run_id, {}).get("status") == "active":
        return run_id, "native"
    raise KeyError(f"unknown or inactive run {run_id!r}")
