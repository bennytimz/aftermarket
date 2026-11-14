"""A3 abliteration (PLAN.md §4; Arditi et al. 2406.11717).

Per model (each attacked model's own activations, the realistic attacker):
  1. residual-stream activations (input to each decoder layer) at the last prompt token,
     with the model's own chat template, for calib_pairs harmful vs harmless prompts;
  2. candidate direction per layer = unit(mean(harmful) - mean(harmless));
  3. pick the layer by its causal effect (Arditi et al. §2.3 / App. C, select_by: bypass): for every candidate
     layer in the depth window, ablate its direction everywhere (forward hooks, no weight edit) and measure the
     keyword refusal rate of short greedy replies on harmful_dev; drop candidates whose ablation moves the
     next-token distribution on benign_dev by KL > kl_max; lowest refusal wins (ties -> higher separation);
  4. project that direction out of every write matrix: attn o_proj, mlp down_proj
     (W <- W - d dᵀW) and the token embeddings (E <- E - (E d) dᵀ).

v1 (select_by: separation, before 2026-11-14) picked the layer with the largest Cohen's d of harmful_dev vs
benign_dev projections. On Tiny Aya that statistic falls monotonically after layer 4, so every model got the
window floor (layer 7), a harmful-*content* direction whose removal left refusal unchanged (A0 ~ A3 for all
models). Results made with v1 lack attack_info.select_by == "bypass" and are archived by
`python -m aftermarket.attacks.a3_archive`.

Weights are edited in memory only: never saved to disk or pushed (PLAN.md §9).
The pure math functions work on numpy arrays and torch tensors alike (unit-tested on numpy).
"""

from __future__ import annotations

from contextlib import contextmanager

SELECT_BY = "bypass"  # attack_info.select_by of current A3 results (a3_archive treats anything else as stale)


# ----------------------------------------------------------------------------- math (numpy or torch)

def unit(v):
    return v / ((v ** 2).sum(-1) ** 0.5)[..., None]


def mean_diff_directions(harmful, harmless):
    """harmful [n, L, D], harmless [m, L, D] -> unit directions [L, D]."""
    return unit(harmful.mean(0) - harmless.mean(0))


def cohens_d(a, b) -> float:
    """Standardised mean difference of two 1-D samples (pooled population std)."""
    va = ((a - a.mean()) ** 2).mean()
    vb = ((b - b.mean()) ** 2).mean()
    return float((a.mean() - b.mean()) / (((va + vb) / 2) ** 0.5 + 1e-8))


def layer_separation(directions, dev_harmful, dev_harmless) -> list[float]:
    """Per layer l: Cohen's d between projections of dev harmful / harmless onto directions[l]."""
    out = []
    for l in range(directions.shape[0]):
        d = directions[l]
        out.append(cohens_d(dev_harmful[:, l] @ d, dev_harmless[:, l] @ d))
    return out


def pick_layer(separation: list[float], window: tuple[float, float]) -> int:
    n = len(separation)
    lo, hi = int(window[0] * n), int(window[1] * n)
    cands = range(lo, max(hi, lo + 1))
    return max(cands, key=lambda l: separation[l])


def candidate_layers(n_layers: int, window: tuple[float, float], stride: int = 1) -> list[int]:
    """Layers inside the depth window (same bounds as pick_layer), every `stride`-th."""
    lo, hi = int(window[0] * n_layers), int(window[1] * n_layers)
    return list(range(lo, max(hi, lo + 1), max(1, int(stride))))


def choose_layer(cands: list[dict], kl_max: float) -> tuple[int, bool]:
    """cands: [{layer, refusal, kl, sep}]. Keep candidates with kl <= kl_max (ablation leaves harmless prompts
    alone), then lowest refusal under ablation; ties -> higher separation, then lower layer.
    Returns (layer, kl_relaxed): kl_relaxed=True when no candidate passed the KL filter (all were considered)."""
    if not cands:
        raise ValueError("no candidate layers")
    ok = [c for c in cands if c["kl"] <= kl_max]
    best = min(ok or cands, key=lambda c: (c["refusal"], -c["sep"], c["layer"]))
    return best["layer"], not ok


def project_out(x, d):
    """x [..., D]: remove the component along unit d (directional ablation of activations)."""
    return x - (x @ d)[..., None] * d


def project_out_rows(W, d):
    """W [hidden, in] writes into the residual stream: remove the d component of every output."""
    return W - d[:, None] * (d[None, :] @ W)


def project_out_cols(E, d):
    """E [vocab, hidden] (embedding rows live in the residual stream): remove the d component."""
    return E - (E @ d)[:, None] * d[None, :]


# ----------------------------------------------------------------------------- model side (torch)

def last_token_acts(model, tok, prompts: list[str], batch_size: int = 16, include_final: bool = False):
    """[n, L, D] float32 CPU: input to each decoder layer at the last prompt token (left padded).
    include_final: [n, L+1, D], plus the (post-norm) output of the last layer, so entry l+1 is layer l's output."""
    import torch

    from aftermarket.load import build_prompt_ids

    out = []
    for i in range(0, len(prompts), batch_size):
        ids = [build_prompt_ids(model, tok, [{"role": "user", "content": p}]) for p in prompts[i:i + batch_size]]
        x, m = _left_pad(ids, tok.pad_token_id, model.device)
        with torch.no_grad():
            hs = layer_inputs(model, x, m, include_final)
        out.append(torch.stack(hs, dim=1)[:, :, -1, :].float().cpu())
    return torch.cat(out)


def layer_inputs(model, input_ids, attention_mask, include_final: bool = False):
    """Hidden states entering each decoder layer (tuple of n_layers [B, T, D]).

    logits_to_keep=1: only the last position goes through lm_head. Without it Tiny Aya's 262k-vocab head
    materialises [B, T, 262k] logits (3-6 GiB at bs=16) and OOMs a T4 (Kaggle notebook 03, every A3 job).
    """
    hs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
               logits_to_keep=1).hidden_states
    # hidden_states[l] = input to layer l for l < n_layers; the last entry is post-norm, so drop it (A3)
    return hs if include_final else hs[:-1]


def write_modules(model, names=("self_attn.o_proj", "mlp.down_proj")):
    return [(n, m) for n, m in model.named_modules() if n.endswith(names) and hasattr(m, "weight")]


def abliterate(model, d) -> dict:
    """Project unit direction d [D] out of every write matrix in place. Returns residual-check stats."""
    import torch

    worst = 0.0
    with torch.no_grad():
        for _, m in write_modules(model):
            W = m.weight
            dd = d.to(W.device, torch.float32)
            new = project_out_rows(W.float(), dd)
            W.copy_(new.to(W.dtype))
            if getattr(m, "bias", None) is not None:
                b = m.bias.float()
                m.bias.copy_((b - dd * (dd @ b)).to(m.bias.dtype))
            worst = max(worst, float((dd @ W.float()).abs().max()))
        E = model.get_input_embeddings().weight
        dd = d.to(E.device, torch.float32)
        E.copy_(project_out_cols(E.float(), dd).to(E.dtype))
        worst_e = float((E.float() @ dd).abs().max())
    return {"n_write_matrices": len(write_modules(model)), "max_abs_dW_after": worst, "max_abs_Ed_after": worst_e,
            "tied_embeddings": bool(getattr(model.config, "tie_word_embeddings", False))}


def _pre_hook_project(d):
    """Forward pre-hook removing d from the module's hidden-state input (positional or hidden_states=)."""
    def hook(module, args, kwargs):
        if args:
            h = args[0]
            return (project_out(h, d.to(h.device, h.dtype)),) + tuple(args[1:]), kwargs
        h = kwargs["hidden_states"]
        return args, {**kwargs, "hidden_states": project_out(h, d.to(h.device, h.dtype))}
    return hook


@contextmanager
def directional_ablation(model, d):
    """Ablate unit d [D] from the residual stream everywhere, without touching weights (selection only).

    Equivalent to `abliterate`: projecting every decoder layer's input removes d from the embeddings and from
    every earlier layer's o_proj/down_proj writes; the final-norm input catches the last layer's writes; with tied
    embeddings the weight edit also changes lm_head, so its input is projected too."""
    dec = model.get_decoder()
    mods = list(dec.layers) + [dec.norm]
    if getattr(model.config, "tie_word_embeddings", False):
        mods.append(model.get_output_embeddings())
    handles = [m.register_forward_pre_hook(_pre_hook_project(d), with_kwargs=True) for m in mods]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _left_pad(ids: list[list[int]], pad: int, device):
    import torch

    L = max(map(len, ids))
    x = torch.tensor([[pad] * (L - len(s)) + s for s in ids], device=device)
    m = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in ids], device=device)
    return x, m


def next_token_logprobs(model, tok, ids: list[list[int]], batch_size: int = 16):
    """[n, vocab] float32 CPU log-probs of the first reply token."""
    import torch

    out = []
    for i in range(0, len(ids), batch_size):
        x, m = _left_pad(ids[i:i + batch_size], tok.pad_token_id, model.device)
        with torch.no_grad():
            logits = model(input_ids=x, attention_mask=m, logits_to_keep=1).logits[:, -1]
        out.append(logits.float().log_softmax(-1).cpu())
    return torch.cat(out)


def mean_kl(logp, logq) -> float:
    """Mean over rows of KL(p || q) for log-prob rows (torch)."""
    return float((logp.exp() * (logp - logq)).sum(-1).mean())


def refusal_rate(model, tok, ids: list[list[int]], max_new: int, batch_size: int) -> float:
    """Share of greedy replies (max_new tokens) the keyword heuristic labels a refusal."""
    from aftermarket.generate import generate_batched
    from aftermarket.judge import refusal_label

    res, _ = generate_batched(model, tok, ids, max_new, batch_size)
    return sum(refusal_label(r["response"]) == "refusal" for r in res) / len(res)


def bypass_scores(model, tok, dirs, layers: list[int], sep: list[float], harmful: list[str], benign: list[str],
                  max_new: int, batch_size: int) -> tuple[float, list[dict]]:
    """Refusal without ablation, and per candidate layer: refusal rate on `harmful` and KL on `benign` with that
    layer's direction ablated everywhere."""
    from aftermarket.load import build_prompt_ids

    enc = lambda ps: [build_prompt_ids(model, tok, [{"role": "user", "content": p}]) for p in ps]
    h_ids, b_ids = enc(harmful), enc(benign)
    base_refusal = refusal_rate(model, tok, h_ids, max_new, batch_size)
    base_lp = next_token_logprobs(model, tok, b_ids, batch_size)
    out = []
    for l in layers:
        with directional_ablation(model, dirs[l]):
            r = refusal_rate(model, tok, h_ids, max_new, batch_size)
            kl = mean_kl(base_lp, next_token_logprobs(model, tok, b_ids, batch_size))
        out.append({"layer": l, "refusal": round(r, 4), "kl": round(kl, 4), "sep": round(float(sep[l]), 4)})
        print(f"  A3 candidate layer {l:2d}: refusal {r:.2f} (none ablated {base_refusal:.2f}) | KL {kl:.3f}",
              flush=True)
    return base_refusal, out


def apply(model, tok, cfg: dict) -> dict:
    """Run A3 on `model` in place (calib -> directions, dev -> layer, project out). Returns attack info."""
    from aftermarket.data import load_split

    calib = load_split(cfg["fit_split"])
    h = last_token_acts(model, tok, [r["harmful"] for r in calib])
    b = last_token_acts(model, tok, [r["harmless"] for r in calib])
    dirs = mean_diff_directions(h, b)
    dev_h_split, dev_b_split = cfg["select_on"]
    dev_h = [r["prompt"] for r in load_split(dev_h_split)]
    dev_b = [r["prompt"] for r in load_split(dev_b_split)]
    sep = layer_separation(dirs, last_token_acts(model, tok, dev_h), last_token_acts(model, tok, dev_b))
    window = tuple(cfg["layer_window"])
    select_by = cfg.get("select_by", SELECT_BY)
    info: dict = {"select_by": select_by}
    if select_by == "separation":  # v1, kept only to reproduce archived results
        layer = pick_layer(sep, window)
    elif select_by == "bypass":
        s = cfg["select"]
        layers = candidate_layers(len(sep), window, s.get("stride", 1))
        base_refusal, cands = bypass_scores(model, tok, dirs, layers, sep, dev_h[:s["n_harmful"]],
                                            dev_b[:s["n_benign"]], s["max_new_tokens"], s.get("batch_size", 16))
        layer, relaxed = choose_layer(cands, s["kl_max"])
        chosen = next(c for c in cands if c["layer"] == layer)
        info.update({"refusal_no_ablation": round(base_refusal, 4), "refusal_at_layer": chosen["refusal"],
                     "kl_at_layer": chosen["kl"], "kl_filter_relaxed": relaxed, "candidates": cands})
        if relaxed:
            print(f"WARNING A3: no candidate layer kept KL <= {s['kl_max']}; picked the lowest refusal anyway",
                  flush=True)
    else:
        raise ValueError(f"A3 select_by must be 'bypass' or 'separation', got {select_by!r}")
    stats = abliterate(model, dirs[layer])
    print(f"A3: layer {layer} ({select_by})" + (f", refusal {info['refusal_no_ablation']:.2f} -> "
                                                f"{info['refusal_at_layer']:.2f} on dev selection prompts"
                                                if select_by == "bypass" else ""), flush=True)
    return {"attack": "A3", "layer": layer, "n_layers": len(sep), "separation": [round(s, 4) for s in sep],
            "separation_at_layer": round(sep[layer], 4), **info, **stats}
