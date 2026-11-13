"""A3 abliteration (PLAN.md §4; Arditi et al. 2406.11717).

Per model (each attacked model's own activations, the realistic attacker):
  1. residual-stream activations (input to each decoder layer) at the last prompt token,
     with the model's own chat template, for calib_pairs harmful vs harmless prompts;
  2. candidate direction per layer = unit(mean(harmful) - mean(harmless));
  3. pick the layer whose direction best separates harmful_dev from benign_dev
     (Cohen's d of projections), restricted to the configured depth window;
  4. project that direction out of every write matrix: attn o_proj, mlp down_proj
     (W <- W - d dᵀW) and the token embeddings (E <- E - (E d) dᵀ).

Weights are edited in memory only: never saved to disk or pushed (PLAN.md §9).
The pure math functions work on numpy arrays and torch tensors alike (unit-tested on numpy).
"""

from __future__ import annotations


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


def project_out_rows(W, d):
    """W [hidden, in] writes into the residual stream: remove the d component of every output."""
    return W - d[:, None] * (d[None, :] @ W)


def project_out_cols(E, d):
    """E [vocab, hidden] (embedding rows live in the residual stream): remove the d component."""
    return E - (E @ d)[:, None] * d[None, :]


# ----------------------------------------------------------------------------- model side (torch)

def last_token_acts(model, tok, prompts: list[str], batch_size: int = 16):
    """[n, L, D] float32 CPU: input to each decoder layer at the last prompt token (left padded)."""
    import torch

    from aftermarket.load import build_prompt_ids

    out = []
    pad = tok.pad_token_id
    for i in range(0, len(prompts), batch_size):
        ids = [build_prompt_ids(model, tok, [{"role": "user", "content": p}]) for p in prompts[i:i + batch_size]]
        L = max(map(len, ids))
        x = torch.tensor([[pad] * (L - len(s)) + s for s in ids], device=model.device)
        m = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in ids], device=model.device)
        with torch.no_grad():
            hs = model(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states
        # hidden_states[l] = input to layer l for l < n_layers; the last entry is post-norm, so drop it
        out.append(torch.stack(hs[:-1], dim=1)[:, :, -1, :].float().cpu())
    return torch.cat(out)


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


def apply(model, tok, cfg: dict) -> dict:
    """Run A3 on `model` in place (calib -> directions, dev -> layer, project out). Returns attack info."""
    from aftermarket.data import load_split

    calib = load_split(cfg["fit_split"])
    h = last_token_acts(model, tok, [r["harmful"] for r in calib])
    b = last_token_acts(model, tok, [r["harmless"] for r in calib])
    dirs = mean_diff_directions(h, b)
    dev_h_split, dev_b_split = cfg["select_on"]
    dh = last_token_acts(model, tok, [r["prompt"] for r in load_split(dev_h_split)])
    db = last_token_acts(model, tok, [r["prompt"] for r in load_split(dev_b_split)])
    sep = layer_separation(dirs, dh, db)
    layer = pick_layer(sep, tuple(cfg["layer_window"]))
    stats = abliterate(model, dirs[layer])
    return {"attack": "A3", "layer": layer, "n_layers": len(sep), "separation": [round(s, 4) for s in sep],
            "separation_at_layer": round(sep[layer], 4), **stats}
