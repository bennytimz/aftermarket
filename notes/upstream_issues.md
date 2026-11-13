# Upstream issues

Real gaps/bugs found in Lexsi kits (CuratorKIT, SafeTune, CircuitKIT, AuditKIT) on Tiny Aya. See PLAN.md §7.

## CircuitKIT 0.2.3: `discover_circuit` cannot pin a model revision
`config["model"]` takes `name`, `precision`, `device`, `trust_remote_code`, but no `revision`, so EAP-IG on
`CohereLabs/tiny-aya-global` always loads the Hub default branch. We pin `af89d219` everywhere else; the
locate step records this caveat in `eap_ig.json`. Proposal: accept `model.revision` and pass it to the
HF/TransformerLens loader. (Found reading `circuitkit/api.py`, 2026-11-13.)

## CircuitKIT 0.2.3: per-layer protect tiers vs HQQ per-type configs
Not a CircuitKIT bug, a gap we hit for the AM compression shield: transformers' `HqqConfig(dynamic_config=...)`
is keyed by module TYPE (`self_attn.q_proj`), not by layer, so "L* at 8-bit, the rest at 3-bit" cannot be
expressed through it. We swap `nn.Linear` -> `hqq.HQQLinear` per layer ourselves (`attacks/compress.py`).
A CircuitKIT protect-tier export to a per-layer HQQ map would cover this.

## SafeTune 0.1.18: SafeGrad KL mode needs a full second model
`SafeGradHFTrainer`'s paper-faithful KL alignment loads a frozen reference model next to the trainee; on a
16 GB T4 with Tiny Aya (fp16 ~6.7 GB) plus QLoRA training this does not fit, so AM uses the documented
SFT-loss alignment signal (`reference_model=None` + `safety_dataset`). A reference in 4-bit, or cached
reference logits for a fixed alignment set, would make KL mode usable on small GPUs.
