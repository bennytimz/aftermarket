"""Build AM and its controls (PLAN.md §6.2-6.5).

Circuit-restricted SafeGrad/Lisa LoRA on L*, refusal spreading with the top
refusal direction ablated, compression shield (L* at 8-bit), optional ReSta
re-anchor. Controls: same recipe on random layers (3 seeds) and plain SafeGrad
over all layers.
"""
