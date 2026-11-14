"""A3 v2: layer chosen by causal effect (bypass), not by separation; stale v1 results archived.

The v1 bug: Cohen's d separation falls monotonically after layer 4 on Tiny Aya, so every model got the window
floor (layer 7) and A3 left refusal unchanged (A0 ~ A3 on harmful_test for all 8 models).
"""

import numpy as np
import pytest
import yaml

from aftermarket.attacks.a3_archive import a3_cells, is_stale, plan_moves, remove_local
from aftermarket.attacks.abliterate import (SELECT_BY, candidate_layers, choose_layer, pick_layer, project_out,
                                            project_out_rows)
from aftermarket.data import REPO_ROOT

# BASE's A3 manifest (v1): separation per layer input, Tiny Aya 36 layers
BASE_SEP = [float("nan"), 0.32, 2.39, 4.49, 4.7, 3.77, 3.57, 3.03, 2.8, 2.45, 2.43, 2.39, 2.06, 1.66, 1.58, 1.9,
            2.02, 1.75, 1.71, 1.77, 2.02, 2.2, 2.23, 2.25, 2.13, 2.25, 2.25, 2.2, 2.15, 2.08, 2.15, 2.18, 2.16,
            2.16, 2.09, 2.07]


def test_v1_rule_always_hit_the_window_floor():
    lo = int(0.2 * len(BASE_SEP))
    assert pick_layer(BASE_SEP, (0.2, 0.8)) == lo == 7  # the bug: the floor, whatever refusal does


def test_candidate_layers_match_window_and_stride():
    assert candidate_layers(36, (0.2, 0.8)) == list(range(7, 28))
    assert candidate_layers(36, (0.2, 0.8), stride=3) == list(range(7, 28, 3))
    assert candidate_layers(10, (0.5, 0.5)) == [5]


def test_choose_layer_lowest_refusal_among_kl_ok():
    cands = [{"layer": 7, "refusal": 0.80, "kl": 0.01, "sep": 3.0},   # v1's pick: no bypass
             {"layer": 14, "refusal": 0.05, "kl": 0.40, "sep": 1.6},  # bypasses but breaks harmless prompts
             {"layer": 16, "refusal": 0.10, "kl": 0.05, "sep": 2.0},
             {"layer": 18, "refusal": 0.10, "kl": 0.02, "sep": 1.7}]
    assert choose_layer(cands, kl_max=0.1) == (16, False)  # tie on refusal -> higher separation
    assert choose_layer(cands, kl_max=1.0) == (14, False)


def test_choose_layer_relaxes_kl_when_nothing_passes():
    cands = [{"layer": 9, "refusal": 0.5, "kl": 0.3, "sep": 2.0}, {"layer": 10, "refusal": 0.2, "kl": 0.5, "sep": 2.0}]
    assert choose_layer(cands, kl_max=0.1) == (10, True)
    with pytest.raises(ValueError):
        choose_layer([], 0.1)


def test_project_out_activations():
    rng = np.random.default_rng(0)
    d = rng.normal(size=16); d /= np.linalg.norm(d)
    x = rng.normal(size=(4, 5, 16))
    assert np.abs(project_out(x, d) @ d).max() < 1e-12
    np.testing.assert_allclose(project_out(project_out(x, d), d), project_out(x, d), atol=1e-12)


def test_config_uses_bypass_selection():
    a3 = yaml.safe_load((REPO_ROOT / "configs" / "attacks.yaml").read_text(encoding="utf-8"))["attacks"]["A3"]
    assert a3["select_by"] == SELECT_BY == "bypass"
    assert {"n_harmful", "n_benign", "max_new_tokens", "kl_max"} <= set(a3["select"])
    assert a3["select_on"] == ["harmful_dev", "benign_dev"]  # dev only, never test


# ----------------------------------------------------------------------------- archive of v1 results

def test_stale_and_moves():
    assert is_stale({"attack_info": {"attack": "A3", "layer": 7}})
    assert is_stale({"attack_info": {"select_by": "separation"}})
    assert not is_stale({"attack_info": {"select_by": "bypass"}})
    files = ["gen/BASE__A3__harmful_test.json", "gen/BASE__A3__harmful_test.parquet",
             "judged/BASE__A3__harmful_test.json", "judged/BASE__A3__harmful_test.parquet",
             "gen/AMc_k6__A3__harmful_dev.json", "gen/AMc_k6__A3__harmful_dev.parquet",
             "gen/BASE__A0__harmful_test.json", "archive/A3_v1/gen/X__A3__harmful_test.json"]
    assert a3_cells(files) == ["AMc_k6__A3__harmful_dev", "BASE__A3__harmful_test"]
    moves = plan_moves(["AMc_k6__A3__harmful_dev", "BASE__A3__harmful_test"], files)
    assert [s for s, _ in moves] == ["gen/BASE__A3__harmful_test.parquet", "gen/BASE__A3__harmful_test.json",
                                     "judged/BASE__A3__harmful_test.parquet", "judged/BASE__A3__harmful_test.json"]
    assert all(d == f"archive/A3_v1/{s}" for s, d in moves)  # AMc_* kept: the frozen selection's record


def test_remove_local_stale_copies(tmp_path):
    for top in ("gen", "judged"):
        (tmp_path / "results" / top).mkdir(parents=True)
        for n in ("BASE__A3__harmful_test", "AMc_k6__A3__harmful_dev", "BASE__A0__harmful_test"):
            (tmp_path / "results" / top / f"{n}.parquet").write_bytes(b"x")
    gone = remove_local(["BASE__A3__harmful_test", "AMc_k6__A3__harmful_dev"], tmp_path)
    assert sorted(p.parent.name for p in gone) == ["gen", "judged"]
    assert (tmp_path / "results" / "gen" / "AMc_k6__A3__harmful_dev.parquet").exists()
    assert (tmp_path / "results" / "gen" / "BASE__A0__harmful_test.parquet").exists()


# ----------------------------------------------------------------------------- hooks == weight edit (torch)

def _tiny_parallel_lm(torch, n_layers=3, D=8, V=20, tied=True):
    """Cohere2-shaped toy: parallel block h + o_proj(attn(norm h)) + down_proj(mlp(norm h)), final norm, lm_head."""
    nn = torch.nn

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(D, D, bias=False)
            self.o_proj = nn.Linear(D, D, bias=False)

        def forward(self, x):
            return self.o_proj(torch.tanh(self.q(x)))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(D, 2 * D, bias=False)
            self.down_proj = nn.Linear(2 * D, D, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.up(x)))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(D)
            self.self_attn, self.mlp = Attn(), MLP()

        def forward(self, hidden_states, attention_mask=None):
            h = self.input_layernorm(hidden_states)
            return hidden_states + self.self_attn(h) + self.mlp(h)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(V, D)
            self.layers = nn.ModuleList([Layer() for _ in range(n_layers)])
            self.norm = nn.LayerNorm(D)

        def forward(self, ids):
            h = self.embed_tokens(ids)
            for i, l in enumerate(self.layers):
                h = l(h) if i % 2 else l(hidden_states=h)  # both call styles: positional and keyword
            return self.norm(h)

    class Config:
        tie_word_embeddings = tied

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Config()
            self.model = Decoder()
            self.lm_head = nn.Linear(D, V, bias=False)
            if tied:
                self.lm_head.weight = self.model.embed_tokens.weight

        def get_decoder(self):
            return self.model

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, ids):
            return self.lm_head(self.model(ids))

    torch.manual_seed(0)
    return LM()


@pytest.mark.parametrize("tied", [True, False])
def test_directional_ablation_equals_weight_abliteration(tied):
    torch = pytest.importorskip("torch")
    import copy

    from aftermarket.attacks.abliterate import abliterate, directional_ablation

    lm = _tiny_parallel_lm(torch, tied=tied).double()
    d = torch.randn(8, dtype=torch.float64); d /= d.norm()
    ids = torch.tensor([[1, 5, 7, 2], [3, 3, 9, 0]])
    with torch.no_grad():
        plain = lm(ids)
        with directional_ablation(lm, d):
            hooked = lm(ids)
        after_hooks = lm(ids)
        edited = copy.deepcopy(lm)
        abliterate(edited, d)
        weight = edited(ids)
    assert torch.allclose(after_hooks, plain)          # hooks removed, weights untouched
    assert not torch.allclose(hooked, plain)
    assert torch.allclose(hooked, weight, atol=1e-10)  # selection measures exactly the attack it then applies


def test_project_out_rows_matches_activation_projection():
    rng = np.random.default_rng(1)
    d = rng.normal(size=8); d /= np.linalg.norm(d)
    W, x = rng.normal(size=(8, 5)), rng.normal(size=5)
    np.testing.assert_allclose(project_out_rows(W, d) @ x, project_out(W @ x, d), atol=1e-12)
