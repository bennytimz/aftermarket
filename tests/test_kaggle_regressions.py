"""Regressions from the Kaggle run of notebook 03 (every A3 and A4 job failed). No torch needed."""

import sys
import types

import pytest

from aftermarket.generate import FAIL_MARKER, Job, failure_summary, one_line_error, parse_fail_line


# ----------------------------------------------------------------------------- A3: lm_head OOM

class _Out:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _VocabModel:
    """Stands in for Cohere2ForCausalLM: full-sequence logits ([B, T, 262k]) OOM a T4, as on Kaggle."""

    def __init__(self, n_layers=4):
        self.n_layers, self.calls = n_layers, []

    def __call__(self, input_ids, attention_mask, output_hidden_states=False, logits_to_keep=0):
        self.calls.append({"output_hidden_states": output_hidden_states, "logits_to_keep": logits_to_keep})
        if logits_to_keep != 1:
            raise MemoryError("CUDA out of memory. Tried to allocate 3.05 GiB (logits * logit_scale)")
        return _Out(tuple(f"h{l}" for l in range(self.n_layers + 1)))  # embeddings + n layers (last = post-norm)


def test_a3_activation_pass_keeps_only_last_logit():
    from aftermarket.attacks.abliterate import layer_inputs

    model = _VocabModel(n_layers=4)
    hs = layer_inputs(model, input_ids=[[1, 2, 3]], attention_mask=[[1, 1, 1]])
    assert model.calls == [{"output_hidden_states": True, "logits_to_keep": 1}]
    assert hs == ("h0", "h1", "h2", "h3")  # input to each decoder layer; post-norm output dropped


# ----------------------------------------------------------------------------- A4: HqqConfig in transformers 5.x

class _Linear:
    def __init__(self, name):
        self.name = name


class _Module:
    pass


class _FakeModel:
    def __init__(self, names):
        self.root = _Module()
        self._names = names
        for n in names:
            obj = self.root
            parts = n.split(".")
            for p in parts[:-1]:
                if not hasattr(obj, p):
                    setattr(obj, p, _Module())
                obj = getattr(obj, p)
            setattr(obj, parts[-1], _Linear(n))

    def named_modules(self):
        for n in self._names:
            yield n, self.get_submodule(n)

    def get_submodule(self, name):
        obj = self.root
        for p in name.split("."):
            obj = getattr(obj, p)
        return obj

    def get_memory_footprint(self):
        return 0


@pytest.fixture
def fake_stack(monkeypatch):
    """Fake torch / hqq / transformers 5.x (whose HQQ loader raises, as on Kaggle) and a fake load_model."""
    torch = types.ModuleType("torch")
    torch.float16 = "fp16"
    torch.nn = types.SimpleNamespace(Linear=_Linear)
    torch.cuda = types.SimpleNamespace(reset_peak_memory_stats=lambda: None, memory_allocated=lambda: 0,
                                       max_memory_allocated=lambda: 0, empty_cache=lambda: None)
    quantized = []

    class HQQLinear:
        def __init__(self, layer, quant_config, compute_dtype, device, del_orig):
            quantized.append((layer.name, quant_config["nbits"], quant_config["group_size"]))

    q = types.ModuleType("hqq.core.quantize")
    q.HQQLinear = HQQLinear
    q.BaseQuantizeConfig = lambda **kw: kw

    def _unsupported(*a, **k):
        raise NotImplementedError("QuantizationMethod.HQQ is not available yet and will be supported soon.")

    tf = types.ModuleType("transformers")
    tf.HqqConfig = _unsupported
    tf.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=_unsupported)
    for name, mod in {"torch": torch, "hqq": types.ModuleType("hqq"), "hqq.core": types.ModuleType("hqq.core"),
                      "hqq.core.quantize": q, "transformers": tf}.items():
        monkeypatch.setitem(sys.modules, name, mod)

    import aftermarket.load as load

    names = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj",
             "model.layers.1.self_attn.q_proj", "lm_head"]
    specs = {"BASE": {"format": "template"}, "AM": {"format": "full",
                                                    "quant_shield": {"bits": 8, "group_size": 128, "layers": [1]}}}
    monkeypatch.setattr(load, "get_spec", lambda mid: specs[mid])
    monkeypatch.setattr(load, "load_model", lambda mid, precision, template: (_FakeModel(names), "tok"))
    return quantized


def test_a4_quantizes_without_transformers_hqq(fake_stack):
    from aftermarket.attacks.compress import load_hqq

    _, _, info = load_hqq("BASE", "native", 3, 64)
    assert fake_stack == [("model.layers.0.self_attn.q_proj", 3, 64), ("model.layers.0.mlp.down_proj", 3, 64),
                          ("model.layers.1.self_attn.q_proj", 3, 64)]  # lm_head stays fp16
    assert info["linear_counts_by_bits"] == {"3": 3} and info["attack"] == "A4q3"


def test_a4_shield_keeps_its_layers_at_shield_bits(fake_stack):
    from aftermarket.attacks.compress import load_hqq

    _, _, info = load_hqq("AM", "native", 2, 64)
    assert ("model.layers.1.self_attn.q_proj", 8, 128) in fake_stack
    assert ("model.layers.0.mlp.down_proj", 2, 64) in fake_stack
    assert info["linear_counts_by_bits"] == {"2": 2, "8": 1}


# ----------------------------------------------------------------------------- end-of-stage FAILED summary

def test_one_line_error_first_line_and_cap():
    e = RuntimeError("CUDA out of memory. Tried to allocate 3.05 GiB.\nmore detail")
    assert one_line_error(e) == "RuntimeError: CUDA out of memory. Tried to allocate 3.05 GiB."
    assert len(one_line_error(RuntimeError("x" * 1000))) == 300
    assert one_line_error(KeyError()) == "KeyError"


def test_fail_line_roundtrip_through_gpu_prefix():
    import json

    line = "[gpu0] " + FAIL_MARKER + json.dumps({"jobs": ["BASE__A3__harmful_dev", "BASE__A3__mmlu_dev"],
                                                  "error": "OutOfMemoryError: CUDA out of memory."}) + "\n"
    assert parse_fail_line(line) == {"BASE__A3__harmful_dev": "OutOfMemoryError: CUDA out of memory.",
                                     "BASE__A3__mmlu_dev": "OutOfMemoryError: CUDA out of memory."}
    assert parse_fail_line("[gpu0] [BASE/A3] FAILED:\n") == {}
    assert parse_fail_line(FAIL_MARKER + "{not json") == {}


def test_failure_summary_one_line_per_failed_job():
    a3, a4 = Job("BASE", "A3", "harmful_dev"), Job("SDT", "A4q2", "mmlu_dev")
    lines = failure_summary([a3, a4], {a3.name: "OutOfMemoryError: CUDA out of memory."}, {0: 0, 1: 0})
    assert lines[0] == "FAILED BASE__A3__harmful_dev: OutOfMemoryError: CUDA out of memory."
    assert lines[1].startswith("FAILED SDT__A4q2__mmlu_dev: no exception reported")
    assert lines[-1] == "2 job(s) FAILED"
    assert failure_summary([], {}, {0: 0}) == []
