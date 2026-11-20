import os
import pytest
from utils import *

# [REPEAT] the adaptive draft-vocabulary tail (--spec-draft-vocab-hot) restarts from its load-time ids at every
# request while no other slot is busy, so the drafts of a request (and with them the sampled text, and at temp 0
# the verify widths) do not depend on what the server served before. LLAMA_SPEC_VOCAB_HOT_CARRY=1 keeps the old
# cross-request memory. Needs LLAMA_TEST_MTP_MODEL_FILE = a Qwen3.8-27B file with an MTP head (the built-in
# 24576-entry shortlist matches that tokenizer only); skipped when the server log shows no hot tail.

PROMPT_A = "Write a short note about caches. " + "The cache keeps recent prompts so that repeated requests are fast. " * 20
PROMPT_B = ("Translate to French, German and Spanish, then explain each grammar point: "
            "the quick brown fox jumps over the lazy dog near the riverbank at dawn. ") * 6


def make_server(tmp_path, name: str) -> ServerProcess:
    model = os.environ.get("LLAMA_TEST_MTP_MODEL_FILE")
    if not model:
        pytest.skip("LLAMA_TEST_MTP_MODEL_FILE (a Qwen3.8-27B model with an MTP head) not set")
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = model
    s.offline = True
    s.n_ctx = 4096
    s.n_slots = 1
    s.n_predict = 96
    s.temperature = 0.0
    s.spec_type = "draft-mtp-adaptive"
    s.spec_draft_n_max = 4
    s.log_path = str(tmp_path / f"server_{name}.log")
    return s


def complete(s: ServerProcess, prompt: str, temperature: float):
    # cache_prompt false: every request prefills the same way, so only state carried by the drafter can differ
    res = s.make_request("POST", "/completion", data={"prompt": prompt, "temperature": temperature, "top_k": 20,
                                                       "top_p": 0.95, "cache_prompt": False, "seed": 7300})
    assert res.status_code == 200, res.body
    t = res.body["timings"]
    return res.body["content"], (t.get("draft_n"), t.get("draft_n_accepted"))


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_spec_vocab_hot_reset_per_request(tmp_path, monkeypatch, temperature):
    monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_VOCAB_MAP", "auto:24576")
    monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_VOCAB_HOT", "2048")
    monkeypatch.delenv("LLAMA_SPEC_VOCAB_HOT_CARRY", raising=False)
    runs = {}
    for name, order in [("boot1", ["B", "A", "B", "A"]), ("boot2", ["A", "B"])]:
        s = make_server(tmp_path, f"{name}_t{temperature}")
        s.start()
        try:
            runs[name] = [(t, *complete(s, PROMPT_A if t == "A" else PROMPT_B, temperature)) for t in order]
        finally:
            s.stop()
        with open(s.log_path) as f:
            log = f.read()
            # the per-request line only appears when the tail is live (compact draft heads, e.g. EXL3, disable
            # it); without it the test would pass without testing anything
            if "draft vocab hot: reset" not in log:
                pytest.skip("the hot tail is not active for this model (no per-request reset line)")
    ref = {}
    for name, rows in runs.items():
        for t, content, drafts in rows:
            assert drafts[0], "the MTP drafter did not run"
            if t not in ref:
                ref[t] = (content, drafts)
            # same content AND the same drafts: the hot tail held only this request's ids
            assert (content, drafts) == ref[t], f"{name}: turn {t} differs from its first run"
