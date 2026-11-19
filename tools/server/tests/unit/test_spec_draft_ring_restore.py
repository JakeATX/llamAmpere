import os
import subprocess

import pytest
from utils import *

# [KVRESTORE] Two slots share one unified MTP draft ring (--spec-draft-window with --kv-unified). Slot 0 prefills
# while slot 1 is idle, so its context checkpoints keep the whole ring (masked cells included). Slot 1 then writes
# its own window. A slot-0 request that diverges from its cached prompt rolls back to a checkpoint; that restore
# used to fail ("failed to find N available cells") and the slot re-processed the whole prompt.
#
# The model is the generated Qwen3.5-style MTP fixture of tests/test-mtp-state.cpp (written with --save-model).
# Set LLAMA_MTP_FIXTURE to a saved fixture, or keep test-mtp-state next to the llama-server binary.

N_VOCAB = 32  # fixture vocabulary: tok_0 .. tok_31
WINDOW = 64
N_PROMPT = 400

server: ServerProcess


def fixture_model(tmp_path_factory) -> str:
    path = os.environ.get("LLAMA_MTP_FIXTURE")
    if path:
        return path
    server_bin = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    tool = os.path.join(os.path.dirname(server_bin), "test-mtp-state")
    if not os.path.exists(tool):
        pytest.skip("needs the MTP fixture: set LLAMA_MTP_FIXTURE or build test-mtp-state")
    out = tmp_path_factory.mktemp("mtp") / "mtp-fixture.gguf"
    subprocess.run([tool, "--save-model", str(out)], check=True, capture_output=True)
    return str(out)


@pytest.fixture(autouse=True)
def create_server(tmp_path_factory):
    global server
    server = ServerProcess()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = fixture_model(tmp_path_factory)
    server.model_alias = "mtp-fixture"
    server.n_ctx = 2048
    server.n_slots = 2
    server.kv_unified = True
    server.n_batch = 32
    server.n_ubatch = 32
    server.n_predict = 4
    server.temperature = 0.0
    server.spec_type = "draft-mtp"
    server.spec_draft_n_max = 3
    server.spec_draft_window = WINDOW
    server.spec_draft_window_sink = 4
    server.spec_n_rs_seq = 0  # a plain KV drafter: its checkpoint part keeps the ring layout (KEEP_LAYOUT)
    server.cache_ram = 0


def tokens(n: int, salt: int) -> list[int]:
    # skip 0..3 (control-like ids of the fixture vocabulary)
    return [4 + (i * 7 + salt) % (N_VOCAB - 4) for i in range(n)]


def complete(prompt: list[int], slot: int):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": slot,
        "cache_prompt": True,
        "n_predict": 4,
        "temperature": 0.0,
    })
    assert res.status_code == 200, res.body
    return res.body


def test_draft_ring_checkpoint_restore_on_the_slot_that_wrote_first():
    global server
    server.start()

    prompt0 = tokens(N_PROMPT, 3)
    prompt1 = tokens(N_PROMPT, 11)

    first = complete(prompt0, 0)
    assert first["timings"]["prompt_n"] == N_PROMPT

    # slot 1 writes its window into the shared ring
    complete(prompt1, 1)

    # slot 0 diverges near the end of its cached prompt: the recurrent state cannot be cut back, so the slot
    # rolls back to a context checkpoint instead of re-processing the prompt
    diverged = prompt0[:N_PROMPT - 2] + tokens(8, 17)
    again = complete(diverged, 0)
    n_new = again["timings"]["prompt_n"]
    assert n_new < len(diverged) // 2, f"slot 0 re-processed {n_new} of {len(diverged)} prompt tokens"
