import pytest
from utils import *

# --slot-ctx: one unified KV pool, every slot reserves its own cap (slot 0 = long slot)

server: ServerProcess

CAPS = [1024, 512, 512]


@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = None  # the pool is sized to sum(CAPS)
    server.n_slots = len(CAPS)
    server.kv_unified = True
    server.slot_ctx = ",".join(str(c) for c in CAPS)
    server.slot_save_path = str(tmp_path)
    server.server_slots = True
    server.temperature = 0.0


def n_tokens(text: str) -> int:
    res = server.make_request("POST", "/tokenize", data={"content": text})
    assert res.status_code == 200
    return len(res.body["tokens"])


def prompt_of(n: int, seed: int = 0) -> str:
    # about n tokens of varied text (tokenize to land under n)
    words = ["cat", "dog", "tree", "house", "river", "sun", "boat", "girl", "boy", "ball", "bird", "cake"]
    out = []
    i = seed
    while True:
        out.append(words[i % len(words)])
        i += 7
        if len(out) % 32 == 0 and n_tokens(" ".join(out)) >= n:
            break
    while n_tokens(" ".join(out)) > n:
        out.pop()
    return " ".join(out)


def complete(prompt: str, id_slot: int = -1, n_predict: int = 8):
    data = {"prompt": prompt, "n_predict": n_predict, "cache_prompt": True, "temperature": 0.0, "top_k": 1}
    if id_slot >= 0:
        data["id_slot"] = id_slot
    return server.make_request("POST", "/completion", data=data)


def slot_tokens(id_slot: int) -> int:
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    return res.body[id_slot].get("n_prompt_tokens", 0)


def test_slot_ctx_caps_and_props():
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["slot_n_ctx"] == CAPS
    assert res.body["default_generation_settings"]["n_ctx"] == max(CAPS)
    res = server.make_request("GET", "/slots")
    assert [s["n_ctx"] for s in res.body] == CAPS


def test_pinned_short_slot_errors_long_slot_fits():
    server.start()
    p = prompt_of(700)
    res = complete(p, id_slot=1)
    assert res.status_code == 400
    assert "exceeds the available context size" in res.body["error"]["message"]
    res = complete(p, id_slot=0)
    assert res.status_code == 200
    assert res.body["id_slot"] == 0


def test_long_prompt_routes_to_long_slot_despite_lcp():
    server.start()
    base = prompt_of(300)
    res = complete(base, id_slot=1)
    assert res.status_code == 200
    # shares a 300-token prefix with slot 1, but only slot 0 can hold it
    res = complete(base + " " + prompt_of(400, seed=3))
    assert res.status_code == 200
    assert res.body["id_slot"] == 0


def test_short_requests_never_take_long_slot():
    server.start()
    p = prompt_of(200)
    for i in range(4):
        res = complete(p + f" {i}")
        assert res.status_code == 200
        assert res.body["id_slot"] in (1, 2)
    # three at once: the third waits for a short slot instead of taking slot 0
    results = parallel_function_calls([(complete, (prompt_of(200, seed=s), -1, 200)) for s in range(3)])
    for res in results:
        assert res.status_code == 200
        assert res.body["id_slot"] in (1, 2)


def test_busy_long_slot_defers_not_errors():
    server.start()
    results = parallel_function_calls([(complete, (prompt_of(700, seed=s), -1, 200)) for s in range(2)])
    for res in results:
        assert res.status_code == 200
        assert res.body["id_slot"] == 0


def test_headroom_is_fixed_not_n_predict():
    # a large n_predict does not push a short prompt onto the long slot
    server.start()
    res = complete(prompt_of(200), n_predict=2000)
    assert res.status_code == 200
    assert res.body["id_slot"] in (1, 2)
    # a prompt that fits only the long slot (less than 64 tokens of room) goes there instead of a 400
    res = complete(prompt_of(1000, seed=2), n_predict=8)
    assert res.status_code == 200
    assert res.body["id_slot"] == 0


def test_short_request_keeps_long_slot_cache():
    # the idle-slot clear of a unified pool is off with --slot-ctx: the long slot keeps its cells
    server.start()
    long_p = prompt_of(700)
    res = complete(long_p)
    assert res.status_code == 200 and res.body["id_slot"] == 0
    n_long = slot_tokens(0)
    assert n_long > 0
    res = complete(prompt_of(200, seed=5))
    assert res.status_code == 200 and res.body["id_slot"] in (1, 2)
    assert slot_tokens(0) == n_long


def test_restore_long_state_into_short_slot_fails_clean():
    server.start()
    res = complete(prompt_of(700), id_slot=0)
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "long.bin"})
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/1?action=restore", data={"filename": "long.bin"})
    assert res.status_code != 200
    assert slot_tokens(1) == 0
    # the slot still works after the failed restore
    res = complete(prompt_of(200), id_slot=1)
    assert res.status_code == 200
    # and the state restores into a slot whose cap fits
    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "long.bin"})
    assert res.status_code == 200


def test_symmetric_slot_ctx_matches_unset():
    # same pool, flag unset vs --slot-ctx C,C,C: greedy output must be identical
    global server
    p = None
    outs = []
    for slot_ctx in [None, "512,512,512"]:
        server = ServerPreset.tinyllama2()
        server.n_ctx = 1536
        server.n_slots = 3
        server.kv_unified = True
        server.slot_ctx = slot_ctx
        server.temperature = 0.0
        server.start()
        if p is None:
            p = prompt_of(300)
        res = complete(p, id_slot=1, n_predict=64)
        assert res.status_code == 200
        outs.append(res.body["content"])
        server.stop()
    assert outs[0] == outs[1]


@pytest.mark.parametrize("bad", [
    {"kv_unified": False},
    {"slot_ctx": "1024,512"},
    {"n_ctx": 1024},
])
def test_slot_ctx_refuses_bad_config(bad):
    for k, v in bad.items():
        setattr(server, k, v)
    with pytest.raises(RuntimeError):
        server.start()
