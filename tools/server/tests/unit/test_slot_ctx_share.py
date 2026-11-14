import os
import time
import pytest
from utils import *

# --slot-ctx with 0 entries (D-share): slot 0 reserves its cap, the shared slots split the rest of one unified pool.
# pool 4096, reserved 2048, 2 shared slots: shared budget = 4096 - 2048 - 2 x 256 margin = 1536

server: ServerProcess

POOL, LONG, BUDGET = 4096, 2048, 1536


@pytest.fixture(autouse=True)
def create_server(tmp_path, monkeypatch):
    global server
    # no prompt cache (RAM or disk tier): a cached prompt from an earlier test would hide the prefill counts below
    monkeypatch.setenv("LLAMA_ARG_NO_CACHE_DISK", "1")
    monkeypatch.setenv("LLAMA_ARG_CACHE_RAM", "0")
    server = ServerPreset.tinyllama2()
    server.n_ctx = POOL
    server.n_slots = 3
    server.kv_unified = True
    server.slot_ctx = f"{LONG},0,0"
    server.slot_save_path = str(tmp_path)
    server.server_slots = True
    server.temperature = 0.0


def n_tokens(text: str) -> int:
    res = server.make_request("POST", "/tokenize", data={"content": text})
    assert res.status_code == 200
    return len(res.body["tokens"])


def prompt_of(n: int, seed: int = 0) -> str:
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


def complete(prompt: str, id_slot: int = -1, n_predict: int = 8, ignore_eos: bool = False):
    data = {"prompt": prompt, "n_predict": n_predict, "cache_prompt": True, "temperature": 0.0, "top_k": 1,
            "ignore_eos": ignore_eos, "loop_guard": not ignore_eos}  # long runs: the tiny model repeats, keep it going
    if id_slot >= 0:
        data["id_slot"] = id_slot
    return server.make_request("POST", "/completion", data=data)


def slots():
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    return res.body


def test_share_caps():
    server.start()
    assert [s["n_ctx"] for s in slots()] == [LONG, BUDGET, BUDGET]
    res = server.make_request("GET", "/props")
    assert res.body["slot_n_ctx"] == [LONG, BUDGET, BUDGET]


def test_share_one_user_takes_the_whole_budget():
    # a fixed split would give each shared slot 768; alone, one shared user gets the whole 1536
    server.start()
    res = complete(prompt_of(1300), n_predict=16)
    assert res.status_code == 200
    assert res.body["id_slot"] in (1, 2)
    assert res.body["timings"]["prompt_n"] >= 1200


def test_share_short_requests_never_take_reserved_slot():
    server.start()
    res = complete(prompt_of(1300), id_slot=1, n_predict=4)
    assert res.status_code == 200
    # fits the shared budget: goes to a shared slot (slot 1's idle cells are reclaimed), never to slot 0
    res = complete(prompt_of(1000, seed=3), n_predict=4)
    assert res.status_code == 200
    assert res.body["id_slot"] in (1, 2)
    # larger than the shared budget but within the reserved cap: the reserved slot takes it
    res = complete(prompt_of(1800, seed=5), n_predict=4)
    assert res.status_code == 200
    assert res.body["id_slot"] == 0


def test_share_reserved_slot_never_purged():
    server.start()
    long_p = prompt_of(1500, seed=1)
    res = complete(long_p, id_slot=0, n_predict=4)
    assert res.status_code == 200
    n_long = slots()[0]["n_prompt_tokens"]
    assert n_long >= 1400
    # shared users fill and overflow the shared budget, one after another and at once
    res = complete(prompt_of(1300, seed=2), id_slot=1, n_predict=4)
    assert res.status_code == 200
    res = complete(prompt_of(1300, seed=4), id_slot=2, n_predict=4)   # needs slot 1's idle cells: slot 1 is cleared
    assert res.status_code == 200
    assert slots()[1]["n_prompt_tokens"] == 0
    # two shared users at once, each asking for more than the whole shared budget: whatever the overlap, each stops
    # with truncated (finish_reason length) inside the shared room, never an error, never a purge of slot 0
    results = parallel_function_calls([(complete, (prompt_of(700, seed=s), s, 2000, True)) for s in (1, 2)])
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["truncated"], res.body
        assert res.body["timings"]["prompt_n"] + res.body["timings"]["predicted_n"] <= BUDGET + 8
    assert sum(s["n_prompt_tokens"] for s in slots()[1:]) <= BUDGET
    # the reserved slot kept every cell: the long user continues without a re-prefill
    assert slots()[0]["n_prompt_tokens"] == n_long
    res = complete(long_p, id_slot=0, n_predict=4)
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] <= 2


def test_share_pinned_request_waits_for_room():
    server.start()
    # slot 1 decodes a long reply; a pinned slot-2 request that only fits once slot 1 is done waits, then succeeds
    results = parallel_function_calls([
        (complete, (prompt_of(900, seed=1), 1, 300)),
        (complete, (prompt_of(1000, seed=2), 2, 8)),
    ])
    for res in results:
        assert res.status_code == 200


def test_share_identity_vs_fixed_slot_ctx():
    # the same shared-slot request at equal fill: D-share vs fixed caps give identical greedy tokens
    global server
    outs = []
    p = None
    for slot_ctx, n_ctx in [(f"{LONG},0,0", POOL), (f"{LONG},{BUDGET},{BUDGET}", LONG + 2 * BUDGET)]:
        server = ServerPreset.tinyllama2()
        server.n_ctx = n_ctx
        server.n_slots = 3
        server.kv_unified = True
        server.slot_ctx = slot_ctx
        server.temperature = 0.0
        server.start()
        if p is None:
            p = prompt_of(1000)
        res = complete(p, id_slot=1, n_predict=64)
        assert res.status_code == 200
        outs.append(res.body["content"])
        server.stop()
    assert outs[0] == outs[1]


def test_share_needs_explicit_pool():
    server.n_ctx = None
    with pytest.raises(RuntimeError):
        server.start()


def test_share_no_room_left_refused():
    server.n_ctx = LONG + 256
    with pytest.raises(RuntimeError):
        server.start()


def test_share_kill_switch_refuses_zero(monkeypatch):
    monkeypatch.setenv("LLAMA_SLOT_CTX_SHARE", "0")
    with pytest.raises(RuntimeError):
        server.start()


def later(delay_s: float, fn, *args):
    time.sleep(delay_s)
    return fn(*args)


def test_share_admitted_user_never_errors():
    # slot 2 idles with an old 1300-token conversation; two new shared users start at once on slots 1 and 2: the old
    # cells are trimmed or reclaimed, both requests succeed (200), none gets a context-size error
    server.start()
    res = complete(prompt_of(1300, seed=4), id_slot=2, n_predict=4)
    assert res.status_code == 200
    for order in ((1, 2), (2, 1)):
        results = parallel_function_calls([
            (complete, (prompt_of(600, seed=order[0]), order[0], 2000, True)),
            (complete, (prompt_of(500, seed=order[1] + 5), order[1], 8)),
        ])
        for res in results:
            assert res.status_code == 200, res.body
        assert sum(s["n_prompt_tokens"] for s in slots()[1:]) <= BUDGET + 8


def test_share_full_room_short_waits_not_reserved():
    # slot 0 idles with its long conversation; slot 1 runs until it holds the whole shared room; an unpinned short
    # request arriving meanwhile waits for shared room and never takes the reserved slot
    server.start()
    res = complete(prompt_of(1500, seed=1), id_slot=0, n_predict=4)
    assert res.status_code == 200
    n_long = slots()[0]["n_prompt_tokens"]
    results = parallel_function_calls([
        (complete, (prompt_of(700, seed=2), 1, 2000, True)),
        (later, (0.3, complete, prompt_of(1000, seed=3), -1, 8)),
    ])
    for res in results:
        assert res.status_code == 200, res.body
    assert results[0].body["truncated"]
    assert results[1].body["id_slot"] in (1, 2)
    assert slots()[0]["n_prompt_tokens"] == n_long


def test_share_refuses_context_shift():
    server.enable_ctx_shift = True
    with pytest.raises(RuntimeError):
        server.start()
