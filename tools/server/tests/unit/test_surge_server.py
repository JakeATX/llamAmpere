import http.client
import json
import os
import time
import pytest
from utils import *

# opt-in server flags, all off by default:
# - LLAMA_SERVER_JSON_KEEPALIVE=S: a non-streaming reply that is not ready after S seconds sends the 200 headers and
#   a space every S seconds until the JSON body (whitespace before a JSON value is valid JSON)
# - LLAMA_SERVER_THINK_LOOP_CLOSE=N: N repeats of one token inside an open think block close the block. Needs a
#   model with a think template: set LLAMA_TEST_THINK_MODEL_FILE to a local GGUF, otherwise the test is skipped

server: ServerProcess

BUSY_N_PREDICT = 6000


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    if "LLAMA_TEST_MODEL_FILE" in os.environ:
        server.model_hf_repo = None
        server.model_hf_file = None
        server.model_file = os.environ["LLAMA_TEST_MODEL_FILE"]
    server.n_ctx = 16384
    server.n_slots = 1
    server.n_predict = -1
    server.server_slots = True
    yield
    for instance in set(server_instances):
        instance.stop()


def _post_raw(path: str, data: dict, timeout: float = 120) -> tuple[int, str, str]:
    conn = http.client.HTTPConnection(server.server_host, server.server_port, timeout=timeout)
    conn.request("POST", path, body=json.dumps(data), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read().decode()
    conn.close()
    return resp.status, resp.getheader("Content-Type") or "", body


def _queued_nonstream_body() -> tuple[int, str, str]:
    """occupy the only slot with a long stream, then send a non-streaming request that has to wait for it"""
    conn = http.client.HTTPConnection(server.server_host, server.server_port, timeout=120)
    conn.request("POST", "/completion", body=json.dumps({
        "prompt": "Once upon a time", "n_predict": BUSY_N_PREDICT, "ignore_eos": True, "stream": True,
    }), headers={"Content-Type": "application/json"})
    busy = conn.getresponse()
    assert busy.status == 200
    busy.readline()

    import threading
    out = {}
    def run():
        out["r"] = _post_raw("/completion", {"prompt": "The cat", "n_predict": 8, "temperature": 0})
    t = threading.Thread(target=run)
    t.start()
    time.sleep(3.5)
    conn.close()  # disconnect frees the slot, the queued request then runs
    t.join(60)
    assert "r" in out
    return out["r"]


def test_json_keepalive_default_off():
    server.start()
    status, ctype, body = _queued_nonstream_body()
    assert status == 200
    assert body[:1] == "{"


def test_json_keepalive_queued_nonstream(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_JSON_KEEPALIVE", "1")
    server.start()
    status, ctype, body = _queued_nonstream_body()
    assert status == 200
    assert "application/json" in ctype
    n_spaces = len(body) - len(body.lstrip(" "))
    assert n_spaces >= 1, "no keep-alive spaces before the body"
    res = json.loads(body)
    assert res["content"]


def test_json_keepalive_fast_reply_unchanged(monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_JSON_KEEPALIVE", "5")
    server.start()
    status, _, body = _post_raw("/completion", {"prompt": "The cat", "n_predict": 8, "temperature": 0})
    assert status == 200
    assert body[:1] == "{"
    # an error before the interval keeps its HTTP status
    status, _, body = _post_raw("/completion", {"prompt": "word " * 20000, "n_predict": 4})
    assert status == 400
    assert "error" in json.loads(body)


@pytest.mark.skipif("LLAMA_TEST_THINK_MODEL_FILE" not in os.environ, reason="needs a local model with a think template")
@pytest.mark.parametrize("close_n", [None, 256])
def test_think_loop_guard(monkeypatch, close_n):
    global server
    if close_n is not None:
        monkeypatch.setenv("LLAMA_SERVER_THINK_LOOP_CLOSE", str(close_n))
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = os.environ["LLAMA_TEST_THINK_MODEL_FILE"]
    server.jinja = True
    server.start(timeout_seconds=300)
    tok = server.make_request("POST", "/tokenize", data={"content": " apple"}).body["tokens"]
    assert len(tok) == 1
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 800, "temperature": 0, "logit_bias": [[tok[0], 100]],
    }, timeout=600)
    assert res.status_code == 200
    msg = res.body["choices"][0]["message"]
    closed = res.body.get("timings", {}).get("think_loop_closed", False)
    if close_n is None:
        # the repeat stop ends the reply inside the think block: no answer
        assert not closed
        assert not msg.get("content")
    else:
        # the think block is closed after about close_n repeats and the answer follows
        assert closed
        n_think = len((msg.get("reasoning_content") or "").split())
        assert close_n <= n_think <= close_n + 16
        assert msg.get("content")
