import http.client
import json
import os
import socket
import subprocess
import threading
import time
import pytest
from utils import *

# server robustness (backlog row #32): queued-request SSE keep-alive (--sse-ping-queued, off by default),
# disconnect cleanup while other slots produce results, slot stall detector options, clean exit on a model
# load failure. timings rely on the tiny model being slow at long generations (attention over the whole
# context on the CPU), so the busy request keeps its slot for several seconds

server: ServerProcess

BUSY_N_PREDICT = 6000


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    # offline runs: LLAMA_TEST_MODEL_FILE points at a local copy of the stories260K model
    if "LLAMA_TEST_MODEL_FILE" in os.environ:
        server.model_hf_repo = None
        server.model_hf_file = None
        server.model_file = os.environ["LLAMA_TEST_MODEL_FILE"]
    server.n_ctx = 16384
    server.n_slots = 1
    server.n_predict = -1
    server.sse_ping_interval = 1
    server.server_slots = True
    yield
    # the suite conftest stops servers too; this keeps the module runnable with --noconftest (offline runs)
    for instance in set(server_instances):
        instance.stop()


def _open_stream(path: str, data: dict) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse, float]:
    """POST and return when the response headers arrive, with the time they arrived"""
    conn = http.client.HTTPConnection(server.server_host, server.server_port, timeout=120)
    conn.request("POST", path, body=json.dumps(data), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return conn, resp, time.time()


def _read_events(resp: http.client.HTTPResponse) -> list[str]:
    """read SSE lines until the stream ends: ':' pings and 'data: ...' events"""
    lines = []
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.decode("utf-8").rstrip("\n")
        if line:
            lines.append(line)
        if line.startswith("data: ") and ('"stop":true' in line.replace(" ", "") or "[DONE]" in line or '"error"' in line):
            break
    return lines


def _slots_busy() -> list[bool]:
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    return [bool(s["is_processing"]) for s in res.body]


def _wait_until(cond, timeout: float) -> bool:
    t_end = time.time() + timeout
    while time.time() < t_end:
        if cond():
            return True
        time.sleep(0.05)
    return False


class BusyRequest:
    """a long non-streaming generation that occupies one slot, run on a thread"""

    def __init__(self, n_predict: int = BUSY_N_PREDICT):
        self.t_done = None
        self.status = None
        self.n_predict = n_predict
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        res = server.make_request("POST", "/completion", data={
            "prompt": "Once upon a time",
            "n_predict": self.n_predict,
            "ignore_eos": True,
        }, timeout=600)
        self.status = res.status_code
        self.t_done = time.time()

    def start(self):
        self.thread.start()
        assert _wait_until(lambda: any(_slots_busy()), 10), "busy request never started"

    def running(self) -> bool:
        return self.t_done is None


def test_queued_stream_default_waits_for_slot():
    """default (no --sse-ping-queued): a queued streaming request gets nothing until its slot starts"""
    global server
    server.start()
    busy = BusyRequest()
    busy.start()

    t0 = time.time()
    conn, resp, t_headers = _open_stream("/completion", {"prompt": "Hello", "n_predict": 8, "stream": True})
    assert resp.status == 200
    events = _read_events(resp)
    conn.close()

    busy.thread.join(timeout=600)
    assert busy.status == 200
    # the headers came only once the busy request had finished and freed the slot
    assert t_headers - t0 > 2.0
    assert t_headers >= busy.t_done - 0.5
    assert not any(line.startswith(":") and t_headers < busy.t_done for line in events)
    assert any(line.startswith("data: ") for line in events)


def test_queued_stream_ping_queued_keepalive():
    """--sse-ping-queued: a queued stream gets HTTP 200 and pings while it waits, then its tokens"""
    global server
    server.sse_ping_queued = True
    server.start()
    busy = BusyRequest()
    busy.start()

    t0 = time.time()
    conn, resp, t_headers = _open_stream("/completion", {"prompt": "Hello", "n_predict": 8, "stream": True})
    assert resp.status == 200
    assert resp.getheader("Content-Type", "").startswith("text/event-stream")
    # headers after about one ping interval, while the busy request still holds the only slot
    assert t_headers - t0 < 5.0
    assert busy.running()
    first = resp.readline().decode("utf-8")
    assert first.startswith(":")

    events = _read_events(resp)
    conn.close()
    busy.thread.join(timeout=600)
    assert busy.status == 200

    data = [json.loads(line[6:]) for line in events if line.startswith("data: ")]
    assert data, "no data events after the pings"
    # no payload-less event (the begin partial must not be sent as 'data: null')
    assert all(d is not None for d in data)
    assert data[-1].get("stop") is True
    assert data[-1].get("tokens_predicted", 0) > 0


def test_queued_stream_ping_queued_oai_done():
    """--sse-ping-queued on the OAI endpoint: pings, chunks, then [DONE], and no 'data: null'"""
    global server
    server.sse_ping_queued = True
    server.start()
    busy = BusyRequest()
    busy.start()

    conn, resp, _ = _open_stream("/v1/completions", {"prompt": "Hello", "max_tokens": 8, "stream": True})
    assert resp.status == 200
    assert busy.running()
    lines = []
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.decode("utf-8").strip()
        if line:
            lines.append(line)
        if line == "data: [DONE]":
            break
    conn.close()
    busy.thread.join(timeout=600)

    assert lines[0].startswith(":")
    assert lines[-1] == "data: [DONE]"
    assert "data: null" not in lines
    chunks = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
    assert chunks and all("choices" in c for c in chunks)


def test_queued_stream_ping_queued_error_event():
    """--sse-ping-queued: an error raised when a queued stream starts is an SSE error event after HTTP 200"""
    global server
    server.sse_ping_queued = True
    server.start()
    busy = BusyRequest()
    busy.start()

    # longer than the 16384-token context: the error is raised when the request leaves the queue
    conn, resp, _ = _open_stream("/completion", {"prompt": "hello " * 20000, "n_predict": 8, "stream": True})
    assert resp.status == 200
    assert busy.running()
    events = _read_events(resp)
    conn.close()
    busy.thread.join(timeout=600)

    assert events[0].startswith(":")
    errors = [json.loads(line[6:]) for line in events if line.startswith("data: ") and '"error"' in line]
    assert len(errors) == 1
    assert errors[0]["error"]["code"] == 400


@pytest.mark.parametrize("ping_queued", [False, True])
def test_idle_stream_error_is_plain_http_error(ping_queued: bool):
    """an error raised before the first ping interval stays a plain HTTP error, with or without --sse-ping-queued"""
    global server
    server.sse_ping_queued = ping_queued
    server.n_ctx = 2048
    server.start()
    conn, resp, _ = _open_stream("/completion", {"prompt": "hello " * 4000, "n_predict": 8, "stream": True})
    assert resp.status == 400
    body = json.loads(resp.read())
    conn.close()
    assert body["error"]["type"] == "exceed_context_size_error"


def test_nonstream_disconnect_frees_slot_while_other_slot_streams():
    """a non-streaming client that disconnects is cancelled within about a second, even while another slot keeps
    producing results (before the recv_with_timeout deadline fix it ran to completion)"""
    global server
    server.n_slots = 2
    server.n_ctx = 32768
    server.start()

    # slot A: a long streaming generation that produces results continuously
    conn_a, resp_a, _ = _open_stream("/completion", {
        "prompt": "Once upon a time", "n_predict": BUSY_N_PREDICT, "ignore_eos": True, "stream": True,
    })
    assert resp_a.status == 200
    resp_a.readline()

    # slot B: a long non-streaming generation over a raw socket, closed once it is running
    body = json.dumps({"prompt": "The cat", "n_predict": BUSY_N_PREDICT, "ignore_eos": True})
    sock = socket.create_connection((server.server_host, server.server_port))
    sock.sendall((
        "POST /completion HTTP/1.1\r\n"
        f"Host: {server.server_host}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n{body}"
    ).encode())
    assert _wait_until(lambda: all(_slots_busy()), 10), "both slots never became busy"
    time.sleep(0.5)
    sock.close()
    t_close = time.time()

    assert _wait_until(lambda: sum(_slots_busy()) == 1, 5), "the disconnected request kept its slot"
    t_freed = time.time() - t_close
    # the streaming request is unaffected
    assert _slots_busy().count(True) == 1
    line = resp_a.readline()
    assert line
    conn_a.close()
    print(f"disconnected request freed {t_freed:.2f} s after close")


def test_queued_stream_pings_under_slots_polling():
    """--sse-ping-queued with another client polling /slots every 50 ms: the pings keep their interval (each
    poll result used to restart the HTTP thread's 1 s wait, so should_stop() and the ping never ran)"""
    global server
    server.sse_ping_queued = True
    server.start()
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            server.make_request("GET", "/slots")
            time.sleep(0.05)

    busy = BusyRequest()
    busy.start()
    poller = threading.Thread(target=poll, daemon=True)
    poller.start()

    conn, resp, _ = _open_stream("/completion", {"prompt": "Hello", "n_predict": 8, "stream": True})
    assert resp.status == 200
    pings = []
    t0 = time.time()
    while busy.running() and time.time() - t0 < 8:
        line = resp.readline().decode("utf-8").strip()
        if line.startswith(":"):
            pings.append(time.time() - t0)
        elif line.startswith("data: "):
            break
    stop.set()
    conn.close()
    busy.thread.join(timeout=600)
    assert len(pings) >= 2, f"pings at {pings}"


def test_stall_detector_default_off(tmp_path):
    global server
    server.log_path = str(tmp_path / "server.log")
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": "Hello", "n_predict": 16})
    assert res.status_code == 200
    server.stop()
    log = open(tmp_path / "server.log").read()
    assert "slot stall detector" not in log


def test_stall_detector_no_false_positive(tmp_path):
    """--slot-stall-timeout 1: two slots sharing batches for several seconds are never reported as stalled"""
    global server
    server.log_path = str(tmp_path / "server.log")
    server.n_slots = 2
    server.slot_stall_timeout = 1
    server.slot_stall_cancel = True
    server.start()
    a = BusyRequest(n_predict=3000)
    b = BusyRequest(n_predict=3000)
    a.thread.start()
    b.thread.start()
    a.thread.join(timeout=600)
    b.thread.join(timeout=600)
    assert a.status == 200 and b.status == 200
    server.stop()
    log = open(tmp_path / "server.log").read()
    assert "slot stall detector enabled: timeout = 1 s, cancel = true" in log
    assert "no progress for" not in log


def test_missing_model_exits_cleanly(tmp_path):
    """a model that cannot be loaded ends the server with exit code 1, not an abort"""
    server_path = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    proc = subprocess.run(
        [server_path, "--model", str(tmp_path / "missing.gguf"), "--port", str(server.server_port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
    )
    out = proc.stdout.decode("utf-8", errors="replace")
    assert proc.returncode == 1, out[-2000:]
