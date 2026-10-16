import hashlib
import json
import os

import pytest
from utils import ServerPreset


@pytest.fixture(scope="session", autouse=True)
def load_server_presets():
    # These tests use only the offline tinyllama preset or an explicit local fixture.
    pass


@pytest.fixture
def cache_server(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_CTX_CHECKPOINTS", "0")
    monkeypatch.setenv("LLAMA_ARG_NO_CACHE_DISK", "1")
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "none")
    monkeypatch.delenv("LLAMA_ARG_EVAL_CACHE", raising=False)
    monkeypatch.delenv("LLAMA_ARG_EVAL_CACHE_BUILD", raising=False)

    def start(name, batch=8192, ubatch=512, folder=None, build=False):
        server = ServerPreset.tinyllama2()
        model = os.environ.get("LLAMA_TEST_EVAL_CACHE_MODEL_FILE")
        if model:
            server.model_file = model
            server.model_hf_repo = None
            server.model_hf_file = None
        server.n_ctx = 8192
        server.n_slots = 2
        server.n_batch = batch
        server.n_ubatch = ubatch
        server.n_threads = 2
        server.n_gpu_layer = 0
        server.cache_ram = 0
        server.server_slots = True
        server.slot_save_path = str(tmp_path)
        server.log_path = str(tmp_path / f"{name}.log")
        if folder is None:
            monkeypatch.delenv("LLAMA_ARG_EVAL_CACHE", raising=False)
        else:
            monkeypatch.setenv("LLAMA_ARG_EVAL_CACHE", str(folder))
        monkeypatch.setenv("LLAMA_ARG_EVAL_CACHE_BUILD", "1" if build else "0")
        server.start()
        return server

    return start


def complete(server, n_tokens, slot=0, save_at=0):
    response = server.make_request("POST", "/completion", data={
        "prompt": [5 + i % 24 for i in range(n_tokens)],
        "id_slot": slot,
        "temperature": 0,
        "seed": 42,
        "n_predict": 16,
        "ignore_eos": True,
        "return_tokens": True,
        "cache_prompt": save_at == 0,
        "eval_cache_save_at": save_at,
    })
    assert response.status_code == 200, response.body
    return response.body


def entries(folder):
    return json.loads((folder / "eval-cache.json").read_text())["entries"]


def snapshot(folder):
    return {str(path.relative_to(folder)): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in folder.rglob("*") if path.is_file()}


@pytest.mark.parametrize("n_tokens", [2003, 2048, 2049])
def test_eval_cache_large_batch(cache_server, tmp_path, n_tokens):
    fresh = cache_server("fresh")
    reference = complete(fresh, n_tokens)
    fresh.stop()
    assert reference["timings"]["prompt_n"] == n_tokens
    assert len(reference["tokens"]) == 16

    folder = tmp_path / "cache"
    builder = cache_server("builder", folder=folder, build=True)
    built = complete(builder, n_tokens, slot=1, save_at=n_tokens)
    builder.stop()
    assert built["tokens"] == reference["tokens"]
    saved = entries(folder)
    assert len(saved) == 1
    saved_n = (n_tokens - 1) // 512 * 512
    assert saved[0]["n_tokens"] == saved_n
    before = snapshot(folder)

    cached = cache_server("cached", folder=folder)
    result = complete(cached, n_tokens)
    cached.stop()
    assert "eval cache: restored entry" in (tmp_path / "cached.log").read_text()
    assert result["timings"]["cache_n"] == saved_n
    assert result["timings"]["prompt_n"] == n_tokens - saved_n
    assert result["tokens"] == reference["tokens"]
    assert result["content"] == reference["content"]
    assert snapshot(folder) == before

    builder = cache_server("rebuild", folder=folder, build=True)
    complete(builder, n_tokens, slot=1, save_at=n_tokens)
    builder.stop()
    assert snapshot(folder) == before


@pytest.mark.parametrize("n_tokens", [2003, 2048])
def test_eval_cache_small_batch_entries(cache_server, tmp_path, n_tokens):
    folder = tmp_path / "cache"
    builder = cache_server("builder", batch=256, ubatch=128, folder=folder, build=True)
    complete(builder, n_tokens, save_at=n_tokens)
    builder.stop()
    # The original saver writes only the start of the last logical batch.
    saved = entries(folder)
    assert len(saved) == 1
    assert saved[0]["n_tokens"] == (n_tokens - 1) // 256 * 256


@pytest.mark.parametrize("n_tokens", [1, 511, 512, 513])
def test_eval_cache_logits_tail(cache_server, tmp_path, n_tokens):
    folder = tmp_path / "cache"
    builder = cache_server("builder", folder=folder, build=True)
    complete(builder, n_tokens, save_at=n_tokens)
    builder.stop()
    saved = entries(folder)
    if n_tokens <= 512:
        assert saved == []
    else:
        assert [entry["n_tokens"] for entry in saved] == [512]


def test_eval_cache_shared_prefix(cache_server, tmp_path):
    folder = tmp_path / "cache"
    builder = cache_server("builder", folder=folder, build=True)
    complete(builder, 2003, save_at=1200)
    builder.stop()
    assert [entry["n_tokens"] for entry in entries(folder)] == [1024]


def test_eval_cache_save_disabled(cache_server, tmp_path):
    folder = tmp_path / "cache"
    builder = cache_server("builder", folder=folder, build=True)
    complete(builder, 2003)
    builder.stop()
    assert entries(folder) == []
