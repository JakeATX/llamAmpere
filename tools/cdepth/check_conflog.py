#!/usr/bin/env python3
"""Source check for the disabled log path. Does not load a model or start a server."""

from pathlib import Path


def function_body(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    nesting = 1
    end = opening + 1
    while nesting:
        nesting += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[opening + 1:end - 1]


def check(root):
    server = (root / "tools/server/server-context.cpp").read_text()
    path = function_body(server, "static const char * spec_conflog_path()")
    assert 'std::getenv("LLAMA_SPEC_CONFLOG")' in path
    assert "path != nullptr && path[0] != '\\0' ? path : nullptr" in path
    opening = function_body(server, "FILE * spec_conflog_open()")
    guard = opening.index("if (path == nullptr)")
    early_return = opening.index("return nullptr; // LLAMA_SPEC_CONFLOG_UNSET_NO_FILE")
    assert guard < early_return < opening.index("std::fopen")
    assert opening.count("std::fopen") == 1
    assert server.count('std::fopen(path, "a")') == 1
    assert "if (spec_conflog_path()) {\n                    spec_conflog_capture" in server
    assert "if (spec_conflog_path()) {\n                spec_conflog_write(slot, ids.size());" in server
    assert "dp.conf_p1 = &slot.spec_conf_p1" in server
    assert "dp.conf_p2 = &slot.spec_conf_p2" in server
    speculative = (root / "common/speculative.h").read_text()
    assert "std::vector<float> * conf_p1 = nullptr;" in speculative
    assert "std::vector<float> * conf_p2 = nullptr;" in speculative
    assert "int * conf_depth = nullptr;" in speculative
    # Probability extraction and file creation occur only behind the nonempty-path guards.
    capture = function_body(server, "void spec_conflog_capture(")
    assert "row - off" in capture
    assert server.count("spec_conflog_capture(slot,") == 1
    return "LLAMA_SPEC_CONFLOG_UNSET_NO_FILE: PASS (source guard inspection)"


if __name__ == "__main__":
    print(check(Path(__file__).resolve().parents[2]))
