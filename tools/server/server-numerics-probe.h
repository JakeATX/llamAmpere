#pragma once

// [TF233] numerics probe for the prompt-cache disk key.
//
// LLAMA_PROMPT_CACHE_NUMERICS_PROBE=1 (default off): before the disk tier opens, decode a fixed short prompt
// on sequence 0 of the target context and hash the last-token logits. The hash goes into the disk key, so a
// disk entry written under any toggle that changes the numbers (a kernel flag, a rounding change, a KV encode
// knob missing from the key's env list) is never restored into a server whose numbers differ.
// The probe runs twice: equal bits -> "bits:<hash>"; unequal bits but the same top-32 order -> "rank:<hash>"
// (a warning is logged: the model's forward is not bit-repeatable); unequal order -> a per-process key
// (nothing written by another process is restored). Sequence 0 is cleared before and after.

#include <string>

struct llama_context;

// returns true and sets `out` to the key part; false with `err` set when the probe cannot run (the key then
// carries "numerics=skipped:<why>" so the probe-on and probe-off keys never match).
bool server_numerics_probe(llama_context * ctx, std::string & out, std::string & err);

bool server_numerics_probe_enabled();
