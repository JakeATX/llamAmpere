# llamAmpere v0.5.1

A point release on top of v0.5. One change: SJ-KVaRN now works with parallel decode slots.

## SJ-KVaRN with parallel slots

v0.5 refused to start SJ-KVaRN with more than one sequence ("SJ-KVaRN rejects n_seq_max = 2"). This also hit the
server's default `--parallel` (auto), which resolves to 4 slots.

In v0.5.1:

- Each slot gets its own SJ-KVaRN stream: sink, staging ring and sealed body.
- `--kv-unified` shares one paged record pool between slots, so any slot can use the whole context.
- Slots stay isolated. 2- and 4-slot runs differ from single-slot runs only as much as plain-cache batching does
  (per-position KL about 1e-4 to 1e-3, the same as the plain-cache control).
- Prompt caching and slot save/restore work across slots, including restoring a saved slot into another slot.
- When the shared pool is full, the server returns an HTTP error instead of crashing.
- `--parallel 1` gives token-identical output to v0.5 at unchanged speed: 91.14 vs 91.59 tok/s at 100K depth,
  within noise.

## Tested

Swift 1.5 IQ4_XS-M with the MTP drafter, at SJ-KVaRN 4/4 and 3/3t, on one RTX 3090 Ti:

| Setup | Context | Peak VRAM |
|---|---|---|
| 4 slots, 3/3t | 196,608 | 23,034 MiB |
| 4 slots, 4/4 `--kv-unified` | 131,072 | 22,750 MiB |

Two slots decode about 110-119 tok/s together, against 103-107 tok/s for one slot alone (temperature 1, MTP).

## Recommended settings

For a single user, keep `--parallel 1` as in the [v0.5 recommended commands](../llamampere-v0.5/RECOMMENDED.md):
each extra slot costs one staging ring of VRAM. For several concurrent users, use `--parallel N`, and add
`--kv-unified` when slots should share the context.
