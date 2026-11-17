# llamAmpere v0.6.4

Fix: Flash-Next ran out of memory, or stalled, on machines with two or more NVIDIA cards, and on a card that another
program was already using (#96).

v0.6.4 runs Flash-Next on one card: the one with the most free memory. Splitting Flash-Next across two cards is not
supported yet.

The Flash-Next launcher (`scripts/flash-next-server.sh`) loads a fixed one-card setup with `--fit off`. Two things went
wrong:

- **Several cards:** the layers were split across all visible cards. The fast card-side expert tables work on one card
  only, so on two idle RTX 3090s the experts ran on the host: a 28-token prompt took 55 s and 512 tokens did not finish
  in 13 minutes.
- **Busy cards:** memory that another program (for example a second AI server) already held was treated as free, so the
  fixed setup did not fit. The server loaded for a minute or more and then failed with an out-of-memory error.

What changes:

- **Several cards:** when no card is named, the launcher now picks the card with the most free memory and prints which
  one. On the same two RTX 3090s that is 79 tokens/s (512 tokens). Use `--device` (for example `--device CUDA1`) to pick
  the card yourself.
- **Busy cards:** the launcher names each card it will use when other programs hold more than 4 GiB of it. If the card
  has less than 8 GiB free, the launcher switches to the memory planner (`-ngl auto --fit on --moe-skip off`). The
  planner keeps what fits on the card and leaves the rest in system RAM. This runs, but slowly: on an RTX 3090 Ti with
  about 3.3 GB left for the server it chose a 4,096-token context and wrote 512 tokens at 3 tokens/s. Close the other
  programs for full speed.
- `FLASH_NEXT_FIT_MIN_MIB` sets the 8 GiB switch point in MiB; `FLASH_NEXT_FIT_MIN_MIB=0` keeps the fixed setup. Your
  own `-ngl`, `--fit`, `-c` or `--moe-skip` still win over the launcher's choice.

One idle card: unchanged. The launcher's arguments and settings are the same as in v0.6.3.11 (checked by
`tests/test-flash-next-launcher.sh`).

Limits:
- Two-card Flash-Next (`FLASH_NEXT_MULTI_GPU=1`, or several cards named with `--device`) is experimental and very slow
  unless the host has enough RAM for the experts; the launcher prints a warning when it is used.
- The 8 GiB switch point is a safe margin, not a measured limit.
- With the planner, the context can be as short as 4,096 tokens. Set `-c` yourself for a fixed context; if it does not
  fit, the load fails.

Dense Swift 1.5 27B and the server itself are unchanged; only the Flash-Next launcher changed. Source-only release.
