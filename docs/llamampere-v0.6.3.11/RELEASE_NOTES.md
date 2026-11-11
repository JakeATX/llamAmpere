# llamAmpere v0.6.3.11

Fix: faster first token on short follow-up messages after a long first message (Flash-Next).

After a long first message, the card's expert tables are still being set up when the next message arrives. A short
follow-up used to wait for that, or ran its experts on the CPU. Follow-ups of up to 256 new tokens now finish the
setup and start writing sooner. Follow-ups of 257 tokens or more work as in v0.6.3.10.

Time to first token on the first follow-up after a 100K-token first message (Flash-Next AI3, shipped launcher, RTX 3090,
fixed test settings):

| new tokens in the follow-up | v0.6.3.10 | v0.6.3.11 |
|---|---|---|
| 13 | 11.0 s | 0.21 s |
| 100 | 3.08 s | 0.93 s |
| 256 | 3.39 s | 2.10 s |
| 257 | 3.39 s | 3.38 s (unchanged) |
| 512 | 3.69 s | 3.68 s (unchanged) |

Limits:
- Follow-ups of 257 new tokens or more are unchanged.
- For follow-ups of 256 tokens or fewer, the first follow-up's output probabilities shift slightly from v0.6.3.10
  (a small numeric difference; in a check against a reference run it was no further from the reference than v0.6.3.10).
- Not tested: a 25-token follow-up with the shipped launcher, more than one user at a time (`--parallel` above 1), and
  the first message to a freshly started server with the shipped launcher.
- `LLAMA_PENDING_PUBLISH_ROWS` (17 to 3072, default 256) moves the 256-token cutoff; `LLAMA_PENDING_PUBLISH=0` turns
  this behavior off.

Dense Swift 1.5 27B is unchanged. Source-only release.
