# llamAmpere v0.7.2

Five fixes on top of v0.7.1: small MoE models run on the card at full speed, Flash-Next no longer collapses when RAM is
short, thinking replies keep room for the answer, a second user no longer forces a full prompt re-read, and the same
request gives the same reply after earlier requests. Same model files as v0.7.1.

## What changed

- **Small MoE models stay on the card.** In v0.7.1 every MoE model got the Flash-Next expert arena and MoE cache, even
  when the whole model fit on the card. A small MoE model then ran from host memory: Mellum2.1-12B-A2.5B-Thinking
  IQ3_XS wrote 10.9 tokens/s on an RTX 3090. Now, when the experts, the other weights and the KV cache fit under the
  fit margin, the model keeps normal GPU placement with the arena and MoE cache off, and the log says
  "stock GPU placement, expert arena and MoE cache off". Mellum IQ3_XS with default flags: 155.1 tokens/s. Models
  whose experts do not fit (Flash-Next) keep the arena as before. `LLAMA_MOEPLACE=0` restores the v0.7.1 behaviour.
  With `--fit off` the check does not run; `--moe-arena off --moe-cache off` is then the manual setting.
- **Flash-Next with too little RAM to pin the experts no longer collapses.** Flash-Next pins its 53.96 GiB of experts
  only when usable RAM is at least the experts plus 7 GiB (about 61 GiB). Below that, v0.7.1 fell back to reading the
  experts from the model file pages, dropped its card cache settings and, on Linux, failed the card cache copies
  silently: writing fell to 0.63 tokens/s. Now the card cache settings stay on, refused copies go through a staging
  buffer on every platform, and startup prints "EXPERTS NOT PINNED (slower decode)" with how much RAM is needed and
  how much more to free. Same RAM-short state, RTX 3090, Flash-Next AI4, 3,256-token prompt: 89.82 tokens/s
  (v0.7.1: 0.63). With enough RAM to pin, nothing changes.
- **Thinking replies keep room for the answer.** When a reply is still thinking as it nears `max_tokens` or a full
  context, the server now closes the think block early so the answer still gets written. The reserve is a quarter of
  the limit, at least min(512, half the limit): 1,280 tokens for a 5,120-token limit, 32,768 for 131,072. Before, a
  long think could use the whole limit and the reply had no answer (0 of 12 test replies had one; now 12 of 12). A
  reply that ends inside the think block also gets the end tag, so the answer follows. Replies that finish thinking
  before the reserve are unchanged, token for token. Off switch: `--reasoning-answer-reserve 0`,
  `LLAMA_ARG_REASONING_ANSWER_RESERVE=0`, or `"reasoning_answer_reserve": 0` in a request; a number sets a fixed reserve.
  Once the reserve closes a think block, the rest of that reply samples on the CPU.
- **A second user no longer forces a full prompt re-read.** With `-np 2` or more and the MTP draft window (24dense with
  2 to 4 slots), the drafter's shared attention ring could not restore a slot's checkpoint after another slot wrote to
  it, so the slot re-read its whole prompt. 24dense `-np 2`, 28,301-token follow-up on an RTX 3090 Ti: 653 tokens re-read
  (v0.7.1: 28,301), 0.6 s instead of 17.9 s. One-user setups did not have this problem.
- **Same request, same reply, whatever came before.** The adaptive draft vocabulary carried ids from earlier requests;
  it now starts from its load-time ids at each request (when no other slot is busy). Speed is unchanged. At
  temperature 1 a reply that follows other requests on the same server can now differ from v0.7.1's (2 of 6 such
  replies in the speed gate), because v0.7.1's text depended on those earlier requests.
  `LLAMA_SPEC_VOCAB_HOT_CARRY=1` restores the carry. (The first request after a server start can still differ from
  later ones: the prompt-batch ladder starts it at 1024 rows instead of 2048. Not changed in this release.)

## Measured

Speed gate, v0.7.2 against v0.7.1 on the same RTX 3090 24 GB (350 W): chat template, temperature 1, 5,120-token reply,
6 seeds per build in ABBA order, prompt disk cache off.

| model and setup | prompt | v0.7.1 tokens/s (sd) | v0.7.2 tokens/s (sd) | v0.7.2 / v0.7.1 |
|---|---:|---:|---:|---:|
| 27B AI4, `--preset 24dense`, 262,144 context | 98,766 | 112.10 (8.85) | 112.56 (8.58) | 1.004 |
| Flash-Next AI4, `scripts/flash-next-server.sh` | 3,256 | 106.34 (1.74) | 107.49 (1.73) | 1.011 |
| Mellum2.1-12B-A2.5B-Thinking IQ3_XS, default flags | 8,072 | 10.83 (0.06) | 155.12 (1.25) | 14.3 |

- Peak card memory: 21,349 MiB in both builds (27B); Mellum 7,807 MiB (v0.7.1: 7,907).
- In all six v0.7.2 Mellum replies the answer reserve closed the think block at 3,839 tokens, leaving 1,281 tokens for
  the answer. It did not fire in the 27B and Flash-Next replies.
- `--preset 8dense` with the 8 GB file still starts at 204,800 tokens of context, with the same peak card memory as
  v0.7.1.

**Linux build:** `llamampere-v0.7.2-linux-x86_64-cuda12-server.tar`, same as v0.7.1: Ubuntu 22.04 or newer (glibc
2.35+), NVIDIA driver for CUDA 12.4 or newer (R550+), no CUDA toolkit needed, built for RTX 30-series cards (sm_86).
It brings the CUDA 12.4 runtime and cuBLAS (NVIDIA redistributables, CUDA-EULA.txt inside) and its own GCC runtime
libraries (licence and source notes inside; the GCC source is attached to the release).

**Ada and Blackwell build:** `llamampere-v0.7.2-linux-x86_64-cuda12.8-sm86-89-120-server-fat.tar` is built for RTX 30
(sm_86), RTX 40 (sm_89) and RTX 50 (sm_120) cards with CUDA 12.8 and needs an NVIDIA driver 570 or newer. Checked on an RTX 3090 (sm_86) and an RTX 5080 (sm_120): it loads and answers.

**Windows build:** `llamampere-v0.7.2-windows-x86_64-cuda12-server.zip`, built by CI like v0.7.1 (CUDA 12.4, sm_86,
driver R550+); not run on a GPU by us.
