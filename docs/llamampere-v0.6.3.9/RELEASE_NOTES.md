# llamAmpere v0.6.3.9

Flash-Next on native Windows builds (MSVC + CUDA) can now use the fast path that keeps expert tables on the graphics card. Before this release that path was Linux-only, so on Windows every expert the card did not hold went through a slow fallback, and writing ran at about 0.26 tokens/s.

Thanks to huhichao (#98), who wrote the Windows port, found the causes and measured it on his own Windows machine.

## What changed (Windows only)

- The card-side expert tables are turned on for Windows builds. Windows free RAM is read with `GlobalMemoryStatusEx`; Windows has no cgroups, so the free-RAM floor is the only memory guard there.
- Windows rejects a copy to the card from a memory range that mixes locked and normal pages, where Linux accepts it. That copy now goes through a small staging buffer on the same GPU stream, and it finishes before the expert slot is used. If locked memory for the buffer cannot be allocated, normal memory is used instead (slower, same order).
- The CPU expert split pins its threads with the Windows APIs. On Windows, core 0 is left out of the expert thread pool (when there are more than 2 cores), and the thread that drives the card is kept off the pool's cores; `GGML_MOE_CPU_SPLIT_WIN_CARD=0` stops restricting the card thread (core 0 stays out of the pool either way).
- Windows MoE runs now see the machine's free RAM (it used to read as 0), so the RAM-fit and read-ahead budgets work on Windows as on Linux. This affects every Windows MoE run, not only the card-side tables.
- MSVC build fixes from the port: `/bigobj` for `ai4.cu`, exported CPU helpers, `_tzcnt_u32` instead of `__builtin_ctz`, and half-float conversion helpers for the AVX2 expert kernels.

## Two new optional settings (all platforms, off unless set)

`GGML_MOE_DEVICE_RAM_FLOOR_MIB` and `GGML_MOE_DEVICE_RAM_MARGIN_MIB` (in MiB) replace the 7 GiB floor and 2 GiB margin of free RAM that is kept when experts are locked in memory for the card. Windows counts less RAM as free than Linux does on the same machine (model pages that are already loaded are not counted), so on a 64 GB machine the default floor can refuse. The contributor's runs set both to 0. A value must be a plain whole number up to 1048576 (1 TiB); anything else is ignored with a log line and the default is kept.

## Speed (contributor's numbers)

RTX 3090 24 GB, 64 GB RAM, Intel i5-12490F, Windows 11, Flash-Next, `-c 110592`, started with `scripts/flash-next-server.sh` through Git Bash, with `GGML_MOE_DEVICE_RAM_FLOOR_MIB=0 GGML_MOE_DEVICE_RAM_MARGIN_MIB=0 GGML_MOE_CPU_SPLIT_GBPS=12`:

| request | before | after |
|---|---|---|
| code, 200 tokens | 0.26 tokens/s | 60.2 tokens/s |
| Chinese prose, 300 tokens | not measured | 24.4 tokens/s |

## How this was tested

We have no Windows machine. Built and tested under MinGW/Wine; Windows speed figures are from the contributor. The MinGW/Wine build is CPU-only, so it cannot compile the CUDA and MSVC parts of this change. The contributor built and measured his version on his own tree (based on v0.6.3.3). Since then the CPU expert split was reorganized and we changed the copy fallback, buffer cleanup and setting checks, so nobody has yet built this exact code with MSVC + CUDA. Please report on #98 if your Windows build fails or the speed differs.

Linux is unchanged: the new code is Windows-only, apart from the two optional settings above, which do nothing unless set. Checks on Linux: CPU suites, the README-flags check, example builds, the MinGW/Wine Windows build and tensor-split checks all passed, dense 27B greedy output is token-for-token identical to v0.6.3.8, and Flash-Next AI3 starts and writes with the shipped launcher.

Earlier Windows fixes: the MSVC build errors (#93) and the Windows model-loading error (#94) were fixed in v0.6.3.3.

Source-only release.
