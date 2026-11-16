# Credits

Mainline: [llama.cpp v1.0](https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0), by the ggml authors and contributors, under its MIT license.

TurboQuant: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant), including its TurboQuant KV-cache work. Our existing base includes that fork through [643956891](https://github.com/TheTom/llama-cpp-turboquant/commit/643956891).

The CUDA follow-ups included here are [PR #412](https://github.com/TheTom/llama-cpp-turboquant/pull/412) (source-less graph node check, 0218b23bd) and [PR #414](https://github.com/TheTom/llama-cpp-turboquant/pull/414) (all batched columns in fused/cuBLAS paths, 95ef72f20), contributed by jasstrong. Credit follows each original commit's author. Later fork commits and open PRs are not claimed as included.

Original authors, license headers, and third-party notices are retained. See [LICENSE](LICENSE) and the [third-party library notices](README.md#license-and-credits).
