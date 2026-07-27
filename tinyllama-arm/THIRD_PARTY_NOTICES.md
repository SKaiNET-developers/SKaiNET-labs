# Third-party notices

This project is MIT-licensed (see `LICENSE`). It depends on, references, or downloads
the components below, each under its own license.

| Component | License | How it is used |
|---|---|---|
| [SKaiNET](https://github.com/SKaiNET-developers/SKaiNET) (`sk.ainet.core:*`) | MIT | Dependency, resolved from Maven Central |
| [SKaiNET-transformers](https://github.com/SKaiNET-developers/SKaiNET-transformers) (`sk.ainet.transformers:*`) | MIT | Dependency, resolved from Maven Central |
| [Arm-Examples/Get-Started-with-Edge-AI](https://github.com/Arm-Examples/Get-Started-with-Edge-AI) — `example_2_tinyllama` | MIT, © 2025 Arm Examples | `benchmarks/python/{tinyllama_benchmark,download_models}.py` are derived from it and kept as the upstream baseline. Full notice: `benchmarks/python/LICENSE.arm-examples` |
| [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) (Lightning AI) and the [TheBloke GGUF quantization](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) | **Apache-2.0** | Model weights. **Downloaded at runtime by `scripts/download-models.sh`, not vendored.** Apache-2.0 attribution for the model is mandatory and is given here |
| [Arm KleidiAI](https://github.com/ARM-software/kleidiai) | Apache-2.0 | **Referenced, not vendored.** `benchmarks/kleidiai/bench.c` includes its headers and links its micro-kernels from a separate checkout, to benchmark our kernels against Arm's. No KleidiAI source is redistributed |
| [IREE](https://github.com/iree-org/iree) | Apache-2.0 with LLVM exception | Used as a build/runtime **tool** (`iree-compile`, `iree-run-module`, `iree-convert-parameters`); not linked into this project's code |
| [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) / [llama.cpp](https://github.com/ggerganov/llama.cpp) | MIT | The baseline yardstick, installed by `benchmarks/python/requirements.txt` |
| [psutil](https://github.com/giampaolo/psutil) | BSD-3-Clause | Baseline RSS measurement |
| [huggingface-hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 | Model download |

## GGUF quantization formats

The NEON kernels implement the ggml block formats (Q4_K, Q5_K, Q6_K, Q8_0, Q4_0) and are
bit-exact against their scalar reference. They are independent implementations written for
this project; no ggml/llama.cpp source is vendored.
