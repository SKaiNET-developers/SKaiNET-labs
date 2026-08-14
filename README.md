# SKaiNET-labs

SKaiNET projects incubator — experiments and applied projects built on
[SKaiNET](https://github.com/SKaiNET-developers/SKaiNET), graduated here once they carry
reproducible results.

## Projects

### [`tinyllama-arm/`](tinyllama-arm/) — TinyLlama on a 2 GB Cortex-A55, in pure Kotlin

TinyLlama-1.1B (GGUF Q4_K_M) on a Synaptics Astra SL2619 (2× Cortex-A55, 1.92 GB RAM, no swap):
Kotlin/Native board binary, hand-written Neon kernels benchmarked head-to-head against Arm's
KleidiAI, and an independent StableHLO→IREE compiled path that cross-checks it. Every number is
tagged, logged, and reproducible on Arm64 hardware you already own — start with the
[reproduction ladder](tinyllama-arm/README.md#try-it--the-reproduction-ladder).

## License

MIT — see [LICENSE](LICENSE). Individual projects carry their own third-party notices.

---

*Arm and Neon are registered trademarks or trademarks of Arm Limited (or its subsidiaries or
affiliates) in the US and/or elsewhere.*
