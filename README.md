<div align="center">

# 🔊 steam-audio-rust

**An AI-driven Rust reimplementation of [Steam Audio](https://github.com/ValveSoftware/steam-audio)**

Spatial audio, physics-based sound propagation and HRTF rendering — memory-safe, portable, and written in Rust.

[![Status](https://img.shields.io/badge/status-planning-lightgrey)](#status)
[![Language](https://img.shields.io/badge/language-Rust-dea584?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Fork of](https://img.shields.io/badge/fork%20of-ValveSoftware%2Fsteam--audio-1b2838?logo=steam)](https://github.com/ValveSoftware/steam-audio)

[Overview](#overview) · [Status](#status) · [Roadmap](#roadmap) · [Repository layout](#repository-layout) · [Getting started](#getting-started) · [Contributing](#contributing) · [Credits](#credits-and-license)

</div>

---

## Overview

[Steam Audio](https://valvesoftware.github.io/steam-audio/) is Valve's spatial audio toolkit: it lets game and VR developers render sound that responds to the 3D world around the listener.

This project is an experiment in reimplementing that functionality in **Rust**, with AI-assisted development doing much of the translation and porting work. The reference behavior is the original C++ implementation, so the goal is functional parity and verifiable results, not a loose reinterpretation.

**Why Rust?**

- **Memory safety** for a real-time audio library that runs inside other people's processes.
- **Fearless concurrency** for the multi-threaded simulation and mixing workloads that spatial audio needs.
- **Easy distribution** through Cargo, with straightforward cross-compilation.

## Status

> 🚧 **Not started yet.** The repository currently holds the fork and initial scaffolding. Nothing here is usable in production.

This README will be updated as modules land. Treat the roadmap below as intent, not a promise.

## Roadmap

Planned scope, mirroring the feature areas of the original library. Boxes are checked only when a module passes comparison tests against the reference implementation.

- [ ] **Foundations**: math types, ambisonics primitives, FFT and audio buffers
- [ ] **Binaural rendering**: HRTF loading and interpolation, panning
- [ ] **Direct sound**: distance attenuation, air absorption, directivity, occlusion and transmission
- [ ] **Scene and geometry**: static and instanced meshes, ray tracing backend
- [ ] **Reflections**: real-time and baked sound propagation
- [ ] **Pathing**: diffraction and occlusion-aware paths between source and listener
- [ ] **C API compatibility layer**: drop-in replacement for the original `phonon` interface

## Repository layout

| Path | Purpose |
| --- | --- |
| [`core/`](core) | Core library: the main body of the reimplementation |
| `.gitignore` | Rust and build artifact ignores |
| `README.md` | You are here |

> The layout will expand as the project grows (for example `docs/`, `examples/`, `tests/`, `benches/`).

## Getting started

There is nothing to install yet. To follow along or build the current state:

```bash
git clone https://github.com/razedproduction/steam-audio-rust.git
cd steam-audio-rust
cargo build        # once a Cargo workspace is in place
cargo test
```

Requires a recent stable [Rust toolchain](https://rustup.rs/).

## How the AI-driven approach works

- The original C++ code is the **specification**. Behavior is matched against it, not guessed.
- AI-generated code is treated like any other contribution: it must compile, pass tests, and be reviewed.
- Output of the Rust port is compared with the reference implementation on identical inputs wherever possible.

## Contributing

The project is at a very early stage, so the most useful contributions right now are:

1. Opening issues with design questions or scope suggestions.
2. Proposing a test strategy for numerical parity with the reference library.
3. Reviewing the module structure in [`core/`](core).

## Credits and license

- **Steam Audio** is developed by [Valve Corporation](https://github.com/ValveSoftware/steam-audio) and released under the Apache License 2.0. Official documentation: <https://valvesoftware.github.io/steam-audio/>.
- This reimplementation is maintained by **razed.production**.
- This project is not affiliated with or endorsed by Valve Corporation.

Licensed under the [Apache License 2.0](LICENSE), consistent with the upstream project.
