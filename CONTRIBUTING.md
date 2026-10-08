# Contributing to steam-audio-rust

Thanks for your interest! The project is at a very early stage, so the process is intentionally lightweight.

## Ground rules

- **The original Steam Audio is the specification.** Behavior should match the reference C++ implementation. If you believe the reference is wrong, open an issue before diverging.
- **AI-assisted code is welcome, but it is held to the same bar** as any other code: it must compile, pass tests, and be reviewed by a human before merging. Please mention in the PR if a substantial part was AI-generated.
- **Keep PRs small and focused.** One module or one fix per pull request.

## Development setup

```bash
git clone https://github.com/razedproduction/steam-audio-rust.git
cd steam-audio-rust
cargo build
cargo test
```

Before opening a PR, make sure these pass:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## Tests and parity

For anything that produces numeric output (DSP, HRTF, propagation), add tests that compare results against reference values produced by the original library on identical inputs. State the tolerance you used and why.

## Commit messages

Use short, imperative subjects, optionally prefixed by the area:

```
core: add ambisonics rotation
docs: clarify roadmap
```

## Reporting bugs and proposing features

Use the issue templates. For security problems, see [SECURITY.md](SECURITY.md) instead of opening a public issue.

## License

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
