# ADR 0002: Reading and writing FlatBuffers

- **Status:** accepted
- **Date:** 2026-10-16

## Context

Upstream serializes scenes, static meshes and baked data (probe batches, energy fields, reverbs, pathing data) as
[FlatBuffers](https://flatbuffers.dev/), with the schemas in `core/src/core/*.fbs`, compiled by `flatc` 1.12 and
written through `flatbuffers::FlatBufferBuilder`. The plan (§2) named the `flatbuffers` crate with code generated
from the same schemas. Things that bound the decision:

- Risk R4: files must be readable by both implementations. Writing the same bytes as upstream makes that trivial to
  check (goldens compared `EXACT`) and lets existing baked files and tools keep working.
- The workspace denies `unsafe_code`. `flatc`'s Rust output (and the `flatbuffers` crate's table accessors since 2.0)
  uses `unsafe` for every field access, so generated code cannot live in the `steam-audio` crate.
- Serialized data comes from users (`iplSceneLoad`, `iplProbeBatchLoad`), so the reader handles untrusted input.
  Upstream never runs the FlatBuffers verifier: malformed data is undefined behaviour.
- The schemas are small: 13 files, all fields 4 bytes wide (offsets, `int32`, `float`) except a few `byte` enums and
  a `uint8` vector, no strings, unions or nested flatbuffers.

## Options

### A. `flatbuffers` crate with `flatc`-generated code

- ➕ The standard approach; the generated code follows schema changes automatically.
- ➖ Needs `unsafe` in our crate (or a separate crate exempt from the lint just for generated code).
- ➖ The Rust builder is a separate implementation of a newer FlatBuffers version, with no guarantee of the C++ 1.12
  builder's exact layout (padding, vtable sharing), so parity might need a structural comparison instead of bytes.
- ➖ Verification is opt-in and checks structure only; the objects' own invariants (index ranges) need checking anyway.

### B. Own small builder and reader (chosen)

- ➕ The builder ports `FlatBufferBuilder` 1.12 operation for operation (back-to-front layout, `PreAlign` with
  upstream's power-of-two mask even for 12-byte structs, defaults left out, vtable deduplication), so output is
  byte-identical and checked `EXACT`.
- ➕ The reader is safe by construction: every offset, vtable and vector is bounds-checked as it is followed, and
  malformed data is `Error::MalformedSerializedData`.
- ➖ Each table's (de)serializer is written by hand (vtable slot constants and the field order of `flatc`'s `Create*`
  functions: largest first, reverse declaration order among equal sizes). With ~10 tables this is a few lines each.

## Decision

Option B: `crate::serialization` (crate-private) holds `Builder`, `Table` and `Vector`; each type serializes itself
next to its definition (`EnergyField`, `Mesh`, `StaticMesh`, `Scene`; probe batches and baked data in 6.4b). Public
APIs exchange plain bytes (`serialize() -> Vec<u8>`, `deserialize(&[u8]) -> Result<Self>`) in place of upstream's
`SerializedObject`.

## Consequences

- Goldens are whole buffers compared byte for byte; deserializing them checks cross-loading (R4).
- Malformed data never panics; the tests truncate and corrupt buffers to check it.
- A schema change upstream needs a matching hand edit; the goldens catch any mismatch in layout.
