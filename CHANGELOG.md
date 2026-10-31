# Changelog

## [0.2.0] - Unreleased

### Added

- Signed and unsigned BC6H encoding from strided RGB(A) f32 and binary16
  slices, HDR input normalization, and a bit-exact block decoder.
- HDR perceptual objective, pinned CC0 corpus manifest, independent decoder
  conformance tests and reproducible Intel comparison/wasm identity tools.

### Performance

- Share BC6H candidate preparation across nested preset budgets and cache pair
  distances across partition estimates, iterating subset masks directly.
- Batch eight HDR blocks for SSE4.1/AVX2 pair-distance evaluation, preserving
  scalar/wasm bytes; retain scalar endpoint fitting and selector search.
- Replace endpoint binary-search quantisation with an exact integer inverse
  and neighbour comparisons, checked exhaustively against the prior search.

### Changed

- Reduce BC6H search cost with exact decoded-half objective tables, shared
  endpoint estimates, ranked partitions, and early delta admission.
- `Format`, `Effort`, and `Error` are now `#[non_exhaustive]`; downstream
  matches need wildcard arms. `Format` gains `Bc6hUf16` and `Bc6hSf16`.
- Version increases to 0.2.0 for these public enum changes.

## [0.1.1] - 2026-10-30

- `Error` implements `Display`, and `std::error::Error` with the `std`
  feature, so `?` converts it into `Box<dyn std::error::Error>`.

## [0.1.0] - 2026-10-30

- Initial release with BC1, BC3, BC4, BC5 and BC7 encoding and Fast,
  Balanced and Quality presets.
- Accept strided R8, RG8 and RGBA8 inputs, replicate partial block edges,
  and return raw compressed blocks or encode into a caller-provided buffer.
- Support native and WebAssembly targets, scalar `no_std` with `alloc`,
  optional SIMD, and optional parallel encoding through Rayon.
- Document the scoped BC4/BC5 Balanced comparisons in the README.
- Require Rust 1.98 or later.
