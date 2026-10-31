# Changelog

## Unreleased

- Add opt-in `Settings::bc45_fit = Bc45Fit::PeakBounded` for BC4/BC5.
  Fast and Balanced use a direct range fit; Quality retains bounded refinement.
  Refinement bounds each block's channel peak error by its range fit while
  retaining or lowering its squared error. BC5 also checks its worst positive-Z
  normal score against the range fit. Existing presets keep their bytes.
  Struct literals for `Settings` need the new field.

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
