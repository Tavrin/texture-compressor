# Changelog

## [0.2.0] - 2026-10-31

### Added

- BC6H encoding, unsigned (`Format::Bc6hUf16`) and signed (`Format::Bc6hSf16`),
  from strided RGB(A) f32 and binary16 slices through `encode_hdr`,
  `encode_hdr_into` and `HdrImageView`, with HDR input normalisation and a
  bit-exact block decoder, `decode_bc6h`.
- HDR perceptual objective, pinned CC0 corpus manifest, independent decoder
  conformance tests and reproducible Intel comparison and wasm identity tools.
- Opt-in `Settings::bc45_fit = Bc45Fit::PeakBounded` for BC4/BC5. Fast and
  Balanced use a direct range fit; Quality retains bounded refinement.
  Refinement bounds each block's channel peak error by its range fit while
  retaining or lowering its squared error. BC5 also checks its worst positive-Z
  normal score against the range fit. Existing presets keep their bytes.
- `Settings::with_alpha` and `Settings::with_bc45_fit`.

### Changed

- `Format`, `Effort`, `AlphaPolicy`, `Bc45Fit`, `Error` and `Settings` are now
  `#[non_exhaustive]`: downstream matches need wildcard arms, and `Settings`
  can no longer be built with a struct literal outside the crate. Build it with
  `Settings::new` plus `with_alpha` and `with_bc45_fit`.
- `Settings` gains the `bc45_fit` field. `Format` gains `Bc6hUf16` and
  `Bc6hSf16`.
- BC6H search cost is reduced with exact decoded-half objective tables, shared
  endpoint estimates and candidate preparation, ranked partitions, cached
  partition distances, eight-block SSE4.1/AVX2 pair-distance evaluation with
  byte-identical scalar and wasm output, and exact integer endpoint
  quantisation.

### Claims

- BC6H log-RMSE quality is at or below Intel ISPC's at Fast, Balanced and
  Quality on the 24-image HDR set (exposure PSNR is within 0.03 dB). No BC6H
  speed claim: Intel's encoder is faster at every preset.
- BC5 `PeakBounded` encodes in 1.66x Intel's wall time on packed RG8 input.
  This is slower than Intel, not faster; the claim is the peak-error bound.

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
