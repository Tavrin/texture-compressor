# Changelog

## [Unreleased]

### Changed

- BC7 Balanced and Quality encode blocks with transparent texels faster:
  the weighted search shares its per-texel sums across colour channels and
  ranks partitions with SSE4.1, AVX2, NEON and wasm simd128 kernels. Output
  bytes are unchanged, and every backend produces the scalar bytes.
- BC7 least-squares endpoint fits accumulate in 32-bit integers (exact for
  16 texels), with the same output.

## [0.2.2] - 2026-11-03

### Fixed

- BC7 Balanced and Quality wrote black colour (RGB 0) wherever source alpha
  was 0, because their search scored colour only on texels with nonzero
  alpha. They now treat alpha as straight and score colour on every texel,
  so colour under zero alpha survives filtering, mipmapping and later
  premultiplication.

### Changed

- BC7 Balanced and Quality end each block with a peak-aware choice: when the
  lowest-error candidate has a channel error of 6 or more, a refined
  candidate with a lower largest channel error and at most 1/8 more squared
  error is taken instead. This lowers the texture-wide maximum byte error at
  a small cost in PSNR. Balanced and Quality output bytes change; Fast and
  the `IgnoreTransparentColour` policy keep their 0.2.1 output.
- The README no longer carries the BC7 Quality performance claim: it was
  measured on texels with nonzero alpha only and does not describe the new
  default. BC7 has no performance claim until a new sealed set is evaluated.

### Added

- `AlphaPolicy::IgnoreTransparentColour` (BC7 only) reproduces the 0.2.1
  Balanced and Quality output byte for byte, as an explicit opt-in for
  premultiplied input or cutouts whose transparent texels are never filtered.
  Other formats reject it with `Error::UnsupportedPolicy`.
### Changed

- BC5 `PeakBounded` Fast and Balanced on x86_64 (SSE4.1) now encodes each block
  row in one vector kernel that deinterleaves R and G with byte shuffles
  straight into registers. The previous per-block gather wrote 16 single bytes
  and reloaded them as one vector, which stalled on store forwarding, and the
  per-block call did not inline in builds with several codegen units. Output
  bytes are unchanged. On 28 consumer textures (single pinned core, RG8 packed
  inside the timed span for both encoders) the pooled time against Intel ISPC
  went from 2.0x to 0.78x with codegen-units = 16, and from 1.6x to 0.79x with
  codegen-units = 1.

## [0.2.1] - 2026-11-03

### Changed

- BC7 Balanced and Quality now spend a fixed per-block work budget instead of
  adapting effort to block difficulty, so encode time no longer depends on how
  hard the content is. Candidates are chosen by cheap estimates, partition
  ranking and the palette selector run across several representatives per
  vector (SSE4.1, AVX2, NEON and wasm simd128 paths), and the mode 2 search is
  removed; no preset emits BC7 mode 2 (the decoder still reads it). Balanced and
  Quality output bytes change. Fast output is unchanged on the identity sample.
  Output stays identical across SIMD levels, thread counts and native/wasm.
- No BC7 performance claim is made for this change.

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
