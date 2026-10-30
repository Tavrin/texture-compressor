# Changelog

## [0.1.0] - 2026-10-30

- Initial release with BC1, BC3, BC4, BC5 and BC7 encoding and Fast,
  Balanced and Quality presets.
- Accept strided R8, RG8 and RGBA8 inputs, replicate partial block edges,
  and return raw compressed blocks or encode into a caller-provided buffer.
- Support native and WebAssembly targets, scalar `no_std` with `alloc`,
  optional SIMD, and optional parallel encoding through Rayon.
- Document the scoped BC4/BC5 Balanced comparisons in the README.
- Require Rust 1.98 or later.
