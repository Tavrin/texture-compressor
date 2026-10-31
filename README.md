# texture-compressor

Encode decoded pixels into GPU block-compressed BC1, BC3, BC4, BC5, BC6H and BC7
textures with Fast, Balanced and Quality search presets. Encoding is
deterministic for a given input and settings. The crate supports native and
WebAssembly targets. It returns raw blocks; it does not decode image files,
generate mipmaps, or write texture containers.

## Install

Requires Rust 1.98 or later (MSRV).

```toml
[dependencies]
texture-compressor = "0.2"
```

The default features are `std` and `simd`. Unsafe code is confined to the
private, audited BC7 SIMD module; its safety arguments are in
[`src/bc7_simd/SAFETY.md`](src/bc7_simd/SAFETY.md). Disabling default features
builds an unsafe-free, scalar `no_std` library that requires `alloc`.
The optional `parallel` feature provides encoding through the current Rayon
pool. The `simd-diagnostics` feature enables a process-local backend override
for testing.

With `std`, x86 SIMD dispatch checks SSE4.1/AVX2 at runtime. Without `std`,
SIMD dispatch uses compile-time target features. AArch64 NEON and WebAssembly
simd128 also require compile-time target features; unsupported targets use
the scalar implementation.

## Usage

Input strides are in bytes. Dimensions need not be multiples of four: encoding
replicates edge texels to fill partial blocks. Output blocks are in row order.

### RGBA8 to BC7

```rust
use texture_compressor::{encode, Effort, Format, ImageView, Settings};

let rgba = [255u8; 4 * 4 * 4];
let image = ImageView::rgba8(&rgba, 4, 4, 16)?;
let blocks = encode(image, &Settings::new(Format::Bc7, Effort::Balanced))?;
assert_eq!(blocks.len(), 16);
# Ok::<(), texture_compressor::Error>(())
```

### Single channel to BC4

```rust
use texture_compressor::{encode, Effort, Format, ImageView, Settings};

let roughness = [128u8; 4 * 4];
let image = ImageView::r8(&roughness, 4, 4, 4)?;
let blocks = encode(image, &Settings::new(Format::Bc4, Effort::Balanced))?;
assert_eq!(blocks.len(), 8);
# Ok::<(), texture_compressor::Error>(())
```

### Normal map to BC5

Store the normal's X and Y components as unsigned R and G bytes. Reconstruct Z
when sampling the texture; the encoder performs no normal-vector conversion.

```rust
use texture_compressor::{encode, Effort, Format, ImageView, Settings};

let normal_xy = [128u8; 4 * 4 * 2];
let image = ImageView::rg8(&normal_xy, 4, 4, 8)?;
let blocks = encode(image, &Settings::new(Format::Bc5, Effort::Balanced))?;
assert_eq!(blocks.len(), 16);
# Ok::<(), texture_compressor::Error>(())
```

## Claims

On a sealed test set of 30 CC0 inputs from 13 assets, using single-thread
instruction cost and visible-texel quality metrics:

- BC4 Balanced matches rusty_dds quality (pooled SSIM margin +7.6e-7) at
  0.86x its single-thread instruction cost.
- BC5 Balanced uses 0.80x the single-thread instruction cost of rusty_dds,
  with +0.13 dB pooled PSNR.

BC1, BC3 and BC7, at all presets, are provided without performance claims.
BC4 and BC5 Fast and Quality are provided without performance claims.

## How claims are measured

The test set was sealed before evaluation; format-specific subsets overlap
and are drawn from those 30 inputs, rather than 30 independent assets.
BC4 used nine inputs and BC5 used fourteen. Retired user-mode instructions
were counted only inside single-thread encode windows. PSNR pools squared
error and sample counts before taking logarithms; SSIM weights clipped 8x8
tiles by visible texel count. Both metrics use source-visible texels (all
texels for BC4/BC5). Pooled quality had to meet the reference, with each
texture and required channel losing at most 0.1 dB PSNR and 0.001 SSIM;
BC5 also had to pass the required R/G channel pools. An independent read-only
check reproduced the results from raw receipts. These results apply to this
test set and these metrics; they establish neither wall-time speed nor
multi-thread performance or quality on every unseen texture.

## Licence

Licensed under either the [MIT licence](LICENSE-MIT) or the
[Apache License, Version 2.0](LICENSE-APACHE), at your option.

## HDR RGB to BC6H

```rust
use texture_compressor::{encode_hdr, decode_bc6h, Effort, Format, HdrImageView, Settings};
let rgb = [2.0f32; 4 * 4 * 3];
let image = HdrImageView::rgb_f32(&rgb, 4, 4, 12)?;
let blocks = encode_hdr(image, &Settings::new(Format::Bc6hUf16, Effort::Balanced))?;
let block: &[u8; 16] = blocks[..16].try_into().expect("one block");
let half_rgb = decode_bc6h(block, Format::Bc6hUf16)?;
assert_eq!(half_rgb.len(), 16);
# Ok::<(), texture_compressor::Error>(())
```

`Bc6hUf16` stores unsigned HDR RGB; `Bc6hSf16` stores signed HDR RGB.
`HdrImageView::{rgb_f32,rgba_f32}` borrow float slices;
`{rgb_f16,rgba_f16}` borrow `u16` IEEE binary16 bit patterns.
HDR strides count samples, not bytes. Alpha is ignored. `encode_hdr_into`
writes into caller storage. Partial blocks replicate edge texels.
The byte-input `encode` API rejects BC6H, and HDR encoding rejects other formats.
NaN becomes zero; infinities and finite overflow clamp to ±65504. Unsigned
encoding additionally clamps negative values to zero. Subnormal values are
retained as input (compression can quantize them to zero); negative zero
normalizes to zero. The decoder returns exact binary16 RGB bits for every
mode and partition, including the signed format's possible negative infinity.
Reserved modes decode to black.

The encoder minimizes `Σ(T(source) - T(decoded))²` over RGB and texels,
where `T(x) = sign(x) * log2(1 + 16 * abs(x))`. The software `libm` implementation
and f64 accumulation keep this objective consistent across native and wasm.
All fourteen modes remain candidates. Partitions are ranked by a cheap
perceptual line-residual estimate, with endpoint pairs shared across modes.
Fast/Balanced/Quality evaluate the best 1/2/4 partitions plus all
single-subset modes, retain the preceding preset's result, then apply
2/3/4 damped Gauss-Newton fitting iterations and 0/0/0 bounded
coordinate refinement rounds. All 32 partitions remain eligible for selection.
Decoded binary16 objective values use an exact software-libm lookup table
shared by native and wasm; source objective values still use software libm.
Transformed endpoint admission precedes bitstream packing. The factor 16
gives shadow detail weight around radiance 1/16 instead of 1. This is a scalar
search, with no per-block heap allocation.

The normative source is [Khronos Data Format Specification 1.4, §20.2 BC6H,
including the endpoint and block-layout tables](https://registry.khronos.org/DataFormat/specs/1.4/dataformat.1.4.html).
The endpoint perturbation strategy was informed by reading
[Microsoft DirectXTex](https://github.com/microsoft/DirectXTex/blob/main/DirectXTex/BC6HBC7.cpp);
its implementation was not copied. Decoder conformance is checked against
`bcdec_rs`, a dev-dependency only. Its signed-zero discrepancy is handled
explicitly: Khronos preserves the pre-scaling negative sign even when the
scaled magnitude becomes zero; a separate normative vector checks that bit. Intel's `intel_tex_2` is also dev-only.

The pinned [corpus manifest](tests/bc6h_manifest.json) records 24 public HDRIs,
source URLs, CC0 licences, source/crop SHA-256 hashes and center crop rectangles.
Selection uses the first 24 alphabetically sorted public HDRI asset IDs;
all crops are 64×64 with original radiance preserved. Recreate with
`python3 tools/bc6h_corpus.py <external-directory>` (requires NumPy and OpenCV).
Run `cargo run --release --example bc6h_measure -- <external-directory>` for
paired timings (alternating order, three rounds, median; same first-image warmup)
and quality against Intel fast/basic/slow settings for Fast/Balanced/Quality.
Both quality metrics pool RGB texels across exposures `[-4,-2,0,2,4]` EV.
Log RMSE compares `log2(1 + 2^EV*x)`. Exposure PSNR uses
Reinhard display values `v = 2^EV*x / (1 + 2^EV*x)`, peak 1, and pooled MSE.
Timing covers the complete corpus including allocations, excluding file I/O
and decoding. The Intel comparison is unsigned only; Intel's wrapper does not
expose signed encoding. These small fixed crops do not establish full-image
performance or quality on other content.

For native/wasm identity, build `bc6h_identity` for native and `wasm32-wasip1`,
then compare its binary stdout on the same external corpus directory.
`node tools/bc6h_wasi.cjs <binary.wasm> <external-directory>` runs the WASI
version. The stream includes signed/unsigned corpus encoding and decoding,
all presets, normalization cases, binary16 inputs, and arbitrary bitstreams
covering every mode and partition. `examples/identity.rs` separately covers
existing-format byte identity.

### BC6H comparison

Measured on 24 fixed 64×64 CC0 HDR crops with the methodology above.
Each quality value pools all five exposures; times are median complete-corpus
wall times from three alternating paired rounds on a shared AMD Ryzen 9
7945HX host (x86_64 Linux, Rust 1.98.1, release profile, intel_tex_2 0.5.0).
These are unsigned results; the independent decoder check covers both signs.

| Preset / Intel preset | Log RMSE ours / Intel | Exposure PSNR ours / Intel (dB) | Encode ms ours / Intel |
| --- | ---: | ---: | ---: |
| Fast / fast | 0.018021 / 0.018640 | 46.96 / 46.84 | 238.38 / 9.05 |
| Balanced / basic | 0.017718 / 0.017845 | 47.09 / 47.10 | 505.20 / 19.80 |
| Quality / slow | 0.017493 / 0.017640 | 47.18 / 47.20 | 1157.54 / 128.53 |

The Intel log-RMSE quality target passes at all three presets.
The encode-time target of at most twice Intel is not achieved at every preset.
These paired timings were collected on a shared host and retain scheduling noise. Full-size images, other HDR content, signed
quality against Intel, and hardware GPU decoding are not qualified by this run.
