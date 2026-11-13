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
private, audited SIMD module; its safety arguments are in
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

BC7 treats alpha as straight (unassociated): colour error counts on every
texel, including texels whose alpha is zero, because filtering, mipmapping or
later premultiplication can still read that colour. For premultiplied input,
or cutouts whose transparent texels are never filtered,
`with_alpha(AlphaPolicy::IgnoreTransparentColour)` stops Balanced and Quality
from scoring that colour (it may then decode black) and spends the precision
on visible texels instead.

For UI cutouts and coverage masks, `with_alpha(AlphaPolicy::PreserveZeroAlpha)`
keeps source alpha zero exact while retaining soft alpha and straight colour.
Only blocks containing zero alpha use this constraint. They try a constrained
mode 6 fit (sixteen alpha levels) and modes 4/5 with separate colour/alpha
selectors. Other blocks use the ordinary fit; nonzero alpha remains lossy.
Opaque blocks and blocks without either alpha extreme retain the normal search.

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

### Bound BC4/BC5 peak byte error

Call `Settings::with_bc45_fit(Bc45Fit::PeakBounded)` to opt in:

```rust
use texture_compressor::{encode, Bc45Fit, Effort, Format, ImageView, Settings};

let xy = [128u8; 4 * 4 * 2];
let settings = Settings::new(Format::Bc5, Effort::Balanced).with_bc45_fit(Bc45Fit::PeakBounded);
let blocks = encode(ImageView::rg8(&xy, 4, 4, 8)?, &settings)?;
# Ok::<(), texture_compressor::Error>(())
```

For each channel and block, this starts with maximum/minimum endpoints and
nearest palette selectors. Fast and Balanced keep these endpoints and need no
search or normal-angle guard. Quality searches for lower squared error without
increasing that fit's maximum absolute byte error, using the preset's fit and a
fixed endpoint neighborhood. The bound refers to the
unsigned integer BC4 reconstruction, including replicated edge texels.
It is relative to this range fit, rather than a universal bound against other
encoders. BC5 additionally rejects a refinement if its worst reconstructed
positive-Z normal angle increases relative to that range fit, using an integer
score with twelve fractional coordinate bits. XY outside the unit disk uses
Z = 0 and normalized vectors. This does not bound mean angular error or promise
angular quality against every other encoder. Quality trades extra encoding work
for lower squared error within the byte-error and normal-angle bounds. BC4 and BC5 support it at every
preset; BC1, BC3 and BC7 ignore it.

`Settings::new` keeps `Bc45Fit::SquaredError`, preserving existing preset output.
`Settings` is `#[non_exhaustive]`: build it with `Settings::new` plus
`with_alpha` and `with_bc45_fit`. `Format`, `Effort`, `AlphaPolicy`, `Bc45Fit`
and `Error` are also `#[non_exhaustive]`, so matches on them need a wildcard arm.

## Claims

On a sealed test set of 30 CC0 inputs from 13 assets, using single-thread
instruction cost and visible-texel quality metrics:

- BC4 Balanced matches rusty_dds quality (pooled SSIM margin +7.6e-7) at
  0.86x its single-thread instruction cost.
- BC5 Balanced uses 0.80x the single-thread instruction cost of rusty_dds,
  with +0.13 dB pooled PSNR.

On other measured sets:

- BC5 with `Bc45Fit::PeakBounded` encodes packed RG8 input in 1.66x Intel's
  wall time (2.10x from RGBA8 input). It is slower than Intel; it adds the
  per-block peak-error bound described above.
- BC6H error (log-space RMSE) is equal to or lower than Intel ISPC's at Fast,
  Balanced and Quality on the 24-image HDR set below; exposure PSNR is within
  0.03 dB.
  There is no BC6H speed claim: Intel is faster at every preset.

BC7 has no performance claim. The earlier BC7 Quality result scored colour
only on texels with nonzero alpha; BC7 now scores colour on every texel by
default (straight alpha), so that result no longer describes the default
encoder. A new claim needs a new sealed test set. BC1 and BC3 are provided
without performance claims, and so are BC4 and BC5 Fast and Quality.

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

### Fuzzing

`fuzz/` holds four libFuzzer targets (run with `cargo +nightly fuzz run
<target>` from the repository root; seed corpora are in `fuzz/corpus/`):

- `bc6h_encode`: arbitrary f32 and binary16 bit patterns (NaN, infinities,
  subnormals, negative zero), RGB and RGBA, both signs, all presets, strided and
  non-multiple-of-four sizes, through `encode_hdr` and `encode_hdr_into`, then
  `decode_bc6h`. Checks the output length, equality of the two entry points, an
  untouched output tail, and finite unsigned decodes.
- `bc6h_decode`: arbitrary 16-byte blocks through `decode_bc6h`, both signs.
- `bc5_peak`: BC4 and BC5 `PeakBounded` at every preset on strided R8, RG8 and
  RGBA8 input. Checks determinism and that no block channel's peak byte error
  exceeds the range fit's, with a separate round-to-nearest BC4 decoder.
- `bc7_encode`: BC7 Fast, Balanced and Quality on RGBA8 images up to 23x23
  texels (strided, non-multiple-of-four) with opaque, cutout, transparent,
  near-opaque, near-transparent and noise alpha, and raw, few-colour and ramp
  colour. Checks that every block has a legal mode and decodes through the
  independent reference decoder in `tests/bc7_decoder.rs`, that `encode` equals
  `encode_into` and repeats, and that the first full block encodes identically
  alone and inside the image. With `BC7_DIGEST_LOG=<file>` it logs input and
  output digests, so a corpus replay under `TEXENC_SIMD=scalar` (build with
  `simd-diagnostics`, which the fuzz crate enables) can be compared with the
  dispatched backend byte for byte. Inputs whose first byte has the high bit
  set use a layout built for mutation: low bits choose Balanced or Quality, the
  rest is RGBA8 texels, 64 bytes per 4x4 block, up to six blocks in a row.

Method: cargo-fuzz 0.13.2, rustc 1.101.0-nightly (2026-10-04), AddressSanitizer,
debug assertions and overflow checks on, x86_64 with runtime-dispatched
SSE4.1/AVX2. Reach was measured first by replaying the seeds with LLVM
coverage: `bc6h_encode` seeds reached 93% of `bc6h.rs` lines and 82% of
`hdr.rs`. Each target then ran in 600 s segments on one core until a segment
added no coverage, then once more with `-rss_limit_mb=512`.

| Target | Segments (s) | Executions | Edges | Final reach | Crashes |
| --- | ---: | ---: | ---: | --- | ---: |
| `bc6h_encode` | 3 x 601 | 214,487 | 878 | `bc6h.rs` 93%, `hdr.rs` 88% of lines | 0 |
| `bc6h_decode` | 3 x 601 | 105,134,469 | 235 | decoder paths (`bc6h.rs` 22%, `hdr.rs` 4% of lines) | 0 |
| `bc5_peak` | 3 x 601 | 787,474 | 908 | `bc4.rs` 82% of lines | 0 |
| `bc7_encode` | 4 + 5 x 600 | 852,839 | 4,948 | `bc7_search.rs` 89%, `bc7_objective.rs` 79%, `bc7_fast.rs` 69%, `bc7_simd/avx2.rs` 87% of lines (seed replay) | 0 |

`bc7_encode` reach was measured first by replaying 120 generated seeds with LLVM
coverage: `bc7_search.rs` 25% of lines, `bc7_fast.rs` 73%, `bc7_objective.rs`
81%. Four 600 s segments from those seeds reached 4,930 edges (the last two
flat). `bc7_search.rs` holds the Balanced and Quality search, so the corpus was
then seeded with 1,500 inputs in the raw layout (a byte with the high bit set
whose low bits select Balanced or Quality, then 64 bytes of RGBA8 per 4x4
block): 150 real blocks per cohort, each at both presets, cut from development
textures (the training set and the second held-out set only). The cohorts are
detailed opaque, cutout, saturated, smooth gradient and alpha-blended (a ramp
alpha put on detailed blocks). Replaying only those seeds took `bc7_search.rs`
from 25% to 89% of lines, `budget.rs` to 93% and `avx2.rs` to 87%. Five more
600 s segments from the combined corpus reached 4,948 edges: 4,947 after the
first, 4,948 after the third, none added in the fourth and fifth. All segments
used `-rss_limit_mb=512` and exited 0. The final corpus (1,606 inputs) was
replayed with `-runs=0` under `TEXENC_SIMD=scalar`, `sse41`, `avx2` and the
default dispatch on an AVX2 machine: 1,605 inputs each, exit 0, and the sorted
per-input output digests of the four runs are identical. No panic, overflow,
illegal block or mismatch was found.

Every run exited 0 (including the 512 MB runs). The first `bc5_peak` run stopped
on its own check: the target's reference decoder truncated BC4 interpolants,
while the format rounds them. The fault was in the target, not the library; it
was corrected and `bc5_peak` was rerun from its seeds, which is the row above.
This establishes that about ten minutes of coverage-guided fuzzing per target
found no panic, overflow, out-of-bounds access or violated check on x86_64. It
does not cover NEON or WebAssembly code paths, other CPUs' dispatch choices,
or images larger than 23x23 texels.

#### 0.2.2 additions (BC5 row kernel, BC7 straight alpha)

`bc5_peak` now takes images up to 67 texels wide (rows of 1 to 17 blocks, so
every remainder of the SSE4.1 `PeakBounded` row kernel) and, with
`BC5_DIGEST_LOG=<file>`, logs input and output digests like `bc7_encode`.
`bc7_encode` now also picks `AlphaPolicy::IgnoreTransparentColour` (raw layout
bit 2; structured mode byte divided by three), can zero the alpha of every odd
block (raw layout bit 3), and under the default policy asserts that a fully
transparent block with at least four texels of visible colour (a channel of 16
or more) does not decode to black. A first version of that assertion covered
any zero-alpha texel with any colour and fired on legitimate output (a texel of
(1, 0, 0), then one of (46, 16, 0) in a block of mostly visible texels, where
the objective may trade that texel); it was narrowed to the case above.

Seeds: about 300 BC7 and 250 BC5 inputs cut from 41 development textures
(opaque, zero, cutout and ramp alpha, and the paired opacity maps) plus the
committed `bc5_peak` corpus. Reach from a seed replay with LLVM coverage:
`bc5_peak` reached `peak_row_sse41` (2,000 calls) and 82% of `bc4.rs` lines;
`bc7_encode` reached 85% of `bc7_search.rs`, 79% of `bc7_objective.rs` and 87%
of `avx2.rs` lines.

| Target | Segments | Edges | Result |
| --- | --- | ---: | --- |
| `bc5_peak` | 2 x 600 s (278,164 executions), edges 861 then 861 | 861 | exit 0, no crash |
| `bc7_encode` | 2 x 600 s after the fix (203,409 executions), edges 5,098 then 5,098 | 5,098 | exit 0, no crash |

The first `bc7_encode` run found an overflow in `peak_choice`
(`src/bc7_search.rs`): `p + PEAK_GAIN <= peak` on `u8` overflows when a
finalist's peak error is 255 (a panic with overflow checks on, a wrap in a
release build). Twenty-seven crashing inputs, one distinct fault, minimal input
14 bytes. The comparison is now `peak.checked_sub(p)`, all 27 inputs replay
without a crash, and `peak_choice_accepts_a_peak_error_of_255` in
`src/bc7_search.rs` covers it, so the inputs were not kept. The rows above are
the runs after the fix.

Backend equality: the working corpora were replayed with `-runs=0` under
`TEXENC_SIMD=scalar`, `sse41` and the default dispatch. `bc5_peak`: 1,027
inputs, identical sorted digests (`9b254c777fa8d799`), exit 0. `bc7_encode`:
3,152 inputs, identical sorted digests (`4ad9a0f03c73183b`), exit 0, before and
after the fix. Both on x86_64 with AVX2; `-rss_limit_mb=512`
throughout.

#### 0.2.3 additions (BC7 transparent SIMD kernels, rounds 3 and 4)

`bc7_encode` gained near-grey colour (structured colour style 4; raw layout bit
6) and, in the raw layout, forced all-255 alpha (bit 4) and 0/255 cutout alpha
(bit 5), next to the existing opaque, cutout, all-zero and noise alpha. It
asserts that an image whose alpha is 255 everywhere decodes to alpha exactly
255 under both alpha policies, on top of the legal-mode, `encode` equals
`encode_into`, repeat and block-independence checks. Fast, Balanced and Quality
run with `Opaque` and `IgnoreTransparentColour`. Scalar against dispatched SIMD
is compared by corpus replay with `BC7_DIGEST_LOG`, as above.

Seeds: the 310 inputs of the previous `bc7_encode` corpus plus 934 raw-layout
inputs cut from 1 to 6 real 4x4 blocks of the 4096x4096 development textures, each with a random choice of
preset and of the new flags. Reach from a seed replay with LLVM coverage
(104-input shards, 5,459 edges): `bc7_search.rs` 90%, `bc7_fast.rs` 83%,
`bc7_objective.rs` 70%, `bc7_simd/avx2.rs` 87% and `bc7_simd/kernels.rs` 99.5%
of lines.

| Target | Segments | Executions | Edges | Result |
| --- | --- | ---: | --- | --- |
| `bc7_encode` | 5 x 601 s on one core (50 CPU-minutes) | 173,403; 98,213; 93,212; 91,515; 93,275 | 5,459 at start; 6,176, 6,177, 6,178, 6,179, 6,179 | exit 0 each, no crash |

The fifth segment added no edge in 601 s. All segments used
`-rss_limit_mb=512` and `ASAN_OPTIONS=quarantine_size_mb=16`. The final corpus
(5,838 files, not committed) was replayed with `-runs=0` under
`TEXENC_SIMD=scalar`, `sse41`, `avx2` and the default dispatch: 5,837 logged
inputs each (one file is too short for either layout and returns before the check), exit 0, and
the sorted output digests of the four runs are identical (`bcaf50e7b52fadb5`
as the first 16 hex digits of their SHA-256). No panic, overflow, illegal block,
alpha mismatch or backend mismatch was found. It does not cover NEON or
WebAssembly, or images larger than 23x23 texels.

#### 0.2.4 additions (BC7 rotation search for two-colour opaque blocks)

`bc7_encode` unchanged; the corpus of the 0.2.3 campaign (5,838 inputs) was
rerun against the new Quality search. Two 600 s chunks on one core, 155,250
executions, coverage 6,181 and 6,181 (the second chunk added none), no crash.

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
1/1/1 damped Gauss-Newton fitting iterations. All 32 partitions remain eligible for selection.
Decoded binary16 objective values use an exact software-libm lookup table
shared by native and wasm; source objective values still use software libm.
Transformed endpoint admission precedes bitstream packing. The factor 16
gives shadow detail weight around radiance 1/16 instead of 1. Source preparation and partition estimates are shared across preset stages;
only newly admitted partitions need initial candidate evaluation. Pair distances
are computed once for eight blocks in structure-of-arrays form, with SSE4.1/AVX2
runtime dispatch and a byte-identical scalar fallback on other targets.
Endpoint search and refinement remain scalar, with no per-block heap allocation.
Integer endpoint quantisation uses a shift and local reconstruction comparisons;
an exhaustive test checks it against the former binary search.

The pruning, bounded refinement, and block batching strategy follows the ideas
in Intel's [ISPC Texture Compressor algorithm description, sections 2–4](https://github.com/GameTechDev/ISPCTextureCompressor/blob/master/ISPC%20Texture%20Compressor.pdf).
That document describes BC7 and maps SIMD instances to independent blocks;
these ideas are applied here to BC6H without copying its implementation.

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
| Fast / fast | 0.018242 / 0.018640 | 46.88 / 46.84 | 150.03 / 7.49 |
| Balanced / basic | 0.017771 / 0.017845 | 47.07 / 47.10 | 201.69 / 19.13 |
| Quality / slow | 0.017517 / 0.017640 | 47.17 / 47.20 | 322.43 / 91.25 |

Our log RMSE is at or below Intel's at all three presets. Intel encodes faster
at every preset, so there is no BC6H speed claim.
These paired timings were collected on a shared host and retain scheduling noise. Full-size images, other HDR content, signed
quality against Intel, and hardware GPU decoding are not qualified by this run.
