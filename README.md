# texture-compressor

Encode decoded pixels into GPU block-compressed BC1, BC3, BC4, BC5 and BC7
textures with Fast, Balanced and Quality search presets. Encoding is
deterministic for a given input and settings. The crate supports native and
WebAssembly targets. It returns raw blocks; it does not decode image files,
generate mipmaps, or write texture containers.

## Install

Requires Rust 1.98 or later (MSRV).

```toml
[dependencies]
texture-compressor = "0.1"
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
