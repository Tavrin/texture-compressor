#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![no_std]
#![cfg_attr(not(feature = "simd"), forbid(unsafe_code))]
#![deny(unsafe_code)]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;
use alloc::vec::Vec;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// GPU block-compression format. BC4 and BC5 encode unsigned channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Format {
    /// RGB with optional one-bit alpha, eight bytes per 4x4 block.
    Bc1,
    /// RGB with a separate alpha channel, sixteen bytes per 4x4 block.
    Bc3,
    /// Red channel, eight bytes per 4x4 block.
    Bc4,
    /// Red and green channels, sixteen bytes per 4x4 block.
    Bc5,
    /// RGBA, sixteen bytes per 4x4 block.
    Bc7,
    /// Unsigned RGB half-float HDR, sixteen bytes per 4x4 block.
    Bc6hUf16,
    /// Signed RGB half-float HDR, sixteen bytes per 4x4 block.
    Bc6hSf16,
}
/// Encoder search preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Effort {
    /// Limited search. BC7 fits each subset along its principal axis, refines
    /// endpoints by least squares, and searches further only where a block's
    /// error stays high or its colours are few.
    Fast,
    /// Intermediate search budget between Fast and Quality.
    Balanced,
    /// Broader search. BC7 adds more partition and rotation candidates,
    /// including three-subset modes, and extra endpoint refinement.
    Quality,
}
/// Alpha handling; punch-through is supported only for BC1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaPolicy {
    /// Opaque BC1 blocks. BC3 and BC7 preserve input alpha with this policy.
    Opaque,
    /// Encode BC1 texels below the threshold as transparent.
    PunchThrough {
        /// Alpha values strictly below this byte are transparent.
        threshold: u8,
    },
}
/// Format, search preset and alpha policy used by an encode operation.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Output block format.
    pub format: Format,
    /// Search preset.
    pub effort: Effort,
    /// Alpha policy; defaults to opaque BC1 and preserved BC3/BC7 alpha.
    pub alpha: AlphaPolicy,
}
impl Settings {
    /// Create settings with [`AlphaPolicy::Opaque`].
    pub fn new(format: Format, effort: Effort) -> Self {
        Self {
            format,
            effort,
            alpha: AlphaPolicy::Opaque,
        }
    }
}
/// Image validation or encoding failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Width or height is zero.
    EmptyImage,
    /// The byte stride is smaller than a complete input row.
    InvalidStride,
    /// The input slice does not contain every visible texel.
    ShortInput,
    /// The output slice is smaller than [`encoded_len`].
    ShortOutput,
    /// Image or output size cannot be represented.
    Overflow,
    /// Reserving the output buffer failed.
    AllocationFailed,
    /// Input channels or alpha policy are incompatible with the output format.
    UnsupportedPolicy,
}
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Error::EmptyImage => "image width or height is zero",
            Error::InvalidStride => "row stride is smaller than a complete input row",
            Error::ShortInput => "input does not contain every visible texel",
            Error::ShortOutput => "output buffer is smaller than the encoded length",
            Error::Overflow => "image or output size cannot be represented",
            Error::AllocationFailed => "reserving the output buffer failed",
            Error::UnsupportedPolicy => {
                "input channels or alpha policy are incompatible with the output format"
            }
        })
    }
}
#[cfg(feature = "std")]
impl std::error::Error for Error {}
/// Borrowed strided R8, RG8 or RGBA8 image with validated dimensions and storage.
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a> {
    data: &'a [u8],
    width: u32,
    height: u32,
    stride: usize,
    channels: usize,
}
impl<'a> ImageView<'a> {
    /// Image width in texels.
    pub fn width(&self) -> u32 {
        self.width
    }
    /// Image height in texels.
    pub fn height(&self) -> u32 {
        self.height
    }
    /// Borrow RGBA8 pixels with a byte stride.
    ///
    /// Fails for zero dimensions, a short stride or input slice, or size overflow.
    /// Padding after the final visible row is not required.
    pub fn rgba8(data: &'a [u8], width: u32, height: u32, stride: usize) -> Result<Self, Error> {
        Self::new(data, width, height, stride, 4)
    }
    /// Borrow single-channel unsigned pixels with a byte stride.
    ///
    /// Fails for zero dimensions, a short stride or input slice, or size overflow.
    /// Padding after the final visible row is not required.
    pub fn r8(data: &'a [u8], width: u32, height: u32, stride: usize) -> Result<Self, Error> {
        Self::new(data, width, height, stride, 1)
    }
    /// Borrow two-channel unsigned pixels with a byte stride.
    ///
    /// Fails for zero dimensions, a short stride or input slice, or size overflow.
    /// Padding after the final visible row is not required.
    pub fn rg8(data: &'a [u8], width: u32, height: u32, stride: usize) -> Result<Self, Error> {
        Self::new(data, width, height, stride, 2)
    }
    fn new(
        data: &'a [u8],
        width: u32,
        height: u32,
        stride: usize,
        channels: usize,
    ) -> Result<Self, Error> {
        if width == 0 || height == 0 {
            return Err(Error::EmptyImage);
        }
        let row = (width as usize)
            .checked_mul(channels)
            .ok_or(Error::Overflow)?;
        if stride < row {
            return Err(Error::InvalidStride);
        }
        let extent = (height as usize - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(row))
            .ok_or(Error::Overflow)?;
        if data.len() < extent {
            return Err(Error::ShortInput);
        }
        Ok(Self {
            data,
            width,
            height,
            stride,
            channels,
        })
    }
}
/// Output bytes required for the format and dimensions, rounding up to 4x4 blocks.
///
/// Returns [`Error::EmptyImage`] for zero dimensions or [`Error::Overflow`]
/// if the output size exceeds the supported address range.
pub fn encoded_len(format: Format, width: u32, height: u32) -> Result<usize, Error> {
    if width == 0 || height == 0 {
        return Err(Error::EmptyImage);
    }
    let w = (width as usize).checked_add(3).ok_or(Error::Overflow)? / 4;
    let h = (height as usize).checked_add(3).ok_or(Error::Overflow)? / 4;
    let len = w
        .checked_mul(h)
        .and_then(|x| x.checked_mul(block_size(format)))
        .ok_or(Error::Overflow)?;
    if len > isize::MAX as usize {
        return Err(Error::Overflow);
    }
    Ok(len)
}
fn block_size(format: Format) -> usize {
    match format {
        Format::Bc1 | Format::Bc4 => 8,
        Format::Bc3 | Format::Bc5 | Format::Bc7 | Format::Bc6hUf16 | Format::Bc6hSf16 => 16,
    }
}

/// Allocate and encode an image into blocks in row order.
///
/// BC1/BC3/BC7 require RGBA8; BC4 reads R from any view; BC5 reads R/G
/// from RG8 or RGBA8. Partial blocks replicate the nearest edge texel.
/// Returns validation errors described by [`encode_into`], size overflow,
/// or [`Error::AllocationFailed`].
pub fn encode(image: ImageView<'_>, settings: &Settings) -> Result<Vec<u8>, Error> {
    let len = encoded_len(settings.format, image.width, image.height)?;
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| Error::AllocationFailed)?;
    out.resize(len, 0);
    encode_into(image, settings, &mut out)?;
    Ok(out)
}
/// Encode into the first [`encoded_len`] bytes of `out`, leaving the rest untouched.
///
/// Uses the channel mapping and edge replication described by [`encode`].
/// Returns [`Error::UnsupportedPolicy`] for incompatible channels or a
/// punch-through policy outside BC1, [`Error::ShortOutput`] for insufficient
/// output space, or [`Error::Overflow`] for an unrepresentable output size.
pub fn encode_into(image: ImageView<'_>, settings: &Settings, out: &mut [u8]) -> Result<(), Error> {
    let (bw, len) = validate(image, settings, out.len())?;
    for (by, row) in out[..len]
        .chunks_mut(bw * block_size(settings.format))
        .enumerate()
    {
        encode_row(image, settings, by, row);
    }
    Ok(())
}

/// Encode into a caller-provided buffer using the current Rayon pool.
///
/// Requires the `parallel` feature. Each block row is independent and produces
/// exactly the same bytes as [`encode_into`], including for padded edges.
/// Channel requirements, output layout and errors are the same as [`encode_into`].
#[cfg(feature = "parallel")]
pub fn encode_into_parallel(
    image: ImageView<'_>,
    settings: &Settings,
    out: &mut [u8],
) -> Result<(), Error> {
    let (bw, len) = validate(image, settings, out.len())?;
    out[..len]
        .par_chunks_mut(bw * block_size(settings.format))
        .enumerate()
        .for_each(|(by, row)| {
            encode_row(image, settings, by, row);
        });
    Ok(())
}

/// Allocate and encode using the current Rayon pool.
///
/// Requires `parallel`. Input requirements, output layout and errors match
/// [`encode`]; output bytes match sequential encoding.
#[cfg(feature = "parallel")]
pub fn encode_parallel(image: ImageView<'_>, settings: &Settings) -> Result<Vec<u8>, Error> {
    let len = encoded_len(settings.format, image.width, image.height)?;
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| Error::AllocationFailed)?;
    out.resize(len, 0);
    encode_into_parallel(image, settings, &mut out)?;
    Ok(out)
}

fn validate(
    image: ImageView<'_>,
    settings: &Settings,
    output_len: usize,
) -> Result<(usize, usize), Error> {
    if matches!(settings.format, Format::Bc6hUf16 | Format::Bc6hSf16)
        || (matches!(settings.format, Format::Bc1 | Format::Bc3 | Format::Bc7)
            && image.channels != 4)
        || (settings.format == Format::Bc5 && !matches!(image.channels, 2 | 4))
        || (settings.format != Format::Bc1 && !matches!(settings.alpha, AlphaPolicy::Opaque))
    {
        return Err(Error::UnsupportedPolicy);
    }
    let len = encoded_len(settings.format, image.width, image.height)?;
    if output_len < len {
        return Err(Error::ShortOutput);
    }
    Ok(((image.width as usize).div_ceil(4), len))
}

fn encode_row(image: ImageView<'_>, settings: &Settings, by: usize, row: &mut [u8]) {
    #[cfg(feature = "simd")]
    if settings.format == Format::Bc7 && settings.effort != Effort::Fast {
        let lanes = bc7::batch_width();
        for (batch, dst) in row.chunks_mut(lanes * 16).enumerate() {
            let mut blocks = [[[0u8; 4]; 16]; 8];
            for (slot, block) in blocks.iter_mut().enumerate().take(dst.len() / 16) {
                for y in 0..4 {
                    for x in 0..4 {
                        let px = ((batch * lanes + slot) * 4 + x).min(image.width as usize - 1);
                        let py = (by * 4 + y).min(image.height as usize - 1);
                        let off = py * image.stride + px * 4;
                        block[y * 4 + x].copy_from_slice(&image.data[off..off + 4]);
                    }
                }
            }
            bc7::encode_batch(&blocks, dst, settings.effort);
        }
        return;
    }
    for (bx, dst) in row
        .chunks_exact_mut(block_size(settings.format))
        .enumerate()
    {
        let mut pixels = [[0u8; 4]; 16];
        for y in 0..4 {
            for x in 0..4 {
                let px = (bx * 4 + x).min(image.width as usize - 1);
                let py = (by * 4 + y).min(image.height as usize - 1);
                let off = py * image.stride + px * image.channels;
                if image.channels == 4 {
                    pixels[y * 4 + x].copy_from_slice(&image.data[off..off + 4]);
                } else if image.channels == 2 {
                    pixels[y * 4 + x] = [image.data[off], image.data[off + 1], 0, 255];
                } else {
                    pixels[y * 4 + x] = [image.data[off], 0, 0, 255];
                }
            }
        }
        match settings.format {
            Format::Bc1 => {
                dst.copy_from_slice(&bc1::encode(&pixels, settings.effort, settings.alpha))
            }
            Format::Bc4 => dst.copy_from_slice(&bc4::encode(&pixels, settings.effort, 0)),
            Format::Bc3 => {
                dst[..8].copy_from_slice(&bc4::encode(&pixels, settings.effort, 3));
                let color = if settings.effort == Effort::Balanced {
                    bc1::encode_bc3_balanced(&pixels)
                } else {
                    bc1::encode(&pixels, settings.effort, AlphaPolicy::Opaque)
                };
                dst[8..].copy_from_slice(&color);
            }
            Format::Bc5 => {
                dst[..8].copy_from_slice(&bc4::encode(&pixels, settings.effort, 0));
                dst[8..].copy_from_slice(&bc4::encode(&pixels, settings.effort, 1));
            }
            Format::Bc7 => dst.copy_from_slice(&bc7::encode(&pixels, settings.effort)),
            Format::Bc6hUf16 | Format::Bc6hSf16 => unreachable!("validated byte input"),
        }
    }
}

mod bc1;
mod bc4;
mod bc7;
mod budget;

mod bc6h;
mod hdr;
pub use hdr::{decode_bc6h, encode_hdr, encode_hdr_into, HdrImageView};
