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
#[non_exhaustive]
pub enum AlphaPolicy {
    /// Opaque BC1 blocks. BC3 and BC7 preserve input alpha with this policy.
    Opaque,
    /// Encode BC1 texels below the threshold as transparent.
    PunchThrough {
        /// Alpha values strictly below this byte are transparent.
        threshold: u8,
    },
}
/// Endpoint objective for unsigned BC4 and BC5 channels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Bc45Fit {
    /// The existing preset's squared-error fit (the default).
    #[default]
    SquaredError,
    /// Fit with the range fit's block peak-error bound.
    ///
    /// Starts with nearest selectors for the block's maximum/minimum endpoints.
    /// Fast and Balanced keep these endpoints; Quality searches for lower
    /// squared error without increasing that fit's maximum absolute byte error.
    /// BC5 applies the bound independently to R and G.
    /// BC5 also rejects refinements whose worst reconstructed positive-Z normal
    /// score is worse than the range fit's, using an integer approximation.
    /// These bounds are relative to that fit, not to every other encoder.
    PeakBounded,
}
/// Format, search preset and fitting policies used by an encode operation.
///
/// Build with [`Settings::new`] and the `with_*` methods; the struct is
/// `#[non_exhaustive]`, so it cannot be built with a struct literal outside
/// this crate and may gain fields in later releases.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Settings {
    /// Output block format.
    pub format: Format,
    /// Search preset.
    pub effort: Effort,
    /// Alpha policy; defaults to opaque BC1 and preserved BC3/BC7 alpha.
    pub alpha: AlphaPolicy,
    /// BC4/BC5 endpoint objective; ignored by BC1, BC3 and BC7.
    pub bc45_fit: Bc45Fit,
}
impl Settings {
    /// Create settings with [`AlphaPolicy::Opaque`].
    pub fn new(format: Format, effort: Effort) -> Self {
        Self {
            format,
            effort,
            alpha: AlphaPolicy::Opaque,
            bc45_fit: Bc45Fit::SquaredError,
        }
    }
    /// Return these settings with the given alpha policy.
    pub fn with_alpha(mut self, alpha: AlphaPolicy) -> Self {
        self.alpha = alpha;
        self
    }
    /// Return these settings with the given BC4/BC5 endpoint objective.
    pub fn with_bc45_fit(mut self, bc45_fit: Bc45Fit) -> Self {
        self.bc45_fit = bc45_fit;
        self
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
    if settings.format == Format::Bc5
        && settings.bc45_fit == Bc45Fit::PeakBounded
        && settings.effort != Effort::Quality
    {
        if image.channels == 4 {
            encode_peak_row::<4>(image, by, row);
        } else {
            encode_peak_row::<2>(image, by, row);
        }
        return;
    }

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
            Format::Bc4 => dst.copy_from_slice(&bc4::encode_with_fit(
                &pixels,
                settings.effort,
                0,
                settings.bc45_fit,
            )),
            Format::Bc3 => {
                dst[..8].copy_from_slice(&bc4::encode(&pixels, settings.effort, 3));
                let color = if settings.effort == Effort::Balanced {
                    bc1::encode_bc3_balanced(&pixels)
                } else {
                    bc1::encode(&pixels, settings.effort, AlphaPolicy::Opaque)
                };
                dst[8..].copy_from_slice(&color);
            }
            Format::Bc5 => dst.copy_from_slice(&bc4::encode_pair(
                &pixels,
                settings.effort,
                settings.bc45_fit,
            )),
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

// Stage only the two fitted channels, and keep their output widths static.
// The generic row encoder remains the path for defaults and Quality.
fn encode_peak_row<const CHANNELS: usize>(image: ImageView<'_>, by: usize, row: &mut [u8]) {
    #[cfg(all(feature = "simd", feature = "std", target_arch = "x86_64"))]
    if let Some(encoder) = bc7::range_encoder_sse41() {
        encode_peak_row_with::<CHANNELS>(image, by, row, |values| encoder.encode(values));
        return;
    }
    #[cfg(all(feature = "simd", feature = "std", target_arch = "x86_64"))]
    if let Some(encoder) = bc7::range_encoder() {
        encode_peak_row_with::<CHANNELS>(image, by, row, |values| encoder.encode(values));
        return;
    }
    encode_peak_row_with::<CHANNELS>(image, by, row, bc4::range_fit_values);
}

fn encode_peak_row_with<const CHANNELS: usize>(
    image: ImageView<'_>,
    by: usize,
    row: &mut [u8],
    fit: impl Fn(&[u8; 16]) -> [u8; 8],
) {
    let source_rows: [usize; 4] =
        core::array::from_fn(|y| (by * 4 + y).min(image.height as usize - 1) * image.stride);
    for (bx, dst) in row.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        let mut red = [0; 16];
        let mut green = [0; 16];
        let complete = bx < image.width as usize / 4;
        for (y, &source_row) in source_rows.iter().enumerate() {
            if complete {
                // One checked span per source row; each texel index is fixed.
                let start = source_row + bx * 4 * CHANNELS;
                let source = &image.data[start..start + 4 * CHANNELS];
                for x in 0..4 {
                    red[y * 4 + x] = source[x * CHANNELS];
                    green[y * 4 + x] = source[x * CHANNELS + 1];
                }
            } else {
                for x in 0..4 {
                    let px = (bx * 4 + x).min(image.width as usize - 1);
                    let off = source_row + px * CHANNELS;
                    red[y * 4 + x] = image.data[off];
                    green[y * 4 + x] = image.data[off + 1];
                }
            }
        }
        dst[..8].copy_from_slice(&fit(&red));
        dst[8..16].copy_from_slice(&fit(&green));
    }
}

#[cfg(test)]
mod peak_row_tests {
    use super::*;

    #[test]
    fn peak_rows_match_block_path_with_padding_and_strides() -> Result<(), Error> {
        for channels in [2, 4] {
            for width in [1, 3, 4, 5, 13] {
                for height in [1, 3, 4, 7] {
                    let stride = width * channels + 7;
                    let mut data = alloc::vec![0; stride * height];
                    for (byte, value) in data.iter_mut().zip((0..=255u8).cycle()) {
                        *byte = value;
                    }
                    let image = if channels == 2 {
                        ImageView::rg8(
                            &data,
                            u32::try_from(width).map_err(|_| Error::Overflow)?,
                            u32::try_from(height).map_err(|_| Error::Overflow)?,
                            stride,
                        )?
                    } else {
                        ImageView::rgba8(
                            &data,
                            u32::try_from(width).map_err(|_| Error::Overflow)?,
                            u32::try_from(height).map_err(|_| Error::Overflow)?,
                            stride,
                        )?
                    };
                    for effort in [Effort::Fast, Effort::Balanced] {
                        let mut settings = Settings::new(Format::Bc5, effort);
                        settings.bc45_fit = Bc45Fit::PeakBounded;
                        let actual = encode(image, &settings)?;
                        let mut expected = Vec::new();
                        for by in 0..height.div_ceil(4) {
                            for bx in 0..width.div_ceil(4) {
                                let pixels = core::array::from_fn(|i| {
                                    let x = (bx * 4 + i % 4).min(width - 1);
                                    let y = (by * 4 + i / 4).min(height - 1);
                                    let off = y * stride + x * channels;
                                    [data[off], data[off + 1], 0, 255]
                                });
                                expected.extend_from_slice(&bc4::encode_pair(
                                    &pixels,
                                    effort,
                                    Bc45Fit::PeakBounded,
                                ));
                            }
                        }
                        assert_eq!(actual, expected, "{channels} {width} {height} {effort:?}");
                    }
                }
            }
        }
        Ok(())
    }
}
