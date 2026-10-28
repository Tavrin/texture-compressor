#![no_std]
#![forbid(unsafe_code)]
extern crate alloc;
#[cfg(feature = "parallel")]
extern crate std;
use alloc::vec::Vec;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Bc1,
    Bc3,
    Bc4,
    Bc5,
    /// BC7. The Fast preset is qualified; Balanced and Quality are still experimental.
    Bc7,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effort {
    /// Fastest search. BC7 fits each subset along its principal axis, refines
    /// endpoints by least squares, and searches further only where a block's
    /// error stays high or its colours are few.
    Fast,
    /// Balanced search. For BC1, BC3, BC4 and BC5 it costs fewer instructions than
    /// common reference encoders at higher quality; for BC7 it is still experimental.
    Balanced,
    Quality,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaPolicy {
    /// Opaque BC1 blocks. BC3 and BC7 preserve input alpha with this policy.
    Opaque,
    PunchThrough {
        threshold: u8,
    },
}
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub format: Format,
    pub effort: Effort,
    pub alpha: AlphaPolicy,
}
impl Settings {
    pub fn new(format: Format, effort: Effort) -> Self {
        Self {
            format,
            effort,
            alpha: AlphaPolicy::Opaque,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    EmptyImage,
    InvalidStride,
    ShortInput,
    ShortOutput,
    Overflow,
    AllocationFailed,
    UnsupportedPolicy,
}
#[derive(Clone, Copy, Debug)]
pub struct ImageView<'a> {
    data: &'a [u8],
    width: u32,
    height: u32,
    stride: usize,
    channels: usize,
}
impl<'a> ImageView<'a> {
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn rgba8(data: &'a [u8], width: u32, height: u32, stride: usize) -> Result<Self, Error> {
        Self::new(data, width, height, stride, 4)
    }
    pub fn r8(data: &'a [u8], width: u32, height: u32, stride: usize) -> Result<Self, Error> {
        Self::new(data, width, height, stride, 1)
    }
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
        Format::Bc3 | Format::Bc5 | Format::Bc7 => 16,
    }
}

pub fn encode(image: ImageView<'_>, settings: &Settings) -> Result<Vec<u8>, Error> {
    let len = encoded_len(settings.format, image.width, image.height)?;
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|_| Error::AllocationFailed)?;
    out.resize(len, 0);
    encode_into(image, settings, &mut out)?;
    Ok(out)
}
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
    if (matches!(settings.format, Format::Bc1 | Format::Bc3 | Format::Bc7) && image.channels != 4)
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
        }
    }
}

mod bc1;
mod bc4;
mod bc7;
