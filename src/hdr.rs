use crate::{bc6h, encoded_len, AlphaPolicy, Error, Format, Settings};
use alloc::vec::Vec;
#[derive(Clone, Copy, Debug)]
enum Samples<'a> {
    Half(&'a [u16]),
    Float(&'a [f32]),
}
/// Borrowed RGB or RGBA HDR samples. Strides count samples, not bytes.
/// Half inputs use IEEE 754 binary16 bit patterns. Alpha is ignored.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct HdrImageView<'a> {
    data: Samples<'a>,
    width: u32,
    height: u32,
    stride: usize,
    channels: usize,
}
impl<'a> HdrImageView<'a> {
    /// Borrow RGB f32 samples; stride counts f32 values.
    pub fn rgb_f32(data: &'a [f32], width: u32, height: u32, stride: usize) -> Result<Self, Error> {
        Self::new(Samples::Float(data), width, height, stride, 3)
    }
    /// Borrow RGBA f32 samples; alpha is ignored and stride counts f32 values.
    pub fn rgba_f32(
        data: &'a [f32],
        width: u32,
        height: u32,
        stride: usize,
    ) -> Result<Self, Error> {
        Self::new(Samples::Float(data), width, height, stride, 4)
    }
    /// Borrow RGB binary16 bit patterns; stride counts u16 values.
    pub fn rgb_f16(data: &'a [u16], width: u32, height: u32, stride: usize) -> Result<Self, Error> {
        Self::new(Samples::Half(data), width, height, stride, 3)
    }
    /// Borrow RGBA binary16 bit patterns; alpha is ignored; stride counts u16 values.
    pub fn rgba_f16(
        data: &'a [u16],
        width: u32,
        height: u32,
        stride: usize,
    ) -> Result<Self, Error> {
        Self::new(Samples::Half(data), width, height, stride, 4)
    }
    /// Width in texels.
    pub fn width(&self) -> u32 {
        self.width
    }
    /// Height in texels.
    pub fn height(&self) -> u32 {
        self.height
    }
    fn new(
        data: Samples<'a>,
        width: u32,
        height: u32,
        stride: usize,
        channels: usize,
    ) -> Result<Self, Error> {
        if width == 0 || height == 0 {
            return Err(Error::EmptyImage);
        }
        let w = usize::try_from(width).map_err(|_| Error::Overflow)?;
        let h = usize::try_from(height).map_err(|_| Error::Overflow)?;
        let row = w.checked_mul(channels).ok_or(Error::Overflow)?;
        if stride < row {
            return Err(Error::InvalidStride);
        }
        let extent = (h - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(row))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(Error::Overflow)?;
        let len = match data {
            Samples::Half(s) => s.len(),
            Samples::Float(s) => s.len(),
        };
        if len < extent {
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
/// Allocate BC6H blocks. Partial blocks replicate edges; alpha is ignored.
/// NaN becomes zero, infinities and finite overflow clamp to +/-65504;
/// unsigned input additionally clamps negatives to zero. Subnormals are retained.
/// Returns validation, allocation, or size errors as in [`encode_hdr_into`].
pub fn encode_hdr(image: HdrImageView<'_>, settings: &Settings) -> Result<Vec<u8>, Error> {
    validate(settings)?;
    let n = encoded_len(settings.format, image.width, image.height)?;
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| Error::AllocationFailed)?;
    out.resize(n, 0);
    encode_hdr_into(image, settings, &mut out)?;
    Ok(out)
}
fn validate(settings: &Settings) -> Result<bool, Error> {
    if !matches!(settings.alpha, AlphaPolicy::Opaque) {
        return Err(Error::UnsupportedPolicy);
    }
    match settings.format {
        Format::Bc6hUf16 => Ok(false),
        Format::Bc6hSf16 => Ok(true),
        _ => Err(Error::UnsupportedPolicy),
    }
}
/// Encode HDR samples into the first [`encoded_len`] output bytes.
/// Returns `UnsupportedPolicy` for non-BC6H formats or punch-through alpha,
/// `ShortOutput` for insufficient space, or `Overflow` for size overflow.
/// Input normalization and edge handling match [`encode_hdr`].
pub fn encode_hdr_into(
    image: HdrImageView<'_>,
    settings: &Settings,
    out: &mut [u8],
) -> Result<(), Error> {
    let signed = validate(settings)?;
    let len = encoded_len(settings.format, image.width, image.height)?;
    if out.len() < len {
        return Err(Error::ShortOutput);
    }
    let w = usize::try_from(image.width).map_err(|_| Error::Overflow)?;
    let h = usize::try_from(image.height).map_err(|_| Error::Overflow)?;
    let bw = w.div_ceil(4);
    for (i, dst) in out[..len].as_chunks_mut::<16>().0.iter_mut().enumerate() {
        let mut pixels = [[0.0; 3]; 16];
        for (j, p) in pixels.iter_mut().enumerate() {
            let x = (i % bw * 4 + j % 4).min(w - 1);
            let y = (i / bw * 4 + j / 4).min(h - 1);
            for (c, v) in p.iter_mut().enumerate() {
                let off = y * image.stride + x * image.channels + c;
                let x = match image.data {
                    Samples::Half(s) => bc6h::half(s[off]),
                    Samples::Float(s) => s[off],
                };
                *v = if x.is_nan() {
                    0.0
                } else {
                    x.clamp(if signed { -65504.0 } else { 0.0 }, 65504.0)
                };
            }
        }
        dst.copy_from_slice(&bc6h::encode(&pixels, signed, settings.effort));
    }
    Ok(())
}
/// Decode one BC6H block to IEEE binary16 RGB bit patterns, exactly following
/// all fourteen modes and thirty-two partitions. Reserved modes return black.
/// `format` must be `Bc6hUf16` or `Bc6hSf16`, else `UnsupportedPolicy` is returned.
/// Signed mode 15 can decode negative infinity, as required by the format.
pub fn decode_bc6h(block: &[u8; 16], format: Format) -> Result<[[u16; 3]; 16], Error> {
    match format {
        Format::Bc6hUf16 => Ok(bc6h::decode(block, false)),
        Format::Bc6hSf16 => Ok(bc6h::decode(block, true)),
        _ => Err(Error::UnsupportedPolicy),
    }
}
