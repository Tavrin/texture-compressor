//! Arbitrary HDR samples (any f32/binary16 bit pattern, so NaN, infinities and
//! subnormals occur) through every BC6H encode entry point, then the decoder.
#![no_main]
use libfuzzer_sys::fuzz_target;
use texture_compressor::{
    decode_bc6h, encode_hdr, encode_hdr_into, encoded_len, Effort, Format, HdrImageView, Settings,
};

fuzz_target!(|data: &[u8]| {
    let [mode, w, h, slack, rest @ ..] = data else {
        return;
    };
    let format = if mode & 1 == 0 {
        Format::Bc6hUf16
    } else {
        Format::Bc6hSf16
    };
    let effort = match (mode >> 1) % 3 {
        0 => Effort::Fast,
        1 => Effort::Balanced,
        _ => Effort::Quality,
    };
    let kind = (mode >> 3) & 3;
    let channels = if kind & 1 == 0 { 3 } else { 4 };
    let width = u32::from(*w) % 19 + 1;
    let height = u32::from(*h) % 19 + 1;
    let stride = width as usize * channels + usize::from(*slack % 5);
    let count = (height as usize - 1) * stride + width as usize * channels;
    let settings = Settings::new(format, effort);
    let take = |size: usize, i: usize, out: &mut [u8]| {
        if rest.is_empty() {
            out.fill(0);
        } else {
            for (k, b) in out.iter_mut().enumerate() {
                *b = rest[(i * size + k) % rest.len()];
            }
        }
    };
    let blocks = if kind < 2 {
        let samples: Vec<f32> = (0..count)
            .map(|i| {
                let mut b = [0u8; 4];
                take(4, i, &mut b);
                f32::from_bits(u32::from_le_bytes(b))
            })
            .collect();
        let Ok(image) = (if channels == 3 {
            HdrImageView::rgb_f32(&samples, width, height, stride)
        } else {
            HdrImageView::rgba_f32(&samples, width, height, stride)
        }) else {
            return;
        };
        run(image, &settings, width, height)
    } else {
        let samples: Vec<u16> = (0..count)
            .map(|i| {
                let mut b = [0u8; 2];
                take(2, i, &mut b);
                u16::from_le_bytes(b)
            })
            .collect();
        let Ok(image) = (if channels == 3 {
            HdrImageView::rgb_f16(&samples, width, height, stride)
        } else {
            HdrImageView::rgba_f16(&samples, width, height, stride)
        }) else {
            return;
        };
        run(image, &settings, width, height)
    };
    for block in blocks.chunks_exact(16) {
        let block: &[u8; 16] = block.try_into().unwrap();
        let texels = decode_bc6h(block, format).unwrap();
        if format == Format::Bc6hUf16 {
            // Encoder clamps to +/-65504 and zero; decoded output stays finite.
            assert!(texels.iter().flatten().all(|&v| v & 0x7C00 != 0x7C00));
        }
    }
});

fn run(image: HdrImageView<'_>, settings: &Settings, width: u32, height: u32) -> Vec<u8> {
    let blocks = encode_hdr(image, settings).unwrap();
    let len = encoded_len(settings.format, width, height).unwrap();
    assert_eq!(blocks.len(), len);
    let mut padded = vec![0xA5u8; len + 7];
    encode_hdr_into(image, settings, &mut padded).unwrap();
    assert_eq!(&padded[..len], &blocks[..]);
    assert!(padded[len..].iter().all(|&b| b == 0xA5));
    blocks
}
