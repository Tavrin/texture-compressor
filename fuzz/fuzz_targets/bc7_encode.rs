//! BC7 Fast/Balanced/Quality on arbitrary RGBA8 images up to 23x23 texels with
//! random alpha patterns (opaque, cutout, transparent, noise). Checks: no panic,
//! output length, every block decodes through an independent reference decoder
//! (legal mode), `encode` equals `encode_into`, a repeat encode is identical,
//! and the first full block encodes identically alone and inside the image
//! (independence from batch position). With `BC7_DIGEST_LOG=<file>` each input's
//! output digest is appended, so a replay under `TEXENC_SIMD=scalar` can be
//! compared byte for byte against a replay under the dispatched backend.
#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Write;
use texture_compressor::{encode, encode_into, encoded_len, Effort, Format, ImageView, Settings};

#[path = "../../tests/bc7_decoder.rs"]
#[allow(dead_code)]
mod reference;

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}

fuzz_target!(|data: &[u8]| {
    // Raw layout: high bit of byte 0 set. Bits 0-1 pick Balanced or Quality
    // (0 and 3 are Fast), the rest is RGBA8 texels, 64 bytes per 4x4 block, up
    // to six blocks in one row. Mutations then edit real block bytes directly.
    if let [first, rest @ ..] = data {
        if first & 0x80 != 0 {
            let effort = match first & 3 {
                1 => Effort::Balanced,
                2 => Effort::Quality,
                _ => Effort::Fast,
            };
            let blocks = (rest.len() / 64).clamp(1, 6);
            let width = 4 * blocks as u32;
            let mut px = vec![0u8; blocks * 64];
            for (row, line) in px.chunks_exact_mut(blocks * 16).enumerate() {
                for (i, v) in line.iter_mut().enumerate() {
                    let (b, o) = (i / 16, i % 16);
                    let at = b * 64 + row * 16 + o;
                    *v = rest.get(at).copied().unwrap_or(0);
                }
            }
            check(data, effort, width, 4, blocks * 16, &px);
            return;
        }
    }
    let [mode, w, h, shape, slack, rest @ ..] = data else {
        return;
    };
    let effort = match mode % 3 {
        0 => Effort::Fast,
        1 => Effort::Balanced,
        _ => Effort::Quality,
    };
    let width = u32::from(*w) % 23 + 1;
    let height = u32::from(*h) % 23 + 1;
    let stride = width as usize * 4 + usize::from(*slack % 5);
    let count = (height as usize - 1) * stride + width as usize * 4;
    let src = |i: usize| -> u8 {
        if rest.is_empty() {
            0
        } else {
            rest[i % rest.len()]
        }
    };
    let color_style = (shape >> 4) % 4;
    let alpha_style = shape & 7;
    let mut px = vec![0u8; count];
    for i in 0..count {
        let texel = i / 4;
        let c = i % 4;
        px[i] = match (c, color_style) {
            (3, _) => match alpha_style {
                0 => src(i),
                1 => 255,
                2 => {
                    if src(texel) & 1 == 0 {
                        0
                    } else {
                        255
                    }
                }
                3 => 0,
                4 => {
                    if src(i) < 128 {
                        0
                    } else {
                        255
                    }
                }
                5 => 255 - (src(texel) & 3),
                6 => src(i) & 3,
                _ => {
                    if texel % 16 < 3 {
                        0
                    } else {
                        255
                    }
                }
            },
            (_, 0) => src(i),
            // Few distinct colours: stresses single-colour and low-rank paths.
            (_, 1) => src(c + 4 * usize::from(src(texel) & 3)),
            // Smooth ramp with sparse noise.
            (_, 2) => (texel as u8).wrapping_mul(7).wrapping_add(src(c) & 15),
            _ => src(c),
        };
    }
    check(data, effort, width, height, stride, &px);
});

fn check(data: &[u8], effort: Effort, width: u32, height: u32, stride: usize, px: &[u8]) {
    let view = ImageView::rgba8(px, width, height, stride).unwrap();
    let settings = Settings::new(Format::Bc7, effort);
    let out = encode(view, &settings).unwrap();
    let len = encoded_len(Format::Bc7, width, height).unwrap();
    assert_eq!(out.len(), len);
    for block in out.chunks_exact(16) {
        assert_ne!(block[0], 0, "reserved BC7 mode");
        let _ = reference::decode(block);
    }
    let mut into = vec![0u8; len];
    encode_into(view, &settings, &mut into).unwrap();
    assert_eq!(out, into, "encode vs encode_into");
    assert_eq!(out, encode(view, &settings).unwrap(), "nondeterministic");
    if width >= 4 && height >= 4 {
        let mut cell = [0u8; 64];
        for y in 0..4 {
            cell[y * 16..y * 16 + 16].copy_from_slice(&px[y * stride..y * stride + 16]);
        }
        let single = encode(ImageView::rgba8(&cell, 4, 4, 16).unwrap(), &settings).unwrap();
        assert_eq!(&out[..16], &single[..], "block differs alone vs in image");
    }
    if let Ok(path) = std::env::var("BC7_DIGEST_LOG") {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "{:016x} {:016x}", fnv(data), fnv(&out));
        }
    }
}
