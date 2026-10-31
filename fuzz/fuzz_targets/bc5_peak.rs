//! BC5 and BC4 `PeakBounded` on arbitrary strided RG8/RGBA8/R8 input. Checks
//! the documented bound: no channel's peak absolute byte error in any block
//! exceeds that of the range fit (PeakBounded Balanced keeps the range fit).
#![no_main]
use libfuzzer_sys::fuzz_target;
use texture_compressor::{encode, Bc45Fit, Effort, Format, ImageView, Settings};

/// Reference BC4 palette: interpolants are rounded to nearest, as in the
/// integration tests' independent decoder.
fn bc4_table(b: &[u8]) -> [i32; 8] {
    let (a, c) = (i32::from(b[0]), i32::from(b[1]));
    let mut t = [a, c, 0, 0, 0, 0, 0, 0];
    if a > c {
        for i in 1..7 {
            t[i + 1] = ((7 - i as i32) * a + i as i32 * c + 3) / 7;
        }
    } else {
        for i in 1..5 {
            t[i + 1] = ((5 - i as i32) * a + i as i32 * c + 2) / 5;
        }
        t[6] = 0;
        t[7] = 255;
    }
    t
}

fn decode_bc4(b: &[u8]) -> [i32; 16] {
    let t = bc4_table(b);
    let mut bits = 0u64;
    for (i, &x) in b[2..8].iter().enumerate() {
        bits |= u64::from(x) << (8 * i);
    }
    core::array::from_fn(|i| t[((bits >> (3 * i)) & 7) as usize])
}

fuzz_target!(|data: &[u8]| {
    let [mode, w, h, slack, rest @ ..] = data else {
        return;
    };
    let effort = match mode % 3 {
        0 => Effort::Fast,
        1 => Effort::Balanced,
        _ => Effort::Quality,
    };
    let channels = [1usize, 2, 4][usize::from((mode >> 2) % 3)];
    let format = if channels == 1 {
        Format::Bc4
    } else {
        Format::Bc5
    };
    let width = u32::from(*w) % 23 + 1;
    let height = u32::from(*h) % 23 + 1;
    let stride = width as usize * channels + usize::from(*slack % 5);
    let count = (height as usize - 1) * stride + width as usize * channels;
    let px: Vec<u8> = (0..count)
        .map(|i| {
            if rest.is_empty() {
                0
            } else {
                rest[i % rest.len()]
            }
        })
        .collect();
    let view = match channels {
        1 => ImageView::r8(&px, width, height, stride),
        2 => ImageView::rg8(&px, width, height, stride),
        _ => ImageView::rgba8(&px, width, height, stride),
    }
    .unwrap();
    let peak = Settings::new(format, effort).with_bc45_fit(Bc45Fit::PeakBounded);
    let out = encode(view, &peak).unwrap();
    assert_eq!(out, encode(view, &peak).unwrap());
    let reference = encode(
        view,
        &Settings::new(format, Effort::Balanced).with_bc45_fit(Bc45Fit::PeakBounded),
    )
    .unwrap();
    // Plain presets must stay encodable on the same input.
    encode(view, &Settings::new(format, effort)).unwrap();
    let per = if format == Format::Bc4 { 8 } else { 16 };
    let bw = (width as usize).div_ceil(4);
    for (blk, (mine, base)) in out
        .chunks_exact(per)
        .zip(reference.chunks_exact(per))
        .enumerate()
    {
        let (bx, by) = (blk % bw, blk / bw);
        for ch in 0..per / 8 {
            let (m, r) = (
                decode_bc4(&mine[ch * 8..ch * 8 + 8]),
                decode_bc4(&base[ch * 8..ch * 8 + 8]),
            );
            let (mut em, mut er) = (0, 0);
            for i in 0..16 {
                let x = (bx * 4 + i % 4).min(width as usize - 1);
                let y = (by * 4 + i / 4).min(height as usize - 1);
                let v = i32::from(px[y * stride + x * channels + ch]);
                em = em.max((m[i] - v).abs());
                er = er.max((r[i] - v).abs());
            }
            assert!(em <= er, "peak {em} > range-fit {er} block {blk} ch {ch}");
        }
    }
});
