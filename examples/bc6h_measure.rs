#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::{fs, time::Instant};
    use texture_compressor::{decode_bc6h, encode_hdr, Effort, Format, HdrImageView, Settings};
    let root = std::env::args()
        .nth(1)
        .ok_or("pass corpus image directory")?;
    let mut files = fs::read_dir(root)?.collect::<Result<Vec<_>, _>>()?;
    files.sort_by_key(|f| f.path());
    let mut source = Vec::new();
    let mut intel_source = Vec::new();
    for f in files {
        if f.path().extension().is_some_and(|x| x == "rgb32") {
            let data = fs::read(f.path())?;
            let rgb: Vec<_> = data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            assert_eq!(rgb.len(), 64 * 64 * 3);
            let mut rgba = Vec::new();
            for p in rgb.as_chunks::<3>().0.iter() {
                for &x in p {
                    rgba.extend_from_slice(&to_half(x).to_le_bytes());
                }
                rgba.extend_from_slice(&0x3c00u16.to_le_bytes());
            }
            source.push(rgb);
            intel_source.push(rgba);
        }
    }
    assert_eq!(source.len(), 24);
    println!(
        "preset,our_log_rmse,intel_log_rmse,our_exposure_psnr,intel_exposure_psnr,our_ms,intel_ms"
    );
    for (effort, ispc) in [
        (Effort::Fast, intel_tex_2::bc6h::fast_settings()),
        (Effort::Balanced, intel_tex_2::bc6h::basic_settings()),
        (Effort::Quality, intel_tex_2::bc6h::slow_settings()),
    ] {
        if let Ok(preset) = std::env::var("BC6H_PRESET") {
            if !format!("{effort:?}").eq_ignore_ascii_case(&preset) {
                continue;
            }
        }
        let settings = Settings::new(Format::Bc6hUf16, effort);
        if std::env::var_os("BC6H_PROFILE").is_some() {
            for rgb in &source {
                std::hint::black_box(encode_hdr(
                    HdrImageView::rgb_f32(rgb, 64, 64, 192)?,
                    &settings,
                )?);
            }
            continue;
        }
        let mut times = [Vec::new(), Vec::new()];
        let mut outputs = [Vec::new(), Vec::new()];
        // Untimed warmup for each implementation, on the same first image.
        std::hint::black_box(encode_hdr(
            HdrImageView::rgb_f32(&source[0], 64, 64, 192)?,
            &settings,
        )?);
        std::hint::black_box(intel_tex_2::bc6h::compress_blocks(
            &ispc,
            &intel_tex_2::RgbaSurface {
                data: &intel_source[0],
                width: 64,
                height: 64,
                stride: 512,
            },
        ));
        for round in 0..3 {
            for which in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                let mut out = Vec::new();
                for (rgb, rgba) in source.iter().zip(&intel_source) {
                    out.push(if which == 0 {
                        encode_hdr(HdrImageView::rgb_f32(rgb, 64, 64, 192)?, &settings)?
                    } else {
                        intel_tex_2::bc6h::compress_blocks(
                            &ispc,
                            &intel_tex_2::RgbaSurface {
                                data: rgba,
                                width: 64,
                                height: 64,
                                stride: 512,
                            },
                        )
                    });
                }
                let milliseconds = start.elapsed().as_secs_f64() * 1000.;
                eprintln!("{effort:?} round={round} encoder={which} ms={milliseconds:.6}");
                times[which].push(milliseconds);
                outputs[which] = out;
            }
        }
        let mut metrics = [(0., 0.); 2];
        for which in 0..2 {
            let mut log_error = 0.;
            let mut exposure_error = 0.;
            let mut count = 0.;
            for (rgb, out) in source.iter().zip(&outputs[which]) {
                for (b, block) in out.as_chunks::<16>().0.iter().enumerate() {
                    let decoded = decode_bc6h(block, Format::Bc6hUf16)?;
                    for (i, p) in decoded.iter().enumerate() {
                        let x = b % 16 * 4 + i % 4;
                        let y = b / 16 * 4 + i / 4;
                        for c in 0..3 {
                            let a = f64::from(rgb[(y * 64 + x) * 3 + c]);
                            let d = f64::from(from_half(p[c]));
                            for exposure in [-4, -2, 0, 2, 4] {
                                let scale = 2f64.powi(exposure);
                                log_error +=
                                    ((1. + a * scale).log2() - (1. + d * scale).log2()).powi(2);
                                let view = |v: f64| {
                                    let t = v * scale;
                                    t / (1. + t)
                                };
                                exposure_error += (view(a) - view(d)).powi(2);
                            }
                            count += 1.;
                        }
                    }
                }
            }
            metrics[which] = (
                (log_error / (count * 5.)).sqrt(),
                -10. * (exposure_error / (count * 5.)).log10(),
            );
        }
        for t in &mut times {
            t.sort_by(f64::total_cmp);
        }
        println!(
            "{effort:?},{:.8},{:.8},{:.4},{:.4},{:.3},{:.3}",
            metrics[0].0, metrics[1].0, metrics[0].1, metrics[1].1, times[0][1], times[1][1]
        );
    }
    Ok(())
}
#[cfg(not(target_arch = "wasm32"))]
fn from_half(h: u16) -> f32 {
    let e = u32::from(h >> 10);
    let m = u32::from(h & 1023);
    if e == 0 {
        m as f32 / 16777216.
    } else {
        f32::from_bits(((e + 112) << 23) | (m << 13))
    }
}
#[cfg(not(target_arch = "wasm32"))]
fn to_half(x: f32) -> u16 {
    let mut lo = 0u16;
    let mut hi = 31743u16;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if from_half(mid) < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo > 0 && x - from_half(lo - 1) <= from_half(lo) - x {
        lo - 1
    } else {
        lo
    }
}
#[cfg(target_arch = "wasm32")]
fn main() {}
