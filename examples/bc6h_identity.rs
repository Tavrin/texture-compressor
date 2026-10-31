use std::{
    fs,
    io::{self, Write},
};
use texture_compressor::{decode_bc6h, encode_hdr, Effort, Format, HdrImageView, Settings};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    let mut images = Vec::new();
    if let Some(root) = std::env::args().nth(1) {
        let mut files = fs::read_dir(root)?.collect::<Result<Vec<_>, _>>()?;
        files.sort_by_key(|f| f.path());
        for f in files {
            if f.path().extension().is_some_and(|x| x == "rgb32") {
                let bytes = fs::read(f.path())?;
                images.push((
                    64,
                    64,
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect::<Vec<_>>(),
                ));
            }
        }
    }
    let values = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -65504.,
        -1.,
        -0.,
        0.,
        f32::from_bits(1),
        1. / 16777216.,
        0.5,
        1.,
        65504.,
    ];
    for value in values {
        images.push((1, 1, vec![value; 3]));
    }
    images.push((7, 5, (0..105).map(|i| (i as f32 - 52.) / 3.).collect()));
    for format in [Format::Bc6hUf16, Format::Bc6hSf16] {
        for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
            for (w, h, rgb) in &images {
                let data = encode_hdr(
                    HdrImageView::rgb_f32(rgb, *w, *h, *w as usize * 3)?,
                    &Settings::new(format, effort),
                )?;
                stdout.write_all(&data)?;
                for b in data.as_chunks::<16>().0 {
                    for p in decode_bc6h(b, format)? {
                        for h in p {
                            stdout.write_all(&h.to_le_bytes())?;
                        }
                    }
                }
            }
        }
    }
    for format in [Format::Bc6hUf16, Format::Bc6hSf16] {
        for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
            for value in [
                0u16, 1, 0x3c00, 0x7bff, 0x8000, 0x8001, 0xfbff, 0x7c00, 0xfc00, 0x7fff,
            ] {
                let data = [value, value, value, 0x7fff];
                stdout.write_all(&encode_hdr(
                    HdrImageView::rgba_f16(&data, 1, 1, 4)?,
                    &Settings::new(format, effort),
                )?)?;
            }
        }
    }
    // Arbitrary bitstreams exercise every mode, partition, and signed transform.
    let mut state = 0x8312795bu32;
    for format in [Format::Bc6hUf16, Format::Bc6hSf16] {
        for mode in [0u8, 1, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15] {
            for part in 0..32u128 {
                let mut block = [0; 16];
                for b in &mut block {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    *b = (state >> 24) as u8;
                }
                let mut bits = u128::from_le_bytes(block);
                bits = (bits & !if mode < 2 { 3 } else { 31 }) | u128::from(mode);
                if ![3, 7, 11, 15].contains(&mode) {
                    bits = (bits & !(31 << 77)) | (part << 77);
                }
                for p in decode_bc6h(&bits.to_le_bytes(), format)? {
                    for h in p {
                        stdout.write_all(&h.to_le_bytes())?;
                    }
                }
            }
        }
    }
    Ok(())
}
