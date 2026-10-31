use texture_compressor::{
    decode_bc6h, encode, encode_hdr, encode_hdr_into, Effort, Error, Format, HdrImageView,
    ImageView, Settings,
};
const MODES: [u8; 14] = [0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15];
#[test]
fn decoder_matches_independent_oracle_all_modes_partitions() {
    let mut state = 0x7139a52du32;
    for signed in [false, true] {
        for mode in MODES {
            for partition in 0..32u128 {
                for sample in 0..32 {
                    let mut block = [0; 16];
                    for b in &mut block {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        *b = (state >> 24) as u8;
                    }
                    let mut bits = u128::from_le_bytes(block);
                    let mask = if mode < 2 { 3 } else { 31 };
                    bits = (bits & !mask) | u128::from(mode);
                    if ![3, 7, 11, 15].contains(&mode) {
                        bits = (bits & !(31 << 77)) | (partition << 77);
                    }
                    block = bits.to_le_bytes();
                    let format = if signed {
                        Format::Bc6hSf16
                    } else {
                        Format::Bc6hUf16
                    };
                    let actual = decode_bc6h(&block, format).unwrap();
                    let mut expected = [0; 48];
                    bcdec_rs::bc6h_half(&block, &mut expected, 12, signed);
                    // bcdec_rs scales before extracting the sign, losing -0.
                    // Khronos section 20.2 explicitly restores the original sign.
                    // Compare zeros numerically here; test the normative sign below.
                    for (&a, b) in actual.as_flattened().iter().zip(&mut expected) {
                        if a == 0x8000 && *b == 0 {
                            *b = 0x8000;
                        }
                    }
                    assert_eq!(
                        actual.as_flattened(),
                        expected,
                        "signed={signed} mode={mode} partition={partition} sample={sample}"
                    );
                }
            }
        }
    }
}
#[test]
fn reserved_modes_black_and_signed_infinity() {
    for mode in [19, 23, 27, 31] {
        let mut block = [255; 16];
        block[0] = mode;
        for f in [Format::Bc6hUf16, Format::Bc6hSf16] {
            assert_eq!(decode_bc6h(&block, f).unwrap(), [[0; 3]; 16]);
        }
    }
    // In mode 15 the base endpoint -32768 is exactly negative infinity.
    let bits = 15u128 | (1 << 39) | (1 << 49) | (1 << 59);
    assert_eq!(
        decode_bc6h(&bits.to_le_bytes(), Format::Bc6hSf16).unwrap()[0],
        [0xfc00; 3]
    );
}
#[test]
fn validation_and_tail() {
    assert!(matches!(
        HdrImageView::rgb_f32(&[], 0, 1, 3),
        Err(Error::EmptyImage)
    ));
    assert!(matches!(
        HdrImageView::rgb_f32(&[0.; 3], 1, 1, 2),
        Err(Error::InvalidStride)
    ));
    assert!(matches!(
        HdrImageView::rgb_f32(&[0.; 2], 1, 1, 3),
        Err(Error::ShortInput)
    ));
    assert!(matches!(
        HdrImageView::rgb_f32(&[], 1, 2, usize::MAX),
        Err(Error::Overflow)
    ));
    let data = [1.; 3];
    let image = HdrImageView::rgb_f32(&data, 1, 1, 3).unwrap();
    let settings = Settings::new(Format::Bc6hUf16, Effort::Fast);
    assert!(matches!(
        encode_hdr_into(image, &settings, &mut [0; 15]),
        Err(Error::ShortOutput)
    ));
    let mut out = [77; 20];
    encode_hdr_into(image, &settings, &mut out).unwrap();
    assert_eq!(out[16..], [77; 4]);
    assert!(matches!(
        encode_hdr(image, &Settings::new(Format::Bc7, Effort::Fast)),
        Err(Error::UnsupportedPolicy)
    ));
    assert!(matches!(
        encode(ImageView::rgba8(&[0; 4], 1, 1, 4).unwrap(), &settings),
        Err(Error::UnsupportedPolicy)
    ));
}
#[test]
fn normalization_alpha_half_and_edges() {
    for format in [Format::Bc6hUf16, Format::Bc6hSf16] {
        for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
            let s = Settings::new(format, effort);
            let input = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
            let expected = [
                0.,
                65504.,
                if format == Format::Bc6hSf16 {
                    -65504.
                } else {
                    0.
                },
            ];
            assert_eq!(
                encode_hdr(HdrImageView::rgba_f32(&input, 1, 1, 4).unwrap(), &s).unwrap(),
                encode_hdr(HdrImageView::rgb_f32(&expected, 1, 1, 3).unwrap(), &s).unwrap()
            );
            for value in [0u16, 1, 0x3c00, 0x7bff, 0x8001, 0xbc00] {
                let half = [value; 3];
                let mut rgb = [0.; 3];
                let mut block = [0.; 48];
                // Oracle test helper uses the exact binary16 expansion directly.
                let mag = value & 0x7fff;
                let e = mag >> 10;
                let m = mag & 1023;
                let x = if e == 0 {
                    f32::from(m) / 16777216.
                } else {
                    f32::from_bits(((u32::from(e) + 112) << 23) | (u32::from(m) << 13))
                };
                rgb.fill(if value & 0x8000 != 0 { -x } else { x });
                for p in block.as_chunks_mut::<3>().0.iter_mut() {
                    p.copy_from_slice(&rgb);
                }
                let a = encode_hdr(HdrImageView::rgb_f16(&half, 1, 1, 3).unwrap(), &s).unwrap();
                let b = encode_hdr(HdrImageView::rgb_f32(&block, 4, 4, 12).unwrap(), &s).unwrap();
                assert_eq!(a, b);
                let rgba = [value, value, value, 0x7fff];
                assert_eq!(
                    a,
                    encode_hdr(HdrImageView::rgba_f16(&rgba, 1, 1, 4).unwrap(), &s).unwrap()
                );
            }
        }
    }
}

#[test]
fn gradients_are_deterministic_and_effort_is_monotonic() {
    for format in [Format::Bc6hUf16, Format::Bc6hSf16] {
        let mut data = [0.; 48];
        for (i, p) in data.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let x = (i % 4) as f32;
            let y = (i / 4) as f32;
            p.copy_from_slice(&[x * 30. - 15., y / 8., (3. - x) * 20.]);
        }
        let mut previous = f64::INFINITY;
        for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
            let s = Settings::new(format, effort);
            let image = HdrImageView::rgb_f32(&data, 4, 4, 12).unwrap();
            let encoded = encode_hdr(image, &s).unwrap();
            assert_eq!(encoded, encode_hdr(image, &s).unwrap());
            let decoded = decode_bc6h(encoded[..16].try_into().unwrap(), format).unwrap();
            let mut floats = [0.; 48];
            bcdec_rs::bc6h_float(&encoded, &mut floats, 12, format == Format::Bc6hSf16);
            let mut error = 0.;
            for (a, b) in data.iter().zip(floats) {
                let a = if format == Format::Bc6hUf16 {
                    a.max(0.)
                } else {
                    *a
                };
                let t = |v: f32| f64::from(v.signum()) * (1. + 16. * f64::from(v.abs())).log2();
                error += (t(a) - t(b)).powi(2);
            }
            assert!(error <= previous + 1e-12);
            previous = error;
            assert!(decoded.iter().flatten().all(|h| h & 0x7c00 != 0x7c00));
        }
    }
}
