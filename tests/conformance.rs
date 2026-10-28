use texture_compressor::{encode, AlphaPolicy, Effort, Format, ImageView, Settings};

// Test-only independent decoder. BC1 uses the decoder's integer RGB565
// reconstruction; BC4 uses the UNORM fixed-point interpolation weights.
fn bc1(b: &[u8]) -> [[u8; 4]; 16] {
    let a = u16::from_le_bytes([b[0], b[1]]);
    let z = u16::from_le_bytes([b[2], b[3]]);
    let mut p = [[0u8; 4]; 4];
    p[0][3] = 255;
    p[1][3] = 255;
    p[2][3] = 255;
    if a > z {
        p[3][3] = 255;
    }
    for c in 0..3 {
        let shift = [11, 5, 0][c];
        let mask = [31, 63, 31][c];
        let x = u32::from(a >> shift & mask);
        let y = u32::from(z >> shift & mask);
        p[0][c] = if c == 1 {
            ((x * 259 + 33) >> 6) as u8
        } else {
            ((x * 527 + 23) >> 6) as u8
        };
        p[1][c] = if c == 1 {
            ((y * 259 + 33) >> 6) as u8
        } else {
            ((y * 527 + 23) >> 6) as u8
        };
        p[2][c] = if a > z {
            if c == 1 {
                (((2 * x + y) * 2763 + 1039) >> 11) as u8
            } else {
                (((2 * x + y) * 351 + 61) >> 7) as u8
            }
        } else if c == 1 {
            (((x + y) * 4145 + 1019) >> 11) as u8
        } else {
            (((x + y) * 1053 + 125) >> 8) as u8
        };
        if a > z {
            p[3][c] = if c == 1 {
                (((x + 2 * y) * 2763 + 1039) >> 11) as u8
            } else {
                (((x + 2 * y) * 351 + 61) >> 7) as u8
            };
        }
    }
    let bits = u32::from_le_bytes(b[4..8].try_into().unwrap());
    let mut out = [[0; 4]; 16];
    for i in 0..16 {
        out[i] = p[((bits >> (i * 2)) & 3) as usize];
    }
    out
}
fn bc4(b: &[u8]) -> [u8; 16] {
    let a = i32::from(b[0]);
    let z = i32::from(b[1]);
    let mut p = [a, z, 0, 0, 0, 0, 0, 0];
    if a > z {
        let w = [9363, 18724, 28086, 37450, 46812, 56173];
        for i in 0..6 {
            p[i + 2] = (w[5 - i] * a + w[i] * z + 32768) >> 16;
        }
    } else {
        let w = [13107, 26215, 39321, 52429];
        for i in 0..4 {
            p[i + 2] = (w[3 - i] * a + w[i] * z + 32768) >> 16;
        }
        p[7] = 255;
    }
    let bits = u64::from_le_bytes(b.try_into().unwrap()) >> 16;
    let mut out = [0; 16];
    for i in 0..16 {
        out[i] = p[((bits >> (3 * i)) & 7) as usize] as u8;
    }
    out
}
#[test]
fn known_vectors() {
    assert_eq!(bc1(&[0x00, 0xf8, 0, 0, 0, 0, 0, 0]), [[255, 0, 0, 255]; 16]);
    assert_eq!(
        bc1(&[0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]),
        [[0, 0, 0, 0]; 16]
    );
    assert_eq!(bc4(&[255, 0, 0, 0, 0, 0, 0, 0]), [255; 16]);
    assert_eq!(
        bc4(&[0, 255, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]),
        [255; 16]
    );
    assert_eq!(bc4(&[0, 0, 0, 0, 0, 0, 0, 0]), [0; 16]);
}
#[test]
fn random_npot_and_determinism() {
    let mut state = 1234567u32;
    for &(w, h) in &[(1, 1), (3, 7), (8, 8), (13, 5)] {
        let mut data = vec![0u8; w * h * 4];
        for v in &mut data {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (state >> 24) as u8;
        }
        for f in [Format::Bc1, Format::Bc4] {
            for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
                for alpha in [
                    AlphaPolicy::Opaque,
                    AlphaPolicy::PunchThrough { threshold: 128 },
                ] {
                    if f == Format::Bc4 && alpha != AlphaPolicy::Opaque {
                        continue;
                    }
                    let settings = Settings {
                        format: f,
                        effort,
                        alpha,
                    };
                    let encode_full = || {
                        encode(
                            ImageView::rgba8(&data, w as u32, h as u32, w * 4).unwrap(),
                            &settings,
                        )
                        .unwrap()
                    };
                    let out = encode_full();
                    assert_eq!(out, encode_full());
                    assert_eq!(out, encode_full());
                    for block in out.as_chunks::<8>().0 {
                        if f == Format::Bc1 {
                            let a = u16::from_le_bytes([block[0], block[1]]);
                            let b = u16::from_le_bytes([block[2], block[3]]);
                            if alpha == AlphaPolicy::Opaque {
                                assert!(a > b)
                            } else {
                                assert!(a <= b)
                            }
                            let d = bc1(block);
                            if alpha == AlphaPolicy::Opaque {
                                assert!(d.iter().all(|p| p[3] == 255));
                            }
                        } else {
                            let _ = bc4(block);
                        }
                    }
                    let mut error = 0u64;
                    let mut zero_error = 0u64;
                    for y in 0..h {
                        for x in 0..w {
                            let block_index = (y / 4 * w.div_ceil(4) + x / 4) * 8;
                            let block = &out[block_index..block_index + 8];
                            let pixel = (y % 4) * 4 + x % 4;
                            let src = &data[(y * w + x) * 4..][..4];
                            if f == Format::Bc1 {
                                let decoded = bc1(block)[pixel];
                                if let AlphaPolicy::PunchThrough { threshold } = alpha {
                                    assert_eq!(decoded[3] == 0, src[3] < threshold);
                                }
                                if decoded[3] != 0 {
                                    for c in 0..3 {
                                        let delta = i32::from(src[c]) - i32::from(decoded[c]);
                                        error += (delta * delta) as u64;
                                        zero_error += u64::from(src[c]) * u64::from(src[c]);
                                    }
                                }
                            } else {
                                let decoded = bc4(block)[pixel];
                                let delta = i32::from(src[0]) - i32::from(decoded);
                                error += (delta * delta) as u64;
                                zero_error += u64::from(src[0]) * u64::from(src[0]);
                            }
                        }
                    }
                    assert!(error <= zero_error);
                    let rows = h.div_ceil(4);
                    let parts = std::thread::scope(|scope| {
                        (0..rows)
                            .map(|r| {
                                let data = &data;
                                scope.spawn(move || {
                                    let y = r * 4;
                                    let height = (h - y).min(4);
                                    encode(
                                        ImageView::rgba8(
                                            &data[y * w * 4..],
                                            w as u32,
                                            height as u32,
                                            w * 4,
                                        )
                                        .unwrap(),
                                        &settings,
                                    )
                                    .unwrap()
                                })
                            })
                            .collect::<Vec<_>>()
                            .into_iter()
                            .flat_map(|t| t.join().unwrap())
                            .collect::<Vec<_>>()
                    });
                    assert_eq!(out, parts);
                }
            }
        }
    }
}
#[test]
fn constant_and_transparent_roundtrip() {
    for c in 0..=255 {
        let data = [c; 16];
        let out = encode(
            ImageView::r8(&data, 4, 4, 4).unwrap(),
            &Settings::new(Format::Bc4, Effort::Quality),
        )
        .unwrap();
        assert_eq!(bc4(&out), data);
    }
    let mut red = [0u8; 64];
    for pixel in red.as_chunks_mut::<4>().0 {
        *pixel = [255, 0, 0, 255];
    }
    let encoded = encode(
        ImageView::rgba8(&red, 4, 4, 16).unwrap(),
        &Settings::new(Format::Bc1, Effort::Quality),
    )
    .unwrap();
    assert_eq!(bc1(&encoded), [[255, 0, 0, 255]; 16]);
    let data = [0u8; 64];
    let settings = Settings {
        format: Format::Bc1,
        effort: Effort::Quality,
        alpha: AlphaPolicy::PunchThrough { threshold: 128 },
    };
    let out = encode(ImageView::rgba8(&data, 4, 4, 16).unwrap(), &settings).unwrap();
    assert_eq!(bc1(&out), [[0, 0, 0, 0]; 16]);
}
#[test]
fn validation() {
    assert!(ImageView::rgba8(&[0; 4], 2, 1, 8).is_err());
    assert!(ImageView::rgba8(&[0; 4], 1, 1, 3).is_err());
    assert!(texture_compressor::encoded_len(Format::Bc1, 0, 1).is_err());
    assert!(texture_compressor::encoded_len(Format::Bc1, u32::MAX, u32::MAX).is_err());
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_matches_serial_across_shapes_and_pools() {
    use texture_compressor::{
        encode, encode_into_parallel, AlphaPolicy, Effort, Format, ImageView, Settings,
    };
    for (w, h) in [(1, 1), (5, 7), (32, 33), (67, 19)] {
        let stride = w * 4 + 7;
        let mut data = vec![0; stride * h];
        for (i, x) in data.iter_mut().enumerate() {
            *x = (i.wrapping_mul(37).wrapping_add(i / 7) & 255) as u8;
        }
        let image = ImageView::rgba8(&data, w as u32, h as u32, stride).unwrap();
        for format in [
            Format::Bc1,
            Format::Bc3,
            Format::Bc4,
            Format::Bc5,
            Format::Bc7,
        ] {
            for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
                for alpha in [
                    AlphaPolicy::Opaque,
                    AlphaPolicy::PunchThrough { threshold: 128 },
                ] {
                    if format != Format::Bc1 && alpha != AlphaPolicy::Opaque {
                        continue;
                    }
                    let settings = Settings {
                        format,
                        effort,
                        alpha,
                    };
                    let expected = encode(image, &settings).unwrap();
                    for threads in [1, 2, 4, 32] {
                        let pool = rayon::ThreadPoolBuilder::new()
                            .num_threads(threads)
                            .stack_size(512 * 1024)
                            .build()
                            .unwrap();
                        let mut actual = vec![0xa5; expected.len() + 3];
                        pool.install(|| encode_into_parallel(image, &settings, &mut actual))
                            .unwrap();
                        assert_eq!(&actual[..expected.len()], expected);
                        assert_eq!(&actual[expected.len()..], &[0xa5; 3]);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_r8_and_errors_match_serial() {
    use texture_compressor::{encode_into, encode_into_parallel, Error};
    let pixels: Vec<u8> = (0..11 * 9).map(|i| (i * 73) as u8).collect();
    let image = ImageView::r8(&pixels, 11, 9, 11).unwrap();
    let settings = Settings::new(Format::Bc4, Effort::Fast);
    let mut serial = [0u8; 72];
    let mut parallel = [0u8; 72];
    encode_into(image, &settings, &mut serial).unwrap();
    encode_into_parallel(image, &settings, &mut parallel).unwrap();
    assert_eq!(parallel, serial);
    let mut short = [0xa5; 71];
    assert_eq!(
        encode_into_parallel(image, &settings, &mut short),
        Err(Error::ShortOutput)
    );
    assert_eq!(short, [0xa5; 71]);
    let unsupported = Settings::new(Format::Bc1, Effort::Fast);
    assert_eq!(
        encode_into_parallel(image, &unsupported, &mut parallel),
        Err(Error::UnsupportedPolicy)
    );
}

#[test]
fn fast_handles_opposing_colour_gradients() {
    let row = [
        [140, 45, 49, 255],
        [115, 59, 68, 255],
        [90, 74, 87, 255],
        [66, 89, 107, 255],
    ];
    let data: Vec<u8> = row.into_iter().cycle().take(16).flatten().collect();
    let settings = Settings::new(Format::Bc1, Effort::Fast);
    let out = encode(ImageView::rgba8(&data, 4, 4, 16).unwrap(), &settings).unwrap();
    let decoded = bc1(&out);
    let error: i32 = data
        .as_chunks::<4>()
        .0
        .iter()
        .zip(decoded)
        .map(|(a, b)| {
            (0..3)
                .map(|c| {
                    let d = i32::from(a[c]) - i32::from(b[c]);
                    d * d
                })
                .sum::<i32>()
        })
        .sum();
    assert!(error < 128, "opposing gradient error: {error}");
}

#[test]
fn solid_blocks_match_p3_sampled_bytes() {
    // P3 output vectors cover exact endpoints, interpolation, ties and alpha.
    let cases = [
        (
            [0, 0, 0],
            [1, 0, 0, 0, 85, 85, 85, 85],
            [0, 0, 0, 0, 195, 48, 12, 195],
        ),
        (
            [255, 255, 255],
            [255, 255, 254, 255, 0, 0, 0, 0],
            [255, 255, 255, 255, 195, 48, 12, 195],
        ),
        (
            [1, 2, 3],
            [32, 0, 1, 0, 170, 170, 170, 170],
            [0, 0, 33, 0, 235, 186, 174, 235],
        ),
        (
            [127, 128, 129],
            [33, 128, 247, 125, 255, 255, 255, 255],
            [0, 0, 255, 255, 235, 186, 174, 235],
        ),
        (
            [17, 193, 81],
            [0, 22, 253, 21, 170, 170, 170, 170],
            [0, 0, 10, 22, 215, 117, 93, 215],
        ),
        (
            [253, 2, 254],
            [31, 248, 63, 240, 170, 170, 170, 170],
            [31, 240, 63, 248, 235, 186, 174, 235],
        ),
        (
            [55, 149, 207],
            [22, 59, 223, 55, 170, 170, 170, 170],
            [83, 1, 255, 111, 235, 186, 174, 235],
        ),
        (
            [201, 24, 117],
            [1, 200, 53, 193, 255, 255, 255, 255],
            [0, 144, 156, 249, 235, 186, 174, 235],
        ),
    ];
    for (rgb, opaque, punch) in cases {
        for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
            for (masked, expected) in [(false, opaque), (true, punch)] {
                let mut data = [0; 64];
                for (i, pixel) in data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    *pixel = [
                        rgb[0],
                        rgb[1],
                        rgb[2],
                        if masked && i % 3 == 0 { 0 } else { 255 },
                    ];
                }
                let alpha = if masked {
                    AlphaPolicy::PunchThrough { threshold: 128 }
                } else {
                    AlphaPolicy::Opaque
                };
                let settings = Settings {
                    format: Format::Bc1,
                    effort,
                    alpha,
                };
                let out = encode(ImageView::rgba8(&data, 4, 4, 16).unwrap(), &settings).unwrap();
                assert_eq!(out, expected, "rgb={rgb:?} masked={masked}");
            }
        }
    }
}

#[test]
fn bc3_bc5_decode_layout_and_rg8() {
    let mut rgba = [0u8; 64];
    let mut rg = [0u8; 32];
    for i in 0..16 {
        rgba[i * 4..i * 4 + 4].copy_from_slice(&[
            (i * 17) as u8,
            (255 - i * 13) as u8,
            77,
            (i * 11) as u8,
        ]);
        rg[i * 2..i * 2 + 2].copy_from_slice(&rgba[i * 4..i * 4 + 2]);
    }
    for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
        let image = ImageView::rgba8(&rgba, 4, 4, 16).unwrap();
        let bc3 = encode(image, &Settings::new(Format::Bc3, effort)).unwrap();
        assert_eq!(bc3.len(), 16);
        assert!(u16::from_le_bytes([bc3[8], bc3[9]]) > u16::from_le_bytes([bc3[10], bc3[11]]));
        let color = bc1(&bc3[8..]);
        let alpha = bc4(&bc3[..8]);
        assert!(color.iter().all(|p| p[3] == 255));
        assert!(alpha.iter().copied().max().unwrap() > alpha.iter().copied().min().unwrap());
        let bc5 = encode(image, &Settings::new(Format::Bc5, effort)).unwrap();
        let packed = encode(
            ImageView::rg8(&rg, 4, 4, 8).unwrap(),
            &Settings::new(Format::Bc5, effort),
        )
        .unwrap();
        assert_eq!(bc5, packed);
        for (channel, block) in [(&bc5[..8], 0), (&bc5[8..], 1)] {
            let decoded = bc4(channel);
            let error: u32 = (0..16)
                .map(|i| {
                    let d = i32::from(rgba[i * 4 + block]) - i32::from(decoded[i]);
                    (d * d) as u32
                })
                .sum();
            assert!(error < 2000);
        }
        assert_eq!(
            bc3,
            encode(image, &Settings::new(Format::Bc3, effort)).unwrap()
        );
        assert_eq!(
            bc5,
            encode(image, &Settings::new(Format::Bc5, effort)).unwrap()
        );
    }
    assert_eq!(
        texture_compressor::encoded_len(Format::Bc3, 5, 7).unwrap(),
        64
    );
    assert_eq!(
        texture_compressor::encoded_len(Format::Bc5, 5, 7).unwrap(),
        64
    );
    assert!(encode(
        ImageView::r8(&rg, 4, 4, 4).unwrap(),
        &Settings::new(Format::Bc5, Effort::Fast)
    )
    .is_err());
}

#[test]
fn bc3_bc5_known_channel_blocks_and_policies() {
    let mut rgba = [0u8; 64];
    for px in rgba.as_chunks_mut::<4>().0 {
        *px = [255, 0, 0, 127];
    }
    let image = ImageView::rgba8(&rgba, 4, 4, 16).unwrap();
    for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
        let b3 = encode(image, &Settings::new(Format::Bc3, effort)).unwrap();
        assert_eq!(bc4(&b3[..8]), [127; 16]);
        assert_eq!(bc1(&b3[8..]), [[255, 0, 0, 255]; 16]);
        let b5 = encode(image, &Settings::new(Format::Bc5, effort)).unwrap();
        assert_eq!(bc4(&b5[..8]), [255; 16]);
        assert_eq!(bc4(&b5[8..]), [0; 16]);
        for format in [Format::Bc3, Format::Bc5] {
            let settings = Settings {
                format,
                effort,
                alpha: AlphaPolicy::PunchThrough { threshold: 128 },
            };
            assert_eq!(
                encode(image, &settings),
                Err(texture_compressor::Error::UnsupportedPolicy)
            );
        }
    }
}

// Independent full-format reader: reconstructs fields in stream order.
#[path = "bc7_decoder.rs"]
mod bc7_decoder;

#[test]
fn bc7_roundtrip_and_validation() {
    let mut state = 1234u32;
    for sample in 0..64 {
        let mut data = [0u8; 64];
        for x in &mut data {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *x = if sample == 0 { 0 } else { (state >> 24) as u8 };
        }
        let image = ImageView::rgba8(&data, 4, 4, 16).unwrap();
        for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
            let settings = Settings::new(Format::Bc7, effort);
            let result = encode(image, &settings).unwrap();
            assert_eq!(result, encode(image, &settings).unwrap());
            let decoded = bc7_decoder::decode(&result);
            if sample == 0 {
                assert_eq!(decoded, [[0; 4]; 16]);
            }
            #[cfg(feature = "parallel")]
            for threads in [1, 32] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .stack_size(512 * 1024)
                    .build()
                    .unwrap();
                let actual = pool
                    .install(|| texture_compressor::encode_parallel(image, &settings))
                    .unwrap();
                assert_eq!(result, actual);
            }
        }
    }
    assert_eq!(
        texture_compressor::encoded_len(Format::Bc7, 5, 7).unwrap(),
        64
    );
    assert!(encode(
        ImageView::r8(&[0; 16], 4, 4, 4).unwrap(),
        &Settings::new(Format::Bc7, Effort::Fast)
    )
    .is_err());
}

#[test]
fn bc7_npot_edges_repeat_and_opaque_alpha() {
    let mut state = 0x1827_3a4bu32;
    for (w, h) in [(1usize, 1usize), (3, 7), (13, 5)] {
        for opaque in [false, true] {
            let mut data = vec![0u8; w * h * 4];
            for pixel in data.as_chunks_mut::<4>().0 {
                for value in pixel.iter_mut() {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    *value = (state >> 24) as u8;
                }
                if opaque {
                    pixel[3] = 255;
                }
            }
            let pw = w.div_ceil(4) * 4;
            let ph = h.div_ceil(4) * 4;
            let mut padded = vec![0u8; pw * ph * 4];
            for y in 0..ph {
                for x in 0..pw {
                    let from = ((y.min(h - 1) * w) + x.min(w - 1)) * 4;
                    let to = (y * pw + x) * 4;
                    padded[to..to + 4].copy_from_slice(&data[from..from + 4]);
                }
            }
            for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
                let settings = Settings::new(Format::Bc7, effort);
                let encode_original = || {
                    encode(
                        ImageView::rgba8(&data, w as u32, h as u32, w * 4).unwrap(),
                        &settings,
                    )
                    .unwrap()
                };
                let result = encode_original();
                assert_eq!(result, encode_original());
                assert_eq!(result, encode_original());
                assert_eq!(
                    result,
                    encode(
                        ImageView::rgba8(&padded, pw as u32, ph as u32, pw * 4).unwrap(),
                        &settings
                    )
                    .unwrap()
                );
                for block in result.as_chunks::<16>().0 {
                    assert_ne!(block[0], 0);
                    let decoded = bc7_decoder::decode(block);
                    if opaque {
                        assert!(decoded.iter().all(|p| p[3] == 255));
                    }
                }
            }
        }
    }
}
