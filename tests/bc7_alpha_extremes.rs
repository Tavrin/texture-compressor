use texture_compressor::{encode, AlphaPolicy, Effort, Format, ImageView, Settings};

#[test]
fn preserves_zero_alpha_without_erasing_soft_alpha_or_hidden_colour() {
    for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
        for kind in 0..4 {
            let rgba: Vec<u8> = (0..64 * 64)
                .flat_map(|i| {
                    let alpha = match kind {
                        0 => 0,
                        1 => 255,
                        2 => {
                            if i % 3 == 0 {
                                0
                            } else {
                                255
                            }
                        }
                        _ => [0, 1, 4, 31, 100, 170, 254, 255][i % 8],
                    };
                    [
                        (i * 37) as u8,
                        (i * 71 + 19) as u8,
                        (i * 13 + 23) as u8,
                        alpha,
                    ]
                })
                .collect();
            let settings =
                Settings::new(Format::Bc7, effort).with_alpha(AlphaPolicy::PreserveZeroAlpha);
            let blocks = encode(ImageView::rgba8(&rgba, 64, 64, 256).unwrap(), &settings).unwrap();
            for (index, block) in blocks.chunks_exact(16).enumerate() {
                let mut decoded = [0; 64];
                bcdec_rs::bc7(block, &mut decoded, 16);
                for pixel in 0..16 {
                    let x = index % 16 * 4 + pixel % 4;
                    let y = index / 16 * 4 + pixel / 4;
                    let expected = rgba[(y * 64 + x) * 4 + 3];
                    let actual = decoded[pixel * 4 + 3];
                    if expected == 0 || kind == 1 {
                        assert_eq!(actual, expected, "{effort:?}, kind {kind}, {x},{y}");
                    }
                }
                if kind == 0 {
                    assert!(decoded
                        .chunks_exact(4)
                        .any(|p| p[..3].iter().any(|&c| c > 16)));
                }
            }
            if kind == 3 {
                let mut decoded = [0; 64];
                bcdec_rs::bc7(&blocks[..16], &mut decoded, 16);
                assert!(decoded.chunks_exact(4).any(|p| (1..255).contains(&p[3])));
            }
            let first: Vec<_> = (0..4)
                .flat_map(|y| rgba[y * 256..y * 256 + 16].iter().copied())
                .collect();
            assert_eq!(
                encode(ImageView::rgba8(&first, 4, 4, 16).unwrap(), &settings).unwrap(),
                blocks[..16]
            );
        }
    }
}

#[test]
fn blocks_without_zero_alpha_use_the_ordinary_fit() {
    let rgba = (0..16)
        .flat_map(|i| {
            [
                (i * 37) as u8,
                (i * 71 + 19) as u8,
                (i * 13 + 23) as u8,
                (i * 17).max(1) as u8,
            ]
        })
        .collect::<Vec<_>>();
    for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
        let image = ImageView::rgba8(&rgba, 4, 4, 16).unwrap();
        let ordinary = Settings::new(Format::Bc7, effort);
        let constrained = ordinary.with_alpha(AlphaPolicy::PreserveZeroAlpha);
        assert_eq!(
            encode(image, &ordinary).unwrap(),
            encode(image, &constrained).unwrap()
        );
    }
}

#[test]
fn zero_alpha_does_not_remove_mode6_gradient_precision() {
    let rgba = (0..16)
        .flat_map(|i| {
            [
                (60 + i * i / 2) as u8,
                (80 + i * 4) as u8,
                (140 - i * 3) as u8,
                (i * 17) as u8,
            ]
        })
        .collect::<Vec<_>>();
    for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
        let settings =
            Settings::new(Format::Bc7, effort).with_alpha(AlphaPolicy::PreserveZeroAlpha);
        let encoded = encode(ImageView::rgba8(&rgba, 4, 4, 16).unwrap(), &settings).unwrap();
        let mut decoded = [0; 64];
        bcdec_rs::bc7(&encoded, &mut decoded, 16);
        assert_eq!(decoded[3], 0);
        let alpha_sse = rgba
            .chunks_exact(4)
            .zip(decoded.chunks_exact(4))
            .map(|(s, d)| u32::from(s[3].abs_diff(d[3])).pow(2))
            .sum::<u32>();
        assert!(alpha_sse < 512, "{effort:?}: alpha SSE {alpha_sse}");
    }
}
