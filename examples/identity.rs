use texture_compressor::{encode, AlphaPolicy, Effort, Format, ImageView, Settings};
fn main() {
    let mut state = 1234567u32;
    let mut pixels = [0u8; 13 * 7 * 4];
    for p in &mut pixels {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *p = (state >> 24) as u8;
    }
    for format in [
        Format::Bc1,
        Format::Bc3,
        Format::Bc4,
        Format::Bc5,
        Format::Bc7,
    ] {
        for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
            let settings = Settings {
                format,
                effort,
                alpha: AlphaPolicy::Opaque,
            };
            let data = encode(ImageView::rgba8(&pixels, 13, 7, 52).unwrap(), &settings).unwrap();
            for b in data {
                print!("{b:02x}");
            }
            println!();
        }
    }
    // Exercise exact constants, two-colour blocks, and transparent selectors.
    for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
        let mut block = [0u8; 64];
        for (i, pixel) in block.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            pixel.copy_from_slice(if i % 2 == 0 {
                &[255, 0, 0, 255]
            } else {
                &[0, 0, 255, 0]
            });
        }
        for alpha in [
            AlphaPolicy::Opaque,
            AlphaPolicy::PunchThrough { threshold: 128 },
        ] {
            let settings = Settings {
                format: Format::Bc1,
                effort,
                alpha,
            };
            let data = encode(ImageView::rgba8(&block, 4, 4, 16).unwrap(), &settings).unwrap();
            for byte in data {
                print!("{byte:02x}");
            }
            println!();
        }
        for value in [0u8, 1, 127, 254, 255] {
            let block = [value; 16];
            let data = encode(
                ImageView::r8(&block, 4, 4, 4).unwrap(),
                &Settings::new(Format::Bc4, effort),
            )
            .unwrap();
            for byte in data {
                print!("{byte:02x}");
            }
            println!();
        }
    }
    // BC7 opaque and alpha patterns include sharp subset boundaries,
    // independent scalar gradients, and opposing colour gradients.
    for sample in 0..12u32 {
        let mut block = [0u8; 64];
        for (i, pixel) in block.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let x = (i % 4) as u32;
            let y = (i / 4) as u32;
            let edge = u32::from((x + sample % 4) % 4 < 2);
            *pixel = [
                ((x * 67 + edge * 41 + sample * 17) & 255) as u8,
                ((255 - x * 54 + y * 7 + sample * 11) & 255) as u8,
                ((y * 61 + edge * 83 + sample * 19) & 255) as u8,
                if sample % 2 == 0 {
                    255
                } else {
                    ((y * 53 + x * 17 + sample * 7) & 255) as u8
                },
            ];
        }
        for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
            let result = encode(
                ImageView::rgba8(&block, 4, 4, 16).unwrap(),
                &Settings::new(Format::Bc7, effort),
            )
            .unwrap();
            for byte in result {
                print!("{byte:02x}");
            }
            println!();
        }
    }
    // Synthetic cases selected by deterministic search, one for every BC7 mode.
    // These exercise the mode-specific packers in the native/wasm comparison.
    let bc7_modes: [(u32, [u8; 64]); 8] = [
        (
            0,
            [
                184, 157, 40, 255, 193, 166, 49, 255, 133, 21, 254, 255, 118, 6, 239, 255, 196,
                169, 52, 255, 181, 154, 37, 255, 121, 9, 242, 255, 130, 18, 251, 255, 204, 169, 68,
                255, 213, 178, 77, 255, 222, 187, 86, 255, 207, 172, 71, 255, 216, 181, 80, 255,
                201, 166, 65, 255, 210, 175, 74, 255, 219, 184, 83, 255,
            ],
        ),
        (
            1,
            [
                171, 239, 54, 255, 171, 239, 54, 255, 171, 239, 54, 255, 197, 64, 13, 255, 171,
                239, 54, 255, 171, 239, 54, 255, 171, 239, 54, 255, 248, 11, 60, 255, 171, 239, 54,
                255, 171, 239, 54, 255, 171, 239, 54, 255, 197, 64, 13, 255, 171, 239, 54, 255,
                171, 239, 54, 255, 171, 239, 54, 255, 248, 11, 60, 255,
            ],
        ),
        (
            2,
            [
                190, 39, 180, 255, 129, 99, 179, 255, 30, 140, 242, 255, 188, 37, 178, 255, 127,
                97, 177, 255, 28, 138, 240, 255, 202, 51, 192, 255, 125, 95, 175, 255, 26, 136,
                238, 255, 200, 49, 190, 255, 123, 93, 173, 255, 24, 134, 236, 255, 198, 47, 188,
                255, 121, 91, 171, 255, 38, 148, 250, 255, 196, 45, 186, 255,
            ],
        ),
        (
            3,
            [
                153, 1, 189, 255, 153, 1, 189, 255, 17, 143, 203, 255, 17, 143, 203, 255, 153, 1,
                189, 255, 153, 1, 189, 255, 17, 143, 203, 255, 17, 143, 203, 255, 153, 1, 189, 255,
                132, 153, 57, 255, 132, 153, 57, 255, 17, 143, 203, 255, 132, 153, 57, 255, 132,
                153, 57, 255, 132, 153, 57, 255, 132, 153, 57, 255,
            ],
        ),
        (
            4,
            [
                67, 223, 202, 16, 67, 223, 202, 16, 67, 223, 202, 16, 67, 223, 202, 16, 67, 223,
                202, 16, 211, 33, 21, 240, 211, 33, 21, 240, 67, 223, 202, 16, 67, 223, 202, 16,
                211, 33, 21, 240, 211, 33, 21, 240, 67, 223, 202, 16, 67, 223, 202, 16, 67, 223,
                202, 16, 67, 223, 202, 16, 67, 223, 202, 16,
            ],
        ),
        (
            5,
            [
                101, 8, 224, 41, 77, 176, 218, 233, 118, 221, 5, 132, 196, 186, 62, 8, 88, 217,
                165, 229, 86, 214, 41, 150, 99, 225, 96, 147, 204, 143, 191, 163, 147, 170, 131,
                193, 207, 193, 231, 38, 20, 196, 114, 44, 219, 235, 109, 35, 27, 173, 236, 69, 129,
                143, 223, 98, 15, 0, 22, 243, 247, 164, 71, 177,
            ],
        ),
        (
            6,
            [
                205, 33, 167, 16, 238, 207, 188, 240, 238, 207, 188, 240, 205, 33, 167, 16, 205,
                33, 167, 16, 205, 33, 167, 16, 238, 207, 188, 240, 238, 207, 188, 240, 205, 33,
                167, 16, 205, 33, 167, 16, 238, 207, 188, 240, 238, 207, 188, 240, 238, 207, 188,
                240, 205, 33, 167, 16, 205, 33, 167, 16, 238, 207, 188, 240,
            ],
        ),
        (
            7,
            [
                160, 105, 9, 63, 79, 88, 142, 155, 63, 179, 2, 94, 237, 16, 185, 28, 122, 89, 117,
                125, 185, 104, 197, 144, 166, 25, 154, 65, 170, 168, 157, 99, 112, 146, 41, 34, 49,
                69, 239, 217, 182, 92, 157, 149, 124, 59, 27, 192, 178, 49, 61, 200, 6, 136, 227,
                179, 115, 37, 255, 242, 127, 161, 244, 241,
            ],
        ),
    ];
    for (mode, block) in bc7_modes {
        let result = encode(
            ImageView::rgba8(&block, 4, 4, 16).unwrap(),
            &Settings::new(Format::Bc7, Effort::Fast),
        )
        .unwrap();
        assert_eq!(result[0].trailing_zeros(), mode);
        for byte in result {
            print!("{byte:02x}");
        }
        println!();
    }

    // A larger deterministic BC7 fixture mixes gradients, partitions,
    // constants, grayscale, and independent alpha within the same image.
    let mut fixture = [0u8; 64 * 64 * 4];
    for y in 0..64usize {
        for x in 0..64usize {
            let block = (y / 4) * 16 + x / 4;
            let lx = x % 4;
            let ly = y % 4;
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (state >> 24) as u8;
            let pixel = match block % 8 {
                0 => [noise, noise.wrapping_mul(17), noise.wrapping_add(93), 255],
                1 => [(lx * 63) as u8, (255 - lx * 57) as u8, (ly * 61) as u8, 255],
                2 => {
                    if lx < 2 {
                        [230, 20, 60, 255]
                    } else {
                        [20, 180, 220, 255]
                    }
                }
                3 => {
                    let v = ((block * 13 + lx * 3 + ly) & 255) as u8;
                    [v, v, v, 255]
                }
                4 => [
                    noise,
                    (lx * 71) as u8,
                    (ly * 71) as u8,
                    (lx * 49 + ly * 17) as u8,
                ],
                5 => [
                    ((block * 29) & 255) as u8,
                    ((block * 17) & 255) as u8,
                    33,
                    127,
                ],
                6 => {
                    let v = ((block + lx * 31 + ly * 17) & 255) as u8;
                    [v, 255 - v, v / 2, 255]
                }
                _ => [
                    noise,
                    noise / 2,
                    255 - noise,
                    if (lx + ly).is_multiple_of(2) { 0 } else { 255 },
                ],
            };
            fixture[(y * 64 + x) * 4..][..4].copy_from_slice(&pixel);
        }
    }
    for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
        let result = encode(
            ImageView::rgba8(&fixture, 64, 64, 256).unwrap(),
            &Settings::new(Format::Bc7, effort),
        )
        .unwrap();
        for byte in result {
            print!("{byte:02x}");
        }
        println!();
    }
}
