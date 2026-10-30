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
    // Synthetic cases selected by deterministic search cover every BC7 mode.
    // Retain the earlier three-colour inputs, now selected as mode 2, and add
    // a mode 0 case to keep every packer in the native/wasm comparison.
    let bc7_modes: [(u32, Effort, [u8; 64]); 10] = [
        (
            2,
            Effort::Quality,
            [
                85, 132, 186, 255, 231, 222, 42, 255, 221, 212, 32, 255, 207, 198, 18, 255, 224,
                215, 35, 255, 79, 126, 180, 255, 80, 127, 181, 255, 65, 112, 166, 255, 201, 192,
                12, 255, 159, 46, 186, 255, 85, 132, 186, 255, 74, 121, 175, 255, 161, 48, 188,
                255, 145, 32, 172, 255, 169, 56, 196, 255, 215, 206, 26, 255,
            ],
        ),
        (
            1,
            Effort::Fast,
            [
                126, 45, 143, 255, 223, 132, 95, 255, 223, 132, 95, 255, 174, 24, 54, 255, 223,
                132, 95, 255, 126, 45, 143, 255, 126, 45, 143, 255, 126, 45, 143, 255, 126, 45,
                143, 255, 174, 24, 54, 255, 174, 24, 54, 255, 174, 24, 54, 255, 126, 45, 143, 255,
                126, 45, 143, 255, 174, 24, 54, 255, 126, 45, 143, 255,
            ],
        ),
        (
            2,
            Effort::Quality,
            [
                198, 171, 174, 255, 46, 87, 147, 255, 198, 171, 174, 255, 143, 62, 23, 255, 46, 87,
                147, 255, 199, 172, 175, 255, 48, 250, 25, 255, 45, 86, 146, 255, 200, 173, 176,
                255, 47, 88, 148, 255, 200, 173, 176, 255, 143, 62, 23, 255, 47, 88, 148, 255, 45,
                86, 146, 255, 198, 171, 174, 255, 47, 88, 148, 255,
            ],
        ),
        (
            3,
            Effort::Fast,
            [
                140, 20, 172, 255, 141, 21, 173, 255, 135, 15, 167, 255, 255, 253, 135, 255, 138,
                18, 170, 255, 140, 20, 172, 255, 251, 249, 131, 255, 249, 247, 129, 255, 142, 22,
                174, 255, 134, 14, 166, 255, 255, 254, 136, 255, 252, 250, 132, 255, 141, 21, 173,
                255, 255, 253, 135, 255, 252, 250, 132, 255, 253, 251, 133, 255,
            ],
        ),
        (
            4,
            Effort::Fast,
            [
                132, 139, 222, 178, 133, 140, 223, 179, 102, 179, 49, 18, 102, 179, 49, 18, 122,
                127, 154, 90, 131, 138, 221, 177, 103, 180, 50, 19, 122, 127, 154, 90, 121, 126,
                153, 89, 132, 139, 222, 178, 132, 139, 222, 178, 131, 138, 221, 177, 131, 138, 221,
                177, 133, 140, 223, 179, 101, 178, 48, 17, 120, 125, 152, 88,
            ],
        ),
        (
            5,
            Effort::Fast,
            [
                14, 53, 228, 255, 13, 52, 227, 255, 29, 89, 207, 255, 14, 53, 228, 255, 28, 88,
                206, 255, 14, 53, 228, 255, 13, 52, 227, 255, 29, 89, 207, 255, 14, 53, 228, 255,
                13, 52, 227, 255, 12, 51, 226, 255, 14, 53, 228, 255, 27, 87, 205, 255, 29, 89,
                207, 255, 13, 52, 227, 255, 27, 87, 205, 255,
            ],
        ),
        (
            6,
            Effort::Fast,
            [
                63, 73, 231, 255, 63, 73, 231, 255, 63, 73, 231, 255, 125, 156, 202, 255, 125, 156,
                202, 255, 63, 73, 231, 255, 125, 156, 202, 255, 63, 73, 231, 255, 125, 156, 202,
                255, 63, 73, 231, 255, 63, 73, 231, 255, 63, 73, 231, 255, 125, 156, 202, 255, 63,
                73, 231, 255, 125, 156, 202, 255, 125, 156, 202, 255,
            ],
        ),
        (
            7,
            Effort::Fast,
            [
                229, 41, 228, 106, 184, 177, 110, 151, 41, 63, 199, 20, 184, 177, 110, 151, 184,
                177, 110, 151, 148, 179, 107, 106, 229, 41, 228, 106, 184, 177, 110, 151, 184, 177,
                110, 151, 148, 179, 107, 106, 148, 179, 107, 106, 229, 41, 228, 106, 148, 179, 107,
                106, 148, 179, 107, 106, 41, 63, 199, 20, 184, 177, 110, 151,
            ],
        ),
        (
            2,
            Effort::Quality,
            [
                137, 198, 149, 255, 141, 131, 168, 255, 138, 197, 149, 255, 102, 255, 178, 255,
                138, 197, 150, 255, 137, 196, 150, 255, 105, 255, 175, 255, 105, 254, 174, 255,
                104, 255, 176, 255, 142, 130, 166, 255, 143, 130, 166, 255, 141, 130, 167, 255,
                141, 131, 164, 255, 106, 255, 177, 255, 140, 198, 148, 255, 103, 255, 178, 255,
            ],
        ),
        (
            0,
            Effort::Quality,
            [
                67, 12, 136, 255, 188, 186, 121, 255, 175, 224, 117, 255, 69, 10, 136, 255, 186,
                183, 124, 255, 66, 12, 138, 255, 177, 223, 115, 255, 179, 227, 119, 255, 186, 184,
                124, 255, 67, 13, 140, 255, 189, 186, 123, 255, 179, 224, 118, 255, 189, 185, 124,
                255, 186, 187, 122, 255, 178, 225, 119, 255, 69, 11, 138, 255,
            ],
        ),
    ];
    let mut covered_modes = 0u8;
    for (mode, effort, block) in bc7_modes {
        let result = encode(
            ImageView::rgba8(&block, 4, 4, 16).unwrap(),
            &Settings::new(Format::Bc7, effort),
        )
        .unwrap();
        assert_eq!(result[0].trailing_zeros(), mode);
        covered_modes |= 1 << mode;
        for byte in result {
            print!("{byte:02x}");
        }
        println!();
    }

    assert_eq!(covered_modes, u8::MAX);

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
