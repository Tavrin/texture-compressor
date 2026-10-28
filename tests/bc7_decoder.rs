//! Independent test decoder based on the Microsoft BC7 and Khronos BPTC specifications.
//! This reader does not share encoder code or state.
#[path = "bc7_tables.rs"]
mod tables;

pub fn decode(block: &[u8]) -> [[u8; 4]; 16] {
    let bits = u128::from_le_bytes(block.try_into().unwrap());
    let mode = (bits as u8).trailing_zeros() as usize;
    if mode == 8 {
        return [[0; 4]; 16];
    }
    let subsets: usize = [3, 2, 3, 2, 1, 1, 1, 2][mode];
    let partition_bits = [4, 6, 6, 6, 0, 0, 0, 6][mode];
    let color_bits = [4, 6, 5, 7, 5, 7, 7, 5][mode];
    let alpha_bits = [0, 0, 0, 0, 6, 8, 7, 5][mode];
    let index_bits = [3, 3, 2, 2, 2, 2, 4, 2][mode];
    let second_bits = [0, 0, 0, 0, 3, 2, 0, 0][mode];
    let mut pos = mode + 1;
    let mut read = |n: usize| -> usize {
        let result = ((bits >> pos) & ((1u128 << n) - 1)) as usize;
        pos += n;
        result
    };
    let partition = read(partition_bits);
    let rotation = if mode == 4 || mode == 5 { read(2) } else { 0 };
    let selection = if mode == 4 { read(1) } else { 0 };
    let mut ep = [[0usize; 4]; 6];
    for channel in 0..3 {
        for endpoint in ep.iter_mut().take(subsets * 2) {
            endpoint[channel] = read(color_bits);
        }
    }
    if alpha_bits > 0 {
        for endpoint in ep.iter_mut().take(subsets * 2) {
            endpoint[3] = read(alpha_bits);
        }
    }
    let has_p = matches!(mode, 0 | 1 | 3 | 6 | 7);
    if mode == 1 {
        for subset in 0..subsets {
            let p = read(1);
            for endpoint in &mut ep[2 * subset..2 * subset + 2] {
                for v in &mut endpoint[..3] {
                    *v = (*v << 1) | p;
                }
            }
        }
    } else if has_p {
        for endpoint in ep.iter_mut().take(subsets * 2) {
            let p = read(1);
            for v in endpoint.iter_mut().take(if alpha_bits > 0 { 4 } else { 3 }) {
                *v = (*v << 1) | p;
            }
        }
    }
    for endpoint in ep.iter_mut().take(subsets * 2) {
        for (c, v) in endpoint.iter_mut().enumerate() {
            let n = if c == 3 { alpha_bits } else { color_bits };
            if n == 0 {
                *v = 255;
            } else {
                let n = n + usize::from(has_p);
                *v = (*v << (8 - n)) | (*v >> (2 * n - 8));
            }
        }
    }
    let pattern = match subsets {
        2 => tables::P2[partition],
        3 => tables::P3[partition],
        _ => 0,
    };
    let anchors = match subsets {
        2 => [0, tables::A2[partition], 16],
        3 => [0, tables::A3A[partition], tables::A3B[partition]],
        _ => [0, 16, 16],
    };
    let mut primary = [0usize; 16];
    for (i, value) in primary.iter_mut().enumerate() {
        *value = read(index_bits - usize::from(anchors.contains(&i)));
    }
    let mut secondary = [0usize; 16];
    if second_bits != 0 {
        for (i, value) in secondary.iter_mut().enumerate() {
            *value = read(second_bits - usize::from(i == 0));
        }
    }
    assert_eq!(pos, 128);
    const W2: [usize; 4] = [0, 21, 43, 64];
    const W3: [usize; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
    const W4: [usize; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];
    let weight = |n, i| match n {
        2 => W2[i],
        3 => W3[i],
        4 => W4[i],
        _ => unreachable!(),
    };
    let mut out = [[0u8; 4]; 16];
    for (i, pixel) in out.iter_mut().enumerate() {
        let subset = ((pattern >> (2 * i)) & 3) as usize;
        for (c, value) in pixel.iter_mut().enumerate() {
            let use_second = second_bits != 0 && ((c == 3) != (selection == 1));
            let w = if use_second {
                weight(second_bits, secondary[i])
            } else {
                weight(index_bits, primary[i])
            };
            *value = (((64 - w) * ep[subset * 2][c] + w * ep[subset * 2 + 1][c] + 32) >> 6) as u8;
        }
        if rotation != 0 {
            pixel.swap(3, rotation - 1);
        }
    }
    out
}

#[test]
fn zero_endpoints_for_each_legal_mode_and_reserved() {
    assert_eq!(decode(&[0; 16]), [[0; 4]; 16]);
    for mode in 0..8 {
        let mut block = [0; 16];
        block[0] = 1 << mode;
        let expected = if mode < 4 { [0, 0, 0, 255] } else { [0; 4] };
        assert_eq!(decode(&block), [expected; 16], "mode {mode}");
    }
}

// Construct solid-per-subset blocks without invoking the encoder. A different
// extreme per subset and component tests endpoint ordering, partitions, alpha,
// both mode-4 index-selection values and all channel rotations.
#[test]
fn solid_legal_blocks_cover_partitions_rotations_and_index_selection() {
    for mode in 0..8 {
        let subsets: usize = [3, 2, 3, 2, 1, 1, 1, 2][mode];
        let partition_bits = [4, 6, 6, 6, 0, 0, 0, 6][mode];
        let cb = [4, 6, 5, 7, 5, 7, 7, 5][mode];
        let ab = [0, 0, 0, 0, 6, 8, 7, 5][mode];
        let rotations = if mode == 4 || mode == 5 { 4 } else { 1 };
        let selections = if mode == 4 { 2 } else { 1 };
        for partition in 0..(1 << partition_bits) {
            for rotation in 0..rotations {
                for selection in 0..selections {
                    let mut bits = 1u128 << mode;
                    let mut pos = mode + 1;
                    let has_p = matches!(mode, 0 | 1 | 3 | 6 | 7);
                    {
                        let mut put = |value: usize, width: usize| {
                            bits |= (value as u128) << pos;
                            pos += width;
                        };
                        put(partition, partition_bits);
                        if rotations == 4 {
                            put(rotation, 2);
                        }
                        if mode == 4 {
                            put(selection, 1);
                        }
                        // P-bits are zero. Components use either zero or all
                        // one bits, and their exact expansion is computed below.
                        for c in 0..4 {
                            let width = if c == 3 { ab } else { cb };
                            if width == 0 {
                                continue;
                            }
                            for subset in 0..subsets {
                                let raw = if (subset + c).is_multiple_of(2) {
                                    0
                                } else {
                                    (1 << width) - 1
                                };
                                put(raw, width);
                                put(raw, width);
                            }
                        }
                        if has_p {
                            for _ in 0..if mode == 1 { subsets } else { subsets * 2 } {
                                put(0, 1);
                            }
                        }
                    }
                    let block = bits.to_le_bytes();
                    let decoded = decode(&block);
                    let pattern = match subsets {
                        2 => tables::P2[partition],
                        3 => tables::P3[partition],
                        _ => 0,
                    };
                    for (i, pixel) in decoded.iter().enumerate() {
                        let subset = ((pattern >> (2 * i)) & 3) as usize;
                        let mut expected = [0u8; 4];
                        for (c, value) in expected.iter_mut().enumerate() {
                            let width = if c == 3 { ab } else { cb };
                            *value = if width == 0 {
                                255
                            } else if (subset + c).is_multiple_of(2) {
                                0
                            } else if has_p {
                                let n = width + 1;
                                let v = ((1 << width) - 1) << 1;
                                ((v << (8 - n)) | (v >> (2 * n - 8))) as u8
                            } else {
                                255
                            };
                        }
                        if rotation != 0 {
                            expected.swap(3, rotation - 1);
                        }
                        assert_eq!(*pixel, expected, "mode {mode}, partition {partition}, rotation {rotation}, selection {selection}, texel {i}");
                    }
                }
            }
        }
    }
}

// Fixed legal blocks and expected pixels generated by bcdec_rs 0.2.0 (MIT),
// independently checked against this reader. Every legal partition occurs.
#[test]
fn oracle_vectors_cover_all_modes_and_partitions() {
    for (line_number, line) in include_str!("bc7_vectors.txt").lines().enumerate() {
        assert_eq!(line.len(), 160);
        let mut bytes = [0u8; 80];
        for (value, pair) in bytes
            .iter_mut()
            .zip(line.as_bytes().as_chunks::<2>().0.iter())
        {
            *value = u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap();
        }
        assert_eq!(
            decode(&bytes[..16]).as_flattened(),
            &bytes[16..],
            "vector {line_number}"
        );
    }
}
