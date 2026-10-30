//! Block-local work allocation. Integer arithmetic fixes decisions across targets.

// Compare the seed SSE with the block's centred energy. A quantization-noise
// allowance prevents tiny ranges from consuming the largest search budget.
// Return zero only for an exact seed; every other block gets 1..=4 units.
pub(crate) fn difficulty(
    pixels: &[[u8; 4]; 16],
    error: u32,
    channels: usize,
    transparent: Option<u8>,
) -> usize {
    if error == 0 {
        return 0;
    }
    let mut count = 0u64;
    let mut sum = [0u64; 4];
    let mut square = [0u64; 4];
    for p in pixels {
        if transparent.is_some_and(|t| p[3] < t) {
            continue;
        }
        count += 1;
        for c in 0..channels {
            let v = u64::from(p[c]);
            sum[c] += v;
            square[c] += v * v;
        }
    }
    let energy: u64 = (0..channels)
        .map(|c| count * square[c] - sum[c] * sum[c])
        .sum();
    let noise = count * count * channels as u64 * 4;
    1 + (4 * count * u64::from(error) / (energy + noise).max(1)).min(3) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_seeds_extremes_and_masked_texels() {
        let mut px = [[127; 4]; 16];
        assert_eq!(difficulty(&px, 0, 4, None), 0);
        assert_eq!(difficulty(&px, u32::MAX, 4, None), 4);
        for (i, p) in px.iter_mut().enumerate() {
            *p = [if i % 2 == 0 { 0 } else { 255 }; 4];
        }
        for channels in [3, 4] {
            let mut previous = 0;
            for error in [0, 1, 16, 256, 4096, 65536, 1040400, u32::MAX] {
                let units = difficulty(&px, error, channels, None);
                assert!((previous..=4).contains(&units));
                previous = units;
            }
        }
        let mut masked = [[20, 40, 60, 255]; 16];
        for p in &mut masked[8..] {
            *p = [0, 0, 0, 0];
        }
        let before = difficulty(&masked, 32, 3, Some(128));
        for p in &mut masked[8..] {
            *p = [255, 255, 255, 0];
        }
        assert_eq!(difficulty(&masked, 32, 3, Some(128)), before);
    }
}
