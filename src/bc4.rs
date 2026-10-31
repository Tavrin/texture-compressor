use crate::Effort;

fn palette(a: u8, b: u8) -> [i32; 8] {
    let (a, b) = (i32::from(a), i32::from(b));
    let mut p = [a, b, 0, 0, 0, 0, 0, 0];
    if a > b {
        let w = [9363, 18724, 28086, 37450, 46812, 56173];
        for i in 0..6 {
            p[i + 2] = (w[5 - i] * a + w[i] * b + 32768) >> 16;
        }
    } else {
        let w = [13107, 26215, 39321, 52429];
        for i in 0..4 {
            p[i + 2] = (w[3 - i] * a + w[i] * b + 32768) >> 16;
        }
        p[7] = 255;
    }
    p
}
// Ordered unique decoded values and the first input selecting the next value.
// Equal decoded values retain the smallest selector, as does the brute oracle.
fn thresholds(p: &[i32; 8], a: u8, b: u8) -> ([i32; 7], [usize; 8], usize) {
    let order = if a > b {
        [1, 7, 6, 5, 4, 3, 2, 0]
    } else {
        [6, 0, 2, 3, 4, 5, 1, 7]
    };
    let mut indices = [0usize; 8];
    let mut n = 0;
    for j in order {
        if n > 0 && p[indices[n - 1]] == p[j] {
            indices[n - 1] = indices[n - 1].min(j);
        } else {
            indices[n] = j;
            n += 1;
        }
    }
    let mut bounds = [256; 7];
    for i in 0..n - 1 {
        let l = indices[i];
        let r = indices[i + 1];
        let sum = p[l] + p[r];
        bounds[i] = (sum + 1) / 2 + i32::from(sum % 2 == 0 && l < r);
    }
    (bounds, indices, n)
}
fn nearest(
    p: &[i32; 8],
    bounds: &[i32; 7],
    indices: &[usize; 8],
    _n: usize,
    x: u8,
) -> (u32, usize) {
    let x = i32::from(x);
    // Unused boundaries are 256, so three fixed comparisons work for all
    // palette sizes without a data-dependent binary-search loop.
    let mut lo = if x >= bounds[3] { 4 } else { 0 };
    if x >= bounds[lo + 1] {
        lo += 2;
    }
    if x >= bounds[lo] {
        lo += 1;
    }
    let j = indices[lo];
    let d = x - p[j];
    ((d * d) as u32, j)
}
fn evaluate_balanced(v: &[u8; 16], a: u8, b: u8, best: &mut (u32, u8, u8, u64)) {
    let p = palette(a, b);
    let (bounds, indices, n) = thresholds(&p, a, b);
    let mut err = 0u32;
    let mut bits = 0u64;
    for (i, &x) in v.iter().enumerate() {
        let (d, j) = nearest(&p, &bounds, &indices, n, x);
        err += d;
        if err > best.0 {
            return;
        }
        bits |= (j as u64) << (3 * i);
    }
    *best = (*best).min((err, a, b, bits));
}

fn evaluate(v: &[u8; 16], a: u8, b: u8, best: &mut (u32, u8, u8, u64)) {
    let p = palette(a, b);
    let mut err = 0u32;
    let mut bits = 0u64;
    for (i, &x) in v.iter().enumerate() {
        let mut choice = (u32::MAX, 0usize);
        for (j, &c) in p.iter().enumerate() {
            let d = i32::from(x) - c;
            choice = choice.min(((d * d) as u32, j));
        }
        err += choice.0;
        // The accumulated error is a lower bound for the final error.
        if err > best.0 {
            return;
        }
        bits |= (choice.1 as u64) << (3 * i);
    }
    *best = (*best).min((err, a, b, bits));
}
fn refine(v: &[u8; 16], a: u8, b: u8, thresholded: bool) -> (u8, u8) {
    let p = palette(a, b);
    let (bounds, indices, n) = if thresholded {
        thresholds(&p, a, b)
    } else {
        ([0; 7], [0; 8], 0)
    };
    let mut aa = 0i64;
    let mut ab = 0i64;
    let mut bb = 0i64;
    let mut ax = 0i64;
    let mut bx = 0i64;
    for &x in v {
        let j = if thresholded {
            nearest(&p, &bounds, &indices, n, x).1
        } else {
            let mut best = (i32::MAX, 0usize);
            for (j, &c) in p.iter().enumerate() {
                let d = i32::from(x) - c;
                best = best.min((d * d, j));
            }
            best.1
        };
        let (u, w, den) = if a > b {
            match j {
                0 => (7, 0, 7),
                1 => (0, 7, 7),
                _ => (8 - j as i64, j as i64 - 1, 7),
            }
        } else {
            match j {
                0 => (5, 0, 5),
                1 => (0, 5, 5),
                2..=5 => (6 - j as i64, j as i64 - 1, 5),
                _ => (0, 0, 5),
            }
        };
        aa += u * u;
        ab += u * w;
        bb += w * w;
        ax += u * i64::from(x) * den;
        bx += w * i64::from(x) * den;
    }
    let det = aa * bb - ab * ab;
    if det == 0 {
        return (a, b);
    }
    let round = |n: i64| ((n + det / 2) / det).clamp(0, 255) as u8;
    (round(ax * bb - bx * ab), round(bx * aa - ax * ab))
}
pub fn encode(pixels: &[[u8; 4]; 16], effort: Effort, channel: usize) -> [u8; 8] {
    let mut v = [0u8; 16];
    for i in 0..16 {
        v[i] = pixels[i][channel];
    }
    let min = *v.iter().min().unwrap();
    let max = *v.iter().max().unwrap();
    if min == max {
        // Both extrema are represented exactly by the six-entry mode.
        let (a, b, selector) = if min == 0 || min == 255 {
            (0, 0, if min == 0 { 0u64 } else { 7 })
        } else {
            (0, min, 1)
        };
        let bits = (0..16).fold(0u64, |bits, i| bits | (selector << (3 * i)));
        let mut out = [0u8; 8];
        out[0] = a;
        out[1] = b;
        out[2..].copy_from_slice(&bits.to_le_bytes()[..6]);
        return out;
    }
    let mut best = (u32::MAX, 0, 0, 0);
    if effort == Effort::Fast || effort == Effort::Balanced {
        let (mut a, mut b) = refine(&v, max, min, effort == Effort::Balanced);
        if a <= b {
            (a, b) = (max, min);
        }
        if effort == Effort::Balanced {
            evaluate_balanced(&v, a, b, &mut best);
        } else {
            evaluate(&v, a, b, &mut best);
        }
        // The six-value mode has fixed 0 and 255 entries; it can improve
        // the fit when either extreme is present.
        if (effort == Effort::Balanced && best.0 > 4) || min == 0 || max == 255 {
            let inner_min = v
                .iter()
                .copied()
                .filter(|&x| x != 0 && x != 255)
                .min()
                .unwrap_or(0);
            let inner_max = v
                .iter()
                .copied()
                .filter(|&x| x != 0 && x != 255)
                .max()
                .unwrap_or(255);
            let (mut a, mut b) = if effort == Effort::Balanced && min != 0 && max != 255 {
                (inner_min, inner_max)
            } else {
                refine(&v, inner_min, inner_max, effort == Effort::Balanced)
            };
            if a > b {
                (a, b) = (inner_min, inner_max);
            }
            if effort == Effort::Balanced {
                evaluate_balanced(&v, a, b, &mut best);
            } else {
                evaluate(&v, a, b, &mut best);
            }
        }
        // Corpus diagnostic: low-span distributed roughness blocks can
        // retain residual error below the ordinary neighborhood gate.
        if effort == Effort::Balanced && (8..=15).contains(&(max - min)) && best.0 > 0 {
            evaluate_balanced(&v, max - 1, min + 1, &mut best);
        }
        if effort == Effort::Balanced && best.0 > 16 {
            let (a, b) = (best.1, best.2);
            for da in -2..=2 {
                for db in -2..=2 {
                    if (da == 0) != (db == 0) {
                        evaluate_balanced(
                            &v,
                            (i32::from(a) + da).clamp(0, 255) as u8,
                            (i32::from(b) + db).clamp(0, 255) as u8,
                            &mut best,
                        );
                    }
                }
            }
        }
        let mut out = [0u8; 8];
        out[0] = best.1;
        out[1] = best.2;
        out[2..].copy_from_slice(&best.3.to_le_bytes()[..6]);
        return out;
    }
    evaluate(&v, max, min, &mut best);
    evaluate(&v, min, max, &mut best);
    let mut unique = v;
    unique.sort_unstable();
    let count = if effort == Effort::Quality { 16 } else { 2 };
    let mut seen = [(0u8, 0u8); 32];
    let mut seen_count = 0;
    for i in 0..count {
        let a = unique[i * 15 / (count - 1)];
        let b = unique[15 - i * 15 / (count - 1)];
        for (mut a, mut b) in [(a, b), (b, a)] {
            if seen[..seen_count].contains(&(a, b)) {
                continue;
            }
            seen[seen_count] = (a, b);
            seen_count += 1;
            let refinements = if effort == Effort::Quality { 2 } else { 1 };
            for _ in 0..refinements {
                evaluate(&v, a, b, &mut best);
                let next = refine(&v, a, b, false);
                if next == (a, b) {
                    break;
                }
                (a, b) = next;
            }
            evaluate(&v, a, b, &mut best);
        }
    }
    let radius = if effort == Effort::Quality { 3 } else { 1 };
    let (a, b) = (best.1, best.2);
    for da in -radius..=radius {
        for db in -radius..=radius {
            evaluate(
                &v,
                (i32::from(a) + da).clamp(0, 255) as u8,
                (i32::from(b) + db).clamp(0, 255) as u8,
                &mut best,
            );
        }
    }
    let mut out = [0u8; 8];
    out[0] = best.1;
    out[1] = best.2;
    let b = best.3.to_le_bytes();
    out[2..8].copy_from_slice(&b[..6]);
    out
}

// Fast/Balanced need no endpoint search to satisfy the range-fit bound.
// Assign exact nearest selectors without accumulating an unused SSE.
#[inline]
fn range_fit(pixels: &[[u8; 4]; 16], channel: usize) -> [u8; 8] {
    let v = pixels.map(|p| p[channel]);
    range_fit_values(&v)
}

#[inline(always)]
pub(crate) fn range_fit_values(v: &[u8; 16]) -> [u8; 8] {
    #[cfg(all(feature = "simd", feature = "std", target_arch = "x86_64"))]
    if let Some(encoder) = crate::bc7::range_encoder_sse41() {
        return encoder.encode(v);
    }
    #[cfg(all(feature = "simd", feature = "std", target_arch = "x86_64"))]
    if let Some(encoded) = crate::bc7::range_channel(v) {
        return encoded;
    }
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let (min, max) = crate::bc7::byte_bounds(v);
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    let (min, max) = scalar_range_bounds(v);
    if min == max {
        return [max, min, 0, 0, 0, 0, 0, 0];
    }
    let bits = range_selectors(v, min, max);
    pack((0, max, min, bits))
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
fn scalar_range_bounds(v: &[u8; 16]) -> (u8, u8) {
    // A fixed tree exposes independent byte min/max operations to SIMD.
    let lo8: [u8; 8] = core::array::from_fn(|i| v[i].min(v[i + 8]));
    let hi8: [u8; 8] = core::array::from_fn(|i| v[i].max(v[i + 8]));
    let lo4: [u8; 4] = core::array::from_fn(|i| lo8[i].min(lo8[i + 4]));
    let hi4: [u8; 4] = core::array::from_fn(|i| hi8[i].max(hi8[i + 4]));
    let min = lo4[0].min(lo4[2]).min(lo4[1].min(lo4[3]));
    let max = hi4[0].max(hi4[2]).max(hi4[1].max(hi4[3]));
    (min, max)
}

// For spans up to seven, every source byte in the range is represented
// exactly. These are the smallest selectors for each offset from the minimum.
pub(crate) const NARROW_SELECTORS: [[u8; 8]; 8] = [
    [0; 8],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 3, 0, 0, 0, 0, 0, 0],
    [1, 5, 3, 0, 0, 0, 0, 0],
    [1, 6, 4, 2, 0, 0, 0, 0],
    [1, 6, 5, 4, 2, 0, 0, 0],
    [1, 7, 6, 4, 3, 2, 0, 0],
    [1, 7, 6, 5, 4, 3, 2, 0],
];

pub(crate) const RANGE_BOUND_OFFSETS: [[u8; 7]; 256] = range_bound_offsets();

const fn range_bound_offsets() -> [[u8; 7]; 256] {
    let mut table = [[0; 7]; 256];
    let weights = [9363u32, 18724, 28086, 37450, 46812, 56173];
    let mut index = 0;
    let mut span = 0;
    while index < 256 {
        let mut p = [0; 6];
        let mut i = 0;
        while i < 6 {
            p[i] = (weights[i] * span + 32768) >> 16;
            i += 1;
        }
        // Every used offset (span > 7) is <= span <= 255.
        table[index] = [
            (p[0] / 2 + 1).to_le_bytes()[0],
            (p[0] + p[1]).div_ceil(2).to_le_bytes()[0],
            (p[1] + p[2]).div_ceil(2).to_le_bytes()[0],
            (p[2] + p[3]).div_ceil(2).to_le_bytes()[0],
            (p[3] + p[4]).div_ceil(2).to_le_bytes()[0],
            (p[4] + p[5]).div_ceil(2).to_le_bytes()[0],
            (p[5] + span).div_ceil(2).to_le_bytes()[0],
        ];
        index += 1;
        span += 1;
    }
    table
}

fn range_selectors(v: &[u8; 16], min: u8, max: u8) -> u64 {
    let selectors = if max - min <= 7 {
        let table = &NARROW_SELECTORS[usize::from(max - min)];
        #[cfg(all(feature = "simd", feature = "std", target_arch = "x86_64"))]
        if let Some(selectors) = crate::bc7::byte_lookup(v, min, table) {
            return pack_selectors(selectors);
        }
        v.map(|x| table[usize::from(x - min)])
    } else {
        // Decoded levels are min + round(weight * span / 65536), so all
        // boundary offsets depend only on the span. The table preserves the
        // exact palette rounding and smallest-selector midpoint ties.
        let bounds = RANGE_BOUND_OFFSETS[usize::from(max - min)].map(|offset| min + offset);
        #[cfg(all(feature = "simd", feature = "std", target_arch = "x86_64"))]
        if let Some(selectors) = crate::bc7::byte_search(v, &bounds) {
            return pack_selectors(selectors);
        }
        // Independent comparisons let the compiler vectorize across texels.
        v.map(|x| {
            let q = u8::from(x >= bounds[0])
                + u8::from(x >= bounds[1])
                + u8::from(x >= bounds[2])
                + u8::from(x >= bounds[3])
                + u8::from(x >= bounds[4])
                + u8::from(x >= bounds[5])
                + u8::from(x >= bounds[6]);
            if q == 0 {
                1
            } else if q == 7 {
                0
            } else {
                8 - q
            }
        })
    };
    pack_selectors(selectors)
}

fn pack_selectors(selectors: [u8; 16]) -> u64 {
    // Compact eight three-bit selectors from byte positions to 24 bits,
    // then concatenate the two halves. Explicit LE loads are portable.
    fn compact(mut x: u64) -> u64 {
        x = (x | (x >> 5)) & 0x003f_003f_003f_003f;
        x = (x | (x >> 10)) & 0x0000_0fff_0000_0fff;
        (x | (x >> 20)) & 0x00ff_ffff
    }
    compact(u64::from_le_bytes(core::array::from_fn(|i| selectors[i])))
        | (compact(u64::from_le_bytes(core::array::from_fn(|i| {
            selectors[i + 8]
        }))) << 24)
}

// Keep the legacy encoder isolated so opt-in fitting cannot change its bytes.
pub fn encode_with_fit(
    pixels: &[[u8; 4]; 16],
    effort: Effort,
    channel: usize,
    fit: crate::Bc45Fit,
) -> [u8; 8] {
    if fit == crate::Bc45Fit::SquaredError {
        return encode(pixels, effort, channel);
    }
    if effort != Effort::Quality {
        return range_fit(pixels, channel);
    }
    encode_peak_quality(pixels, channel)
}

// Keep the broad search's stack frame out of direct range fitting.
#[inline(never)]
fn encode_peak_quality(pixels: &[[u8; 4]; 16], channel: usize) -> [u8; 8] {
    let v = pixels.map(|p| p[channel]);
    let min = *v.iter().min().unwrap();
    let max = *v.iter().max().unwrap();
    let mut best = (u32::MAX, 0, 0, 0);
    evaluate_balanced(&v, max, min, &mut best);
    let peak = peak_error(&v, &best);
    let legacy = encode(pixels, Effort::Quality, channel);
    if best.0 == 0 {
        return pack(best);
    }
    let mut candidate = (u32::MAX, 0, 0, 0);
    evaluate_balanced(&v, legacy[0], legacy[1], &mut candidate);
    if peak_error(&v, &candidate) <= peak {
        best = best.min(candidate);
    }
    // Search both the range and preset fits; fixed work, integer tie breaks.
    for (a, b) in [(max, min), (legacy[0], legacy[1]), (min, max)] {
        for da in -3..=3 {
            for db in -3..=3 {
                let a = (i32::from(a) + da).clamp(0, 255) as u8;
                let b = (i32::from(b) + db).clamp(0, 255) as u8;
                let p = palette(a, b);
                let (bounds, indices, n) = thresholds(&p, a, b);
                let mut err = 0;
                let mut bits = 0;
                for (i, &x) in v.iter().enumerate() {
                    let (d, j) = nearest(&p, &bounds, &indices, n, x);
                    if d > peak * peak {
                        err = u32::MAX;
                        break;
                    }
                    err += d;
                    if err > best.0 {
                        break;
                    }
                    bits |= (j as u64) << (3 * i);
                }
                best = best.min((err, a, b, bits));
            }
        }
    }
    pack(best)
}
// BC5's two byte fits share an additional worst-normal score. This does not
// convert the input: it only rejects a refinement of the unsigned XY range fit.
pub fn encode_pair(pixels: &[[u8; 4]; 16], effort: Effort, fit: crate::Bc45Fit) -> [u8; 16] {
    let mut out = [0; 16];
    out[..8].copy_from_slice(&encode_with_fit(pixels, effort, 0, fit));
    out[8..].copy_from_slice(&encode_with_fit(pixels, effort, 1, fit));
    if fit == crate::Bc45Fit::SquaredError || effort != Effort::Quality {
        return out;
    }
    let mut range = [0; 16];
    for c in 0..2 {
        let v = pixels.map(|p| p[c]);
        let mut seed = (u32::MAX, 0, 0, 0);
        evaluate_balanced(
            &v,
            *v.iter().max().unwrap(),
            *v.iter().min().unwrap(),
            &mut seed,
        );
        range[c * 8..c * 8 + 8].copy_from_slice(&pack(seed));
    }
    if out != range && normal_score(pixels, &out) < normal_score(pixels, &range) {
        return range;
    }
    out
}
fn normal_score(pixels: &[[u8; 4]; 16], block: &[u8; 16]) -> i64 {
    let p = [palette(block[0], block[1]), palette(block[8], block[9])];
    let bits = [
        u64::from_le_bytes(block[..8].try_into().unwrap()) >> 16,
        u64::from_le_bytes(block[8..].try_into().unwrap()) >> 16,
    ];
    let mut worst = i64::MAX;
    for (i, source) in pixels.iter().enumerate() {
        let a = normal(source[0], source[1]);
        let b = normal(
            p[0][((bits[0] >> (i * 3)) & 7) as usize] as u8,
            p[1][((bits[1] >> (i * 3)) & 7) as usize] as u8,
        );
        let dot: i64 = (0..3).map(|c| a[c] * b[c]).sum();
        let aa: i64 = a.iter().map(|x| x * x).sum();
        let bb: i64 = b.iter().map(|x| x * x).sum();
        // Signed cosine squared is monotone in cosine on [-1,1]. The
        // 40-bit score and 12 fractional coordinate bits avoid floating point.
        // |dot| <= 2*(255*4096)^2; its square shifted by 40 fits u128.
        let magnitude = u128::from(dot.unsigned_abs());
        let score = (((magnitude * magnitude) << 40) / (aa as u128 * bb as u128)) as i64;
        worst = worst.min(if dot < 0 { -score } else { score });
    }
    worst
}
fn normal(r: u8, g: u8) -> [i64; 3] {
    let x = (2 * i64::from(r) - 255) * 4096;
    let y = (2 * i64::from(g) - 255) * 4096;
    let z2 = ((255i64 * 4096).pow(2) - x * x - y * y).max(0) as u64;
    [x, y, z2.isqrt() as i64]
}

fn pack(fit: (u32, u8, u8, u64)) -> [u8; 8] {
    let mut out = [0; 8];
    out[0] = fit.1;
    out[1] = fit.2;
    out[2..].copy_from_slice(&fit.3.to_le_bytes()[..6]);
    out
}

fn peak_error(v: &[u8; 16], fit: &(u32, u8, u8, u64)) -> u32 {
    let p = palette(fit.1, fit.2);
    v.iter()
        .enumerate()
        .map(|(i, &x)| (i32::from(x) - p[((fit.3 >> (3 * i)) & 7) as usize]).unsigned_abs())
        .max()
        .unwrap()
}

#[cfg(test)]
mod threshold_tests {
    use super::*;
    #[test]
    fn range_fit_matches_oracle_for_all_endpoint_pairs() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                let v = core::array::from_fn(|i| if i % 2 == 0 { a } else { b });
                let mut oracle = (u32::MAX, 0, 0, 0);
                evaluate_balanced(&v, a.max(b), a.min(b), &mut oracle);
                assert_eq!(range_fit_values(&v), pack(oracle), "{a} {b}");
            }
        }
    }

    #[test]
    fn range_selectors_match_generic_evaluation() {
        for min in 0..=255u8 {
            for max in min..=255u8 {
                for base in (usize::from(min)..=usize::from(max)).step_by(16) {
                    let v = core::array::from_fn(|i| (base + i).min(usize::from(max)) as u8);
                    let mut oracle = (u32::MAX, 0, 0, 0);
                    evaluate_balanced(&v, max, min, &mut oracle);
                    assert_eq!(
                        range_selectors(&v, min, max),
                        oracle.3,
                        "{min} {max} {base}"
                    );
                    for (lo_slot, hi_slot) in [(0, 15), (1, 14)] {
                        let mut block = v;
                        block[lo_slot] = min;
                        block[hi_slot] = max;
                        let mut fit_oracle = (u32::MAX, 0, 0, 0);
                        evaluate_balanced(&block, max, min, &mut fit_oracle);
                        assert_eq!(
                            range_fit_values(&block),
                            pack(fit_oracle),
                            "{min} {max} {base}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn all_endpoints_and_values_match_brute_force() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                let p = palette(a, b);
                let (bounds, indices, n) = thresholds(&p, a, b);
                for x in 0..=255u8 {
                    let oracle = p
                        .iter()
                        .enumerate()
                        .map(|(j, &c)| {
                            let d = i32::from(x) - c;
                            ((d * d) as u32, j)
                        })
                        .min()
                        .unwrap();
                    assert_eq!(nearest(&p, &bounds, &indices, n, x), oracle, "{a} {b} {x}");
                }
            }
        }
    }
}
