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

#[cfg(test)]
mod threshold_tests {
    use super::*;
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
