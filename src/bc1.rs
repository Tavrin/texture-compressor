use crate::{AlphaPolicy, Effort};

fn components(c: u16) -> [i32; 3] {
    [
        i32::from(c >> 11),
        i32::from((c >> 5) & 63),
        i32::from(c & 31),
    ]
}
fn palette(a: u16, b: u16) -> [[i32; 3]; 4] {
    let a = components(a);
    let b = components(b);
    let mut p = [[0; 3]; 4];
    for c in 0..3 {
        let (mul, add, shift) = if c == 1 { (259, 33, 6) } else { (527, 23, 6) };
        p[0][c] = (a[c] * mul + add) >> shift;
        p[1][c] = (b[c] * mul + add) >> shift;
    }
    p
}
fn colors(a: u16, b: u16) -> [[i32; 3]; 4] {
    let mut p = palette(a, b);
    let x = components(a);
    let y = components(b);
    for c in 0..3 {
        if a > b {
            let (mul, add, shift) = if c == 1 {
                (2763, 1039, 11)
            } else {
                (351, 61, 7)
            };
            p[2][c] = ((2 * x[c] + y[c]) * mul + add) >> shift;
            p[3][c] = ((x[c] + 2 * y[c]) * mul + add) >> shift;
        } else {
            let (mul, add, shift) = if c == 1 {
                (4145, 1019, 11)
            } else {
                (1053, 125, 8)
            };
            p[2][c] = ((x[c] + y[c]) * mul + add) >> shift;
        }
    }
    p
}
fn pack(v: [i32; 3]) -> u16 {
    ((v[0].clamp(0, 31) as u16) << 11)
        | ((v[1].clamp(0, 63) as u16) << 5)
        | (v[2].clamp(0, 31) as u16)
}
const fn quant_table(limit: i32, mul: i32, add: i32) -> [i32; 256] {
    let mut table = [0; 256];
    let mut v = 0;
    while v < 256 {
        let initial = (v as i32 * limit + 127) / 255;
        let mut x = if initial > 0 { initial - 1 } else { 0 };
        let end = if initial < limit { initial + 1 } else { limit };
        let mut distance = i32::MAX;
        while x <= end {
            let delta = v as i32 - ((x * mul + add) >> 6);
            let d = if delta < 0 { -delta } else { delta };
            if d < distance {
                distance = d;
                table[v] = x;
            }
            x += 1;
        }
        v += 1;
    }
    table
}
const Q5: [i32; 256] = quant_table(31, 527, 23);
const Q6: [i32; 256] = quant_table(63, 259, 33);
fn quant(v: [i32; 3]) -> u16 {
    pack([
        Q5[v[0].clamp(0, 255) as usize],
        Q6[v[1].clamp(0, 255) as usize],
        Q5[v[2].clamp(0, 255) as usize],
    ])
}

type SolidOptions = [(i32, i32, i32); 3];
type SolidTable = [[[[SolidOptions; 256]; 2]; 4]; 2];
include!(concat!(env!("OUT_DIR"), "/solid.rs"));

fn evaluate(
    px: &[[u8; 4]; 16],
    transparent: Option<u8>,
    a: u16,
    b: u16,
    best: &mut (u32, u16, u16, u32),
) {
    let three = transparent.is_some();
    if (three && a > b) || (!three && a <= b) {
        return;
    }
    let p = colors(a, b);
    let mut err = 0u32;
    let mut bits = 0u32;
    for (i, v) in px.iter().enumerate() {
        if transparent.is_some_and(|t| v[3] < t) {
            bits |= 3 << (2 * i);
            continue;
        }
        let mut choice = (u32::MAX, 0usize);
        for (j, c) in p.iter().enumerate().take(if three { 3 } else { 4 }) {
            let d = (0..3)
                .map(|k| {
                    let d = i32::from(v[k]) - c[k];
                    (d * d) as u32
                })
                .sum();
            choice = choice.min((d, j));
        }
        err += choice.0;
        if err > best.0 {
            return;
        }
        bits |= (choice.1 as u32) << (2 * i);
    }
    *best = (*best).min((err, a, b, bits));
}
fn candidate(
    px: &[[u8; 4]; 16],
    trans: Option<u8>,
    a: u16,
    b: u16,
    best: &mut (u32, u16, u16, u32),
) {
    if trans.is_some() {
        evaluate(px, trans, a.min(b), a.max(b), best)
    } else if a == b {
        if a < u16::MAX {
            evaluate(px, trans, a + 1, b, best)
        }
        if a > 0 {
            evaluate(px, trans, a, b - 1, best)
        }
    } else {
        evaluate(px, trans, a.max(b), a.min(b), best)
    }
}

// BC3 has no punch-through colour mode. Try a cheap four-colour seed and
// reserve the covariance/refit path for blocks with a larger decoded error.
pub fn encode_bc3_balanced(px: &[[u8; 4]; 16]) -> [u8; 8] {
    let mut lo = [255i32; 3];
    let mut hi = [0i32; 3];
    for p in px {
        for c in 0..3 {
            lo[c] = lo[c].min(i32::from(p[c]));
            hi[c] = hi[c].max(i32::from(p[c]));
        }
    }
    let mut best = (u32::MAX, 0u16, 0u16, 0u32);
    candidate(px, None, quant(lo), quant(hi), &mut best);
    if best.0 > 512 {
        return encode(px, Effort::Fast, AlphaPolicy::Opaque);
    }
    let mut out = [0u8; 8];
    out[..2].copy_from_slice(&best.1.to_le_bytes());
    out[2..4].copy_from_slice(&best.2.to_le_bytes());
    out[4..].copy_from_slice(&best.3.to_le_bytes());
    out
}
// Prefix sums make every ordered cluster fit independent of block size.
fn fit(
    prefix: &[[i64; 3]; 17],
    n: usize,
    cuts: (usize, usize, usize),
    trans: bool,
) -> Option<(u16, u16)> {
    let (i, j, k) = cuts;
    let counts = if trans {
        [i, j - i, n - j, 0]
    } else {
        [i, j - i, k - j, n - k]
    };
    let weights: [(i64, i64); 4] = if trans {
        [(2, 0), (1, 1), (0, 2), (0, 0)]
    } else {
        [(3, 0), (2, 1), (1, 2), (0, 3)]
    };
    let bounds = if trans {
        [0, i, j, n, n]
    } else {
        [0, i, j, k, n]
    };
    let (mut aa, mut ab, mut bb) = (0i64, 0i64, 0i64);
    let (mut ax, mut bx) = ([0i64; 3], [0i64; 3]);
    let den = if trans { 2 } else { 3 };
    for g in 0..4 {
        let (u, v) = weights[g];
        let count = counts[g] as i64;
        aa += count * u * u;
        ab += count * u * v;
        bb += count * v * v;
        for c in 0..3 {
            let sum = prefix[bounds[g + 1]][c] - prefix[bounds[g]][c];
            ax[c] += u * sum * den;
            bx[c] += v * sum * den;
        }
    }
    let det = aa * bb - ab * ab;
    if det == 0 {
        return None;
    }
    let mut a = [0; 3];
    let mut b = [0; 3];
    for c in 0..3 {
        a[c] = ((ax[c] * bb - bx[c] * ab + det / 2) / det).clamp(0, 255) as i32;
        b[c] = ((bx[c] * aa - ax[c] * ab + det / 2) / det).clamp(0, 255) as i32;
    }
    Some((quant(a), quant(b)))
}
// Rank the unquantized least-squares solutions with prefix sums. The omitted
// sum of squared input colors is constant for every partition. Larger is better.
fn explained(
    prefix: &[[i64; 3]; 17],
    n: usize,
    cuts: (usize, usize, usize),
    three: bool,
) -> Option<i64> {
    let (i, j, k) = cuts;
    let (bounds, weights) = if three {
        ([0, i, j, n, n], [(2, 0), (1, 1), (0, 2), (0, 0)])
    } else {
        ([0, i, j, k, n], [(3, 0), (2, 1), (1, 2), (0, 3)])
    };
    let (mut aa, mut ab, mut bb) = (0i64, 0i64, 0i64);
    let (mut ax, mut bx) = ([0i64; 3], [0i64; 3]);
    for g in 0..4 {
        let (u, v) = weights[g];
        let count = (bounds[g + 1] - bounds[g]) as i64;
        aa += count * u * u;
        ab += count * u * v;
        bb += count * v * v;
        for c in 0..3 {
            let sum = prefix[bounds[g + 1]][c] - prefix[bounds[g]][c];
            ax[c] += u * sum;
            bx[c] += v * sum;
        }
    }
    let det = aa * bb - ab * ab;
    if det == 0 {
        return None;
    }
    let mut gain = 0;
    for c in 0..3 {
        gain += bb * ax[c] * ax[c] - 2 * ab * ax[c] * bx[c] + aa * bx[c] * bx[c];
    }
    Some(gain / det)
}

// Bounding-box range fit and a single least-squares refit, using integer
// arithmetic so native and wasm make identical decisions. See stb_dxt:
// https://github.com/nothings/stb/blob/master/stb_dxt.h
fn fast(px: &[[u8; 4]; 16], trans: Option<u8>, lo: [i32; 3], hi: [i32; 3]) -> ([u8; 8], u32) {
    let den = if trans.is_some() { 2i64 } else { 3 };
    let mut aa = 0i64;
    let mut ab = 0i64;
    let mut bb = 0i64;
    let mut ax = [0i64; 3];
    let mut bx = [0i64; 3];
    // Orient the bounding-box diagonal using covariance with the channel
    // having the greatest range. Opposing colour gradients need mixed signs.
    let dominant = (0..3).max_by_key(|&c| hi[c] - lo[c]).unwrap();
    let mut sums = [0i64; 3];
    let mut products = [0i64; 3];
    let mut count = 0i64;
    for p in px {
        if trans.is_some_and(|t| p[3] < t) {
            continue;
        }
        count += 1;
        for c in 0..3 {
            sums[c] += i64::from(p[c]);
            products[c] += i64::from(p[c]) * i64::from(p[dominant]);
        }
    }
    let mut start = lo;
    let mut end = hi;
    for c in 0..3 {
        if count * products[c] < sums[c] * sums[dominant] {
            start[c] = hi[c];
            end[c] = lo[c];
        }
    }
    let axis = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];
    let length: i64 = axis.iter().map(|&x| i64::from(x * x)).sum();
    for p in px {
        if trans.is_some_and(|t| p[3] < t) {
            continue;
        }
        let projection: i64 = (0..3)
            .map(|c| i64::from((i32::from(p[c]) - start[c]) * axis[c]))
            .sum();
        let u = ((projection * den + length / 2) / length).clamp(0, den);
        let v = den - u;
        aa += u * u;
        ab += u * v;
        bb += v * v;
        for c in 0..3 {
            ax[c] += u * i64::from(p[c]) * den;
            bx[c] += v * i64::from(p[c]) * den;
        }
    }
    let det = aa * bb - ab * ab;
    let (mut a, mut b) = (end, start);
    if det != 0 {
        for c in 0..3 {
            a[c] = ((ax[c] * bb - bx[c] * ab + det / 2) / det).clamp(0, 255) as i32;
            b[c] = ((bx[c] * aa - ax[c] * ab + det / 2) / det).clamp(0, 255) as i32;
        }
    }
    let (a, b) = (quant(a), quant(b));
    let (a, b) = if trans.is_some() {
        (a.min(b), a.max(b))
    } else if a == b {
        if a < u16::MAX {
            (a + 1, b)
        } else {
            (a, b - 1)
        }
    } else {
        (a.max(b), a.min(b))
    };
    let mut result = (u32::MAX, 0, 0, 0);
    evaluate(px, trans, a, b, &mut result);
    let mut out = [0u8; 8];
    out[..2].copy_from_slice(&a.to_le_bytes());
    out[2..4].copy_from_slice(&b.to_le_bytes());
    out[4..].copy_from_slice(&result.3.to_le_bytes());
    (out, result.0)
}

pub fn encode(px: &[[u8; 4]; 16], effort: Effort, policy: AlphaPolicy) -> [u8; 8] {
    let trans = match policy {
        AlphaPolicy::Opaque => None,
        AlphaPolicy::PunchThrough { threshold } => Some(threshold),
    };
    let mut best = (u32::MAX, 0u16, 0u16, 0u32);
    let mut order = [0usize; 16];
    let mut n = 0;
    for (i, p) in px.iter().enumerate() {
        if trans.is_none_or(|t| p[3] >= t) {
            order[n] = i;
            n += 1;
        }
    }
    if n == 0 {
        return [0, 0, 0, 0, 255, 255, 255, 255];
    }
    let mut lo = [255i32; 3];
    let mut hi = [0i32; 3];
    let mut sum = [0i64; 3];
    for &i in &order[..n] {
        for c in 0..3 {
            let v = i32::from(px[i][c]);
            lo[c] = lo[c].min(v);
            hi[c] = hi[c].max(v);
            sum[c] += i64::from(v);
        }
    }
    if matches!(effort, Effort::Fast | Effort::Balanced) && lo != hi {
        let (initial, error) = fast(px, trans, lo, hi);
        if effort == Effort::Fast {
            return initial;
        }
        let a = u16::from_le_bytes([initial[0], initial[1]]);
        let b = u16::from_le_bytes([initial[2], initial[3]]);
        best = (
            error,
            a,
            b,
            u32::from_le_bytes(initial[4..].try_into().unwrap()),
        );
        if best.0 > 16 {
            for endpoint in 0..2 {
                for channel in 0..3 {
                    let delta = if endpoint == 0 { -1 } else { 1 };
                    let mut x = components(if endpoint == 0 { a } else { b });
                    x[channel] += delta;
                    let limit = if channel == 1 { 63 } else { 31 };
                    if (0..=limit).contains(&x[channel]) {
                        let changed = pack(x);
                        if endpoint == 0 {
                            candidate(px, trans, changed, b, &mut best);
                        } else {
                            candidate(px, trans, a, changed, &mut best);
                        }
                    }
                }
            }
        }
        // P7 corpus sweep: the largest positive error bins have residual
        // above 1024 and a maximum channel span of 64..159.
        let span = (0..3).map(|c| hi[c] - lo[c]).max().unwrap();
        if best.0 > 1024 && (64..160).contains(&span) {
            for endpoint in 0..2 {
                for channel in 0..3 {
                    let delta = if endpoint == 0 { 1 } else { -1 };
                    let mut x = components(if endpoint == 0 { best.1 } else { best.2 });
                    x[channel] += delta;
                    let limit = if channel == 1 { 63 } else { 31 };
                    if (0..=limit).contains(&x[channel]) {
                        let (a, b) = (best.1, best.2);
                        if endpoint == 0 {
                            candidate(px, trans, pack(x), b, &mut best);
                        } else {
                            candidate(px, trans, a, pack(x), &mut best);
                        }
                    }
                }
            }
        }
        let mut out = [0u8; 8];
        out[..2].copy_from_slice(&best.1.to_le_bytes());
        out[2..4].copy_from_slice(&best.2.to_le_bytes());
        out[4..].copy_from_slice(&best.3.to_le_bytes());
        return out;
    }
    candidate(px, trans, quant(lo), quant(hi), &mut best);
    // Zero is a global lower bound, including exact two-colour blocks.
    if best.0 == 0 {
        let mut out = [0u8; 8];
        out[..2].copy_from_slice(&best.1.to_le_bytes());
        out[2..4].copy_from_slice(&best.2.to_le_bytes());
        out[4..].copy_from_slice(&best.3.to_le_bytes());
        return out;
    }
    // Minimise each channel for each selector and endpoint-order relation.
    // Combining the three relation classes preserves the packed 565 ordering.
    if lo == hi {
        // Every selector and endpoint relation is represented. Its per-channel
        // minimum gives the global minimum and the lexicographically smallest
        // endpoints at that error; rescore once to select the actual palette
        // entry (and preserve the existing selector and alpha-mask ties).
        let mut solid_best = (u32::MAX, 0u16, 0u16);
        for selector in SOLID[usize::from(trans.is_some())]
            .iter()
            .take(if trans.is_some() { 3 } else { 4 })
        {
            let options = [
                selector[0][lo[0] as usize],
                selector[1][lo[1] as usize],
                selector[0][lo[2] as usize],
            ];
            for r in 0..3 {
                for g in 0..3 {
                    for b in 0..3 {
                        let relations = [r, g, b];
                        let direction = relations.into_iter().find(|&v| v != 1).unwrap_or(1);
                        if (trans.is_some() && direction == 2)
                            || (trans.is_none() && direction != 2)
                        {
                            continue;
                        }
                        let error =
                            (options[0][r].0 + options[1][g].0 + options[2][b].0) as u32 * n as u32;
                        let a = [options[0][r].1, options[1][g].1, options[2][b].1];
                        let b = [options[0][r].2, options[1][g].2, options[2][b].2];
                        solid_best = solid_best.min((error, pack(a), pack(b)));
                    }
                }
            }
        }
        evaluate(px, trans, solid_best.1, solid_best.2, &mut best);
        let mut out = [0u8; 8];
        out[..2].copy_from_slice(&best.1.to_le_bytes());
        out[2..4].copy_from_slice(&best.2.to_le_bytes());
        out[4..].copy_from_slice(&best.3.to_le_bytes());
        return out;
    }
    let mut cov = [[0i64; 3]; 3];
    for &i in &order[..n] {
        let d = [
            i64::from(px[i][0]) * n as i64 - sum[0],
            i64::from(px[i][1]) * n as i64 - sum[1],
            i64::from(px[i][2]) * n as i64 - sum[2],
        ];
        for x in 0..3 {
            for y in 0..3 {
                cov[x][y] += d[x] * d[y];
            }
        }
    }
    let mut axis = [
        i64::from(hi[0] - lo[0]),
        i64::from(hi[1] - lo[1]),
        i64::from(hi[2] - lo[2]),
    ];
    for _ in 0..4 {
        let mut next = [0i64; 3];
        for x in 0..3 {
            for y in 0..3 {
                next[x] += cov[x][y] * axis[y];
            }
        }
        let scale = next.iter().map(|v| v.abs()).max().unwrap().max(1);
        for x in 0..3 {
            axis[x] = next[x] * 1024 / scale;
        }
    }
    order[..n].sort_unstable_by_key(|&i| {
        (
            (0..3).map(|c| i64::from(px[i][c]) * axis[c]).sum::<i64>(),
            i,
        )
    });
    let mut prefix = [[0i64; 3]; 17];
    for (j, &i) in order[..n].iter().enumerate() {
        for c in 0..3 {
            prefix[j + 1][c] = prefix[j][c] + i64::from(px[i][c]);
        }
    }
    // Ordered cluster fit: retain a short list by the continuous fit, then
    // rescore decoded RGB565 palettes against the original texels.
    let step = if effort == Effort::Quality { 1 } else { 4 };
    let mut shortlist = [(i64::MIN, (0usize, 0usize, 0usize)); 64];
    for i in (0..=n).step_by(step) {
        for j in (i..=n).step_by(step) {
            let ks = if trans.is_some() { n..=n } else { j..=n };
            for k in ks.step_by(step) {
                let cuts = (i, j, k);
                if effort == Effort::Fast {
                    if let Some((a, b)) = fit(&prefix, n, cuts, trans.is_some()) {
                        candidate(px, trans, a, b, &mut best);
                    }
                } else if let Some(gain) = explained(&prefix, n, cuts, trans.is_some()) {
                    let entry = (gain, cuts);
                    if entry > shortlist[63] {
                        let mut index = 63;
                        while index > 0 && entry > shortlist[index - 1] {
                            shortlist[index] = shortlist[index - 1];
                            index -= 1;
                        }
                        shortlist[index] = entry;
                    }
                }
            }
        }
    }
    if effort == Effort::Quality {
        for &(_, cuts) in &shortlist {
            if let Some((a, b)) = fit(&prefix, n, cuts, trans.is_some()) {
                candidate(px, trans, a, b, &mut best);
            }
        }
    }
    // Neighboring 565 grid points are the only rounding alternatives considered.
    // Exact scoring uses the accumulated squared error as a stopping bound.
    if effort == Effort::Quality && best.0 != 0 {
        for endpoint in 0..2 {
            for channel in 0..3 {
                let a = best.1;
                let b = best.2;
                let mut v = components(if endpoint == 0 { a } else { b });
                let original = v[channel];
                for delta in [-1, 1] {
                    v[channel] = original + delta;
                    if v[channel] < 0 || v[channel] > if channel == 1 { 63 } else { 31 } {
                        continue;
                    }
                    if endpoint == 0 {
                        candidate(px, trans, pack(v), b, &mut best);
                    } else {
                        candidate(px, trans, a, pack(v), &mut best);
                    }
                }
                if best.0 == 0 {
                    break;
                }
            }
        }
    }
    let mut out = [0u8; 8];
    out[..2].copy_from_slice(&best.1.to_le_bytes());
    out[2..4].copy_from_slice(&best.2.to_le_bytes());
    out[4..].copy_from_slice(&best.3.to_le_bytes());
    out
}
