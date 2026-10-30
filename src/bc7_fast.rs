//! Fast BC7 search.
//!
//! Each subset gets one principal-axis endpoint fit. Selectors come from
//! projecting each colour onto the quantized endpoint line and checking the
//! neighbouring palette entries exactly, and a short least-squares pass
//! refines the endpoints. Fits score distinct colours with their pixel
//! counts, so blocks with few colours cost little; those blocks, and grey
//! blocks, also get endpoint searches that matter at low error. Partitioned
//! modes run only for blocks whose error stays high, on the partitions whose
//! estimated residual is lowest. Every reported error is the exact decoded SSE.
#![allow(
    clippy::needless_range_loop,
    reason = "channel loops index several per-channel arrays over const-generic channel ranges"
)]

use super::tables::ANCHOR2;
use super::{put, solid_rgba, MASKS2};

#[path = "bc7_search.rs"]
mod search;

#[path = "bc7_objective.rs"]
mod objective;
pub(super) use search::{encode_with, BALANCED, QUALITY};

#[cfg(feature = "simd")]
#[path = "bc7_simd/mod.rs"]
pub(super) mod simd;

type Block = [[u8; 4]; 16];
type Pair = [[i32; 4]; 2];

const W2: [i32; 4] = [0, 21, 43, 64];
const W3: [i32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const W4: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

// Nearest selector for each interpolation weight 0..=64.
const fn nearest<const N: usize>(w: [i32; N]) -> [u8; 65] {
    let mut out = [0u8; 65];
    let mut k = 0;
    while k <= 64 {
        let mut best = 0;
        let mut j = 1;
        while j < N {
            if (w[j] - k as i32).abs() < (w[best] - k as i32).abs() {
                best = j;
            }
            j += 1;
        }
        out[k] = best as u8;
        k += 1;
    }
    out
}

struct Levels {
    w: &'static [i32],
    near: [u8; 65],
}
static L2: Levels = Levels {
    w: &W2,
    near: nearest(W2),
};
static L3: Levels = Levels {
    w: &W3,
    near: nearest(W3),
};
static L4: Levels = Levels {
    w: &W4,
    near: nearest(W4),
};

#[derive(Clone, Copy, PartialEq)]
enum PBit {
    None,
    Unique,
    Shared,
}

// Endpoint precision: stored bits per channel including any p-bit.
#[derive(Clone, Copy)]
struct Prec {
    bits: usize,
    pbit: PBit,
}

const P1: Prec = Prec {
    bits: 7,
    pbit: PBit::Shared,
};
const P3: Prec = Prec {
    bits: 8,
    pbit: PBit::Unique,
};
const P4C: Prec = Prec {
    bits: 5,
    pbit: PBit::None,
};
const P4A: Prec = Prec {
    bits: 6,
    pbit: PBit::None,
};
const P5C: Prec = Prec {
    bits: 7,
    pbit: PBit::None,
};
const P5A: Prec = Prec {
    bits: 8,
    pbit: PBit::None,
};
const P6: Prec = P3;
const P7: Prec = Prec {
    bits: 6,
    pbit: PBit::Unique,
};

const fn expand(x: u32, n: u32) -> u32 {
    (x << (8 - n)) | (x >> (2 * n - 8))
}

// QUANT[n][p][t]: the decodable byte nearest to t for n stored bits; p is the
// required low bit of the stored value, or 2 for none.
const fn quant_tables() -> [[[u8; 256]; 3]; 9] {
    let mut out = [[[0u8; 256]; 3]; 9];
    let mut n = 4u32;
    while n <= 8 {
        let max = (1u32 << n) - 1;
        let mut p = 0;
        while p < 3 {
            let mut t = 0u32;
            while t < 256 {
                let guess = (t * max + 127) / 255;
                let mut x = guess.saturating_sub(2);
                let mut best = 0u32;
                let mut distance = u32::MAX;
                while x <= max && x <= guess + 2 {
                    if p == 2 || (x & 1) == p as u32 {
                        let v = expand(x, n);
                        let d = v.abs_diff(t);
                        if d < distance {
                            distance = d;
                            best = v;
                        }
                    }
                    x += 1;
                }
                out[n as usize][p][t as usize] = best as u8;
                t += 1;
            }
            p += 1;
        }
        n += 1;
    }
    out
}
static QUANT: [[[u8; 256]; 3]; 9] = quant_tables();

// The pixels a fit scores: one representative pixel per distinct colour with
// its multiplicity, and each pixel's representative.
#[derive(Clone, Copy)]
struct Reps {
    list: [(u8, u8); 16],
    len: usize,
    of: [u8; 16],
    expand: bool,
}

impl Reps {
    // Identical RGBA pixels share one representative.
    fn distinct(px: &Block) -> Self {
        let mut r = Reps {
            list: [(0, 0); 16],
            len: 0,
            of: [0; 16],
            expand: false,
        };
        let keys = px.map(u32::from_le_bytes);
        for (i, &key) in keys.iter().enumerate() {
            match r.list[..r.len]
                .iter_mut()
                .find(|(j, _)| keys[usize::from(*j)] == key)
            {
                Some(slot) => {
                    r.expand = true;
                    slot.1 += 1;
                    r.of[i] = slot.0;
                }
                None => {
                    r.list[r.len] = (i as u8, 1);
                    r.of[i] = i as u8;
                    r.len += 1;
                }
            }
        }
        r
    }

    // Every pixel of the subset is its own representative.
    fn subset(mask: u16) -> Self {
        let mut r = Reps {
            list: [(0, 0); 16],
            len: 0,
            of: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            expand: false,
        };
        let mut m = mask;
        while m != 0 {
            r.list[r.len] = (m.trailing_zeros() as u8, 1);
            r.len += 1;
            m &= m - 1;
        }
        r
    }

    // Restrict full-block colour representatives to a subset. A representative
    // may lie outside the subset, but has the same RGBA value as every pixel
    // counted for it; selectors are expanded through the original mapping.
    fn masked(&self, mask: u16) -> Self {
        if !self.expand {
            return Self::subset(mask);
        }
        let mut r = *self;
        let mut counts = [0u8; 16];
        let mut m = mask;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            counts[usize::from(self.of[i])] += 1;
            m &= m - 1;
        }
        r.len = 0;
        for &(i, _) in &self.list[..self.len] {
            let count = counts[usize::from(i)];
            if count != 0 {
                r.list[r.len] = (i, count);
                r.len += 1;
            }
        }
        r
    }

    fn iter(&self) -> impl Iterator<Item = (usize, i32)> + '_ {
        self.list[..self.len]
            .iter()
            .map(|&(i, n)| (usize::from(i), i32::from(n)))
    }
}

// Quantize endpoint targets (0..=255) over channels C0..C1. A forced p-bit of
// one keeps opaque alpha exactly 255 in the modes that store alpha.
fn quantize<const C0: usize, const C1: usize>(t: &Pair, prec: Prec, force_p1: bool) -> Pair {
    let table = &QUANT[prec.bits];
    let quant_err = |k: usize, p: usize| -> i32 {
        let mut e = 0;
        for c in C0..C1 {
            let v = t[k][c];
            let q = i32::from(table[p][v as usize]);
            e += (v - q) * (v - q);
        }
        e
    };
    let mut choice = [2usize; 2];
    match prec.pbit {
        PBit::None => {}
        PBit::Unique => {
            for (k, p) in choice.iter_mut().enumerate() {
                *p = usize::from(force_p1 || quant_err(k, 1) < quant_err(k, 0));
            }
        }
        PBit::Shared => {
            let p = usize::from(
                force_p1 || quant_err(0, 1) + quant_err(1, 1) < quant_err(0, 0) + quant_err(1, 0),
            );
            choice = [p, p];
        }
    }
    quantize_with::<C0, C1>(t, prec, choice)
}

// Quantize endpoint targets with the given low bit per endpoint (2 for none).
fn quantize_with<const C0: usize, const C1: usize>(
    t: &Pair,
    prec: Prec,
    choice: [usize; 2],
) -> Pair {
    let table = &QUANT[prec.bits];
    let mut d = [[255i32; 4]; 2];
    for k in 0..2 {
        for c in C0..C1 {
            d[k][c] = i32::from(table[choice[k]][t[k][c] as usize]);
        }
    }
    d
}

#[derive(Clone, Copy)]
struct Fit {
    err: u32,
    d: Pair,
    idx: [u8; 16],
}

// Exact decoded SSE of the best selector near each colour's projection,
// stopping early once `limit` is reached (selectors are then incomplete).
fn select<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    d: &Pair,
    lv: &Levels,
    idx: &mut [u8; 16],
    limit: u32,
) -> u32 {
    #[cfg(feature = "simd")]
    if let Some(err) = simd::select::<C0, C1>(px, reps, d, lv, idx, limit, None) {
        return err;
    }
    select_scalar::<C0, C1>(px, reps, d, lv, idx, limit)
}

#[inline]
fn select_scalar<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    d: &Pair,
    lv: &Levels,
    idx: &mut [u8; 16],
    limit: u32,
) -> u32 {
    match lv.w.len() {
        4 => select_palette::<C0, C1, 4>(px, reps, d, lv, idx, limit),
        8 => select_palette::<C0, C1, 8>(px, reps, d, lv, idx, limit),
        16 => select_palette::<C0, C1, 16>(px, reps, d, lv, idx, limit),
        _ => unreachable!(),
    }
}

fn select_palette<const C0: usize, const C1: usize, const N: usize>(
    px: &Block,
    reps: &Reps,
    d: &Pair,
    lv: &Levels,
    idx: &mut [u8; 16],
    limit: u32,
) -> u32 {
    let n = N;
    let mut pal = [[0i32; 4]; N];
    for (j, entry) in pal.iter_mut().enumerate() {
        let w = lv.w[j];
        for c in C0..C1 {
            entry[c] = ((64 - w) * d[0][c] + w * d[1][c] + 32) >> 6;
        }
    }
    let mut dir = [0i32; 4];
    let mut len2 = 0;
    for c in C0..C1 {
        dir[c] = d[1][c] - d[0][c];
        len2 += dir[c] * dir[c];
    }
    let scale = if len2 == 0 { 0.0 } else { 64.0 / len2 as f32 };
    let mut err = 0u32;
    for (i, count) in reps.iter() {
        let p = &px[i];
        let mut t = 0;
        for c in C0..C1 {
            t += (i32::from(p[c]) - d[0][c]) * dir[c];
        }
        let k = ((t as f32 * scale + 0.5) as i32).clamp(0, 64) as usize;
        let j = usize::from(lv.near[k]);
        let mut best = (i32::MAX, 0usize);
        for (jj, entry) in pal
            .iter()
            .enumerate()
            .take((j + 2).min(n))
            .skip(j.saturating_sub(1))
        {
            let mut e = 0;
            for c in C0..C1 {
                let x = i32::from(p[c]) - entry[c];
                e += x * x;
            }
            if e < best.0 {
                best = (e, jj);
            }
        }
        idx[i] = best.1 as u8;
        err += (best.0 * count) as u32;
        if err >= limit {
            return err;
        }
    }
    if reps.expand {
        for i in 0..16 {
            idx[i] = idx[usize::from(reps.of[i])];
        }
    }
    err
}

// Endpoint targets on the principal axis of the colours, spanning the extreme
// projections. Power iteration starts from the covariance column of the most
// variable channel.
fn axis_targets<const C0: usize, const C1: usize>(px: &Block, reps: &Reps) -> Pair {
    let mut n = 0i32;
    let mut s = [0i32; 4];
    let mut prod = [[0i32; 4]; 4];
    for (i, count) in reps.iter() {
        n += count;
        for c in C0..C1 {
            let x = i32::from(px[i][c]);
            s[c] += x * count;
            for e in c..C1 {
                prod[c][e] += x * i32::from(px[i][e]) * count;
            }
        }
    }
    let mut cov = [[0f32; 4]; 4];
    for c in C0..C1 {
        for e in c..C1 {
            let v = (n * prod[c][e] - s[c] * s[e]) as f32;
            cov[c][e] = v;
            cov[e][c] = v;
        }
    }
    let mut mean = [0f32; 4];
    for c in C0..C1 {
        mean[c] = s[c] as f32 / n as f32;
    }
    axis_endpoints::<C0, C1>(px, reps, &mean, &cov)
}

// Project onto an axis from previously computed moments. Covariance uses
// the unnormalized integer moment scale, matching fresh subset fits exactly.
fn axis_endpoints<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    mean: &[f32; 4],
    cov: &[[f32; 4]; 4],
) -> Pair {
    let mut top = C0;
    for c in C0..C1 {
        if cov[c][c] > cov[top][top] {
            top = c;
        }
    }
    let mut t = [[0i32; 4]; 2];
    if cov[top][top] <= 0.0 {
        for c in C0..C1 {
            t[0][c] = (mean[c] + 0.5) as i32;
            t[1][c] = t[0][c];
        }
        return t;
    }
    let mut v = [0f32; 4];
    for c in C0..C1 {
        v[c] = cov[c][top];
    }
    for _ in 0..3 {
        let mut next = [0f32; 4];
        let mut largest = 0f32;
        for c in C0..C1 {
            for e in C0..C1 {
                next[c] += cov[c][e] * v[e];
            }
            largest = largest.max(next[c].abs());
        }
        if largest <= 0.0 {
            break;
        }
        for c in C0..C1 {
            v[c] = next[c] / largest;
        }
    }
    let mut len2 = 0f32;
    for c in C0..C1 {
        len2 += v[c] * v[c];
    }
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for (i, _) in reps.iter() {
        let mut d = 0f32;
        for c in C0..C1 {
            d += (f32::from(px[i][c]) - mean[c]) * v[c];
        }
        lo = lo.min(d);
        hi = hi.max(d);
    }
    for c in C0..C1 {
        let u = v[c] / len2;
        t[0][c] = ((mean[c] + lo * u + 0.5) as i32).clamp(0, 255);
        t[1][c] = ((mean[c] + hi * u + 0.5) as i32).clamp(0, 255);
    }
    t
}

// Least-squares endpoints for fixed selectors.
fn least_squares<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    idx: &[u8; 16],
    lv: &Levels,
) -> Option<Pair> {
    let mut aa = 0i64;
    let mut ab = 0i64;
    let mut bb = 0i64;
    let mut ax = [0i64; 4];
    let mut bx = [0i64; 4];
    for (i, count) in reps.iter() {
        let count = i64::from(count);
        let b = i64::from(lv.w[usize::from(idx[i])]);
        let a = 64 - b;
        aa += a * a * count;
        ab += a * b * count;
        bb += b * b * count;
        for c in C0..C1 {
            let x = i64::from(px[i][c]) * count;
            ax[c] += a * x;
            bx[c] += b * x;
        }
    }
    let det = aa * bb - ab * ab;
    if det == 0 {
        return None;
    }
    // Products stay below 2^53, so the f64 numerators are exact; one
    // reciprocal replaces a division per endpoint channel.
    let scale = 64.0 / det as f64;
    let round = |v: i64| ((v as f64 * scale).clamp(0.0, 255.0) + 0.5) as i32;
    let mut t = [[0i32; 4]; 2];
    for c in C0..C1 {
        t[0][c] = round(ax[c] * bb - bx[c] * ab);
        t[1][c] = round(bx[c] * aa - ax[c] * ab);
    }
    Some(t)
}

// With selectors fixed, each channel's error depends only on its two endpoint
// values, so each channel tries moving either endpoint to a neighbouring code
// (p-bits unchanged), then selectors are chosen again. Returns whether the
// error fell.
fn polish<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    f: &mut Fit,
    prec: Prec,
    lv: &Levels,
) -> bool {
    #[cfg(feature = "simd")]
    if let Some(changed) = simd::polish::<C0, C1>(px, reps, f, prec, lv, None) {
        return changed;
    }
    let n = prec.bits as u32;
    let step = if prec.pbit == PBit::None { 1 } else { 2 };
    let max = (1i32 << n) - 1;
    let mut fixed = 0u32;
    let mut d = f.d;
    for c in C0..C1 {
        let x0 = d[0][c] >> (8 - n);
        let x1 = d[1][c] >> (8 - n);
        let mut endpoints = [[d[0][c], d[1][c]]; 5];
        for (j, (a, b)) in [
            (x0 - step, x1),
            (x0 + step, x1),
            (x0, x1 - step),
            (x0, x1 + step),
        ]
        .into_iter()
        .enumerate()
        {
            if (0..=max).contains(&a) && (0..=max).contains(&b) {
                endpoints[j + 1] = [expand(a as u32, n) as i32, expand(b as u32, n) as i32];
            }
        }
        // The five independent sums share selector, source and multiplicity
        // loads. Invalid moves repeat the centre and cannot win a strict tie.
        let mut errors = [0u32; 5];
        for (i, count) in reps.iter() {
            let w = lv.w[usize::from(f.idx[i])];
            let source = i32::from(px[i][c]);
            for j in 0..5 {
                let [a, b] = endpoints[j];
                let x = source - (((64 - w) * a + w * b + 32) >> 6);
                errors[j] += (x * x * count) as u32;
            }
        }
        let mut best = 0;
        for j in 1..5 {
            if errors[j] < errors[best] {
                best = j;
            }
        }
        fixed += errors[best];
        d[0][c] = endpoints[best][0];
        d[1][c] = endpoints[best][1];
    }
    if fixed >= f.err {
        return false;
    }
    let mut idx = f.idx;
    let err = select::<C0, C1>(px, reps, &d, lv, &mut idx, u32::MAX);
    if err <= fixed {
        *f = Fit { err, d, idx };
    } else {
        f.err = fixed;
        f.d = d;
    }
    true
}

// Move both endpoints by whole lattice steps along the endpoint axis, keeping
// the neighbour with the lowest exact error, for up to `rounds` moves.
fn axis_search<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    f: &mut Fit,
    lv: &Levels,
    step: i32,
    rounds: usize,
) {
    let mut span = 0;
    for c in C0..C1 {
        span = span.max((f.d[1][c] - f.d[0][c]).abs());
    }
    if span == 0 {
        return;
    }
    let mut unit = [0i32; 4];
    for c in C0..C1 {
        let dir = f.d[1][c] - f.d[0][c];
        if 2 * dir.abs() >= span {
            unit[c] = step * dir.signum();
        }
    }
    let mut scratch = [0u8; 16];
    for _ in 0..rounds {
        let centre = f.d;
        let mut best = (f.err, centre);
        let mut best_idx = f.idx;
        for (a, b) in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ] {
            let mut d = centre;
            let mut valid = true;
            for c in C0..C1 {
                d[0][c] += a * unit[c];
                d[1][c] += b * unit[c];
                valid &= (0..=255).contains(&d[0][c]) && (0..=255).contains(&d[1][c]);
            }
            if !valid {
                continue;
            }
            let e = select::<C0, C1>(px, reps, &d, lv, &mut scratch, best.0);
            if e < best.0 {
                best = (e, d);
                best_idx = scratch;
            }
        }
        if best.0 == f.err {
            break;
        }
        f.d = best.1;
        // An improving bounded select completed every representative and
        // expanded duplicate selectors, so its indices are already final.
        f.err = best.0;
        f.idx = best_idx;
    }
}

fn fit<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    prec: Prec,
    lv: &Levels,
    iters: usize,
    rounds: usize,
    force_p1: bool,
) -> Fit {
    let d = quantize::<C0, C1>(&axis_targets::<C0, C1>(px, reps), prec, force_p1);
    let mut best = Fit {
        err: 0,
        d,
        idx: [0; 16],
    };
    best.err = select::<C0, C1>(px, reps, &d, lv, &mut best.idx, u32::MAX);
    improve::<C0, C1>(px, reps, &mut best, prec, lv, iters, rounds, force_p1);
    best
}

// Up to `iters` least-squares passes while the error falls, then up to
// `rounds` polish passes.
#[allow(
    clippy::too_many_arguments,
    reason = "the fit parameters are independent knobs shared by every mode"
)]
fn improve<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    best: &mut Fit,
    prec: Prec,
    lv: &Levels,
    iters: usize,
    rounds: usize,
    force_p1: bool,
) {
    for _ in 0..iters {
        if best.err == 0 {
            break;
        }
        let Some(t) = least_squares::<C0, C1>(px, reps, &best.idx, lv) else {
            break;
        };
        let d = quantize::<C0, C1>(&t, prec, force_p1);
        if d == best.d {
            break;
        }
        let mut idx = best.idx;
        let err = select::<C0, C1>(px, reps, &d, lv, &mut idx, best.err);
        if err >= best.err {
            break;
        }
        *best = Fit { err, d, idx };
    }
    for _ in 0..rounds {
        if best.err == 0 || !polish::<C0, C1>(px, reps, best, prec, lv) {
            break;
        }
    }
}

// Flip a subset so its anchor selector has a clear high bit.
fn anchor(f: &mut Fit, mask: u16, at: usize, levels: usize) {
    if usize::from(f.idx[at]) >= levels / 2 {
        f.d.swap(0, 1);
        let mut m = mask;
        while m != 0 {
            let i = m.trailing_zeros() as usize;
            m &= m - 1;
            f.idx[i] = (levels - 1 - usize::from(f.idx[i])) as u8;
        }
    }
}

fn pack6(mut f: Fit) -> [u8; 16] {
    anchor(&mut f, 0xFFFF, 0, 16);
    let mut bits = 0u128;
    let mut o = 0;
    put(&mut bits, &mut o, 64, 7);
    for c in 0..4 {
        for e in &f.d {
            put(&mut bits, &mut o, (e[c] >> 1) as u32, 7);
        }
    }
    for e in &f.d {
        put(&mut bits, &mut o, (e[0] & 1) as u32, 1);
    }
    for (i, &x) in f.idx.iter().enumerate() {
        put(&mut bits, &mut o, u32::from(x), if i == 0 { 3 } else { 4 });
    }
    debug_assert_eq!(o, 128);
    bits.to_le_bytes()
}

// Mode 5, or mode 4 with index selection `sel`: separate colour and scalar
// selector streams. The fits describe the block after `rotation` moved one
// colour channel into the scalar slot.
fn pack_dual(mode: usize, rotation: usize, sel: usize, mut color: Fit, mut alpha: Fit) -> [u8; 16] {
    let (cw, aw) = if mode == 5 { (7, 8) } else { (5, 6) };
    // Mode 4 uses 2-bit colour and 3-bit alpha selectors unless sel swaps them.
    let (cl, al) = match (mode, sel) {
        (5, _) => (4, 4),
        (_, 0) => (4, 8),
        _ => (8, 4),
    };
    anchor(&mut color, 0xFFFF, 0, cl);
    anchor(&mut alpha, 0xFFFF, 0, al);
    let mut bits = 0u128;
    let mut o = 0;
    if mode == 5 {
        put(&mut bits, &mut o, 32, 6);
        put(&mut bits, &mut o, rotation as u32, 2);
    } else {
        put(&mut bits, &mut o, 16, 5);
        put(&mut bits, &mut o, rotation as u32, 2);
        put(&mut bits, &mut o, sel as u32, 1);
    }
    for c in 0..4 {
        let (src, w) = if c == 3 { (&alpha, aw) } else { (&color, cw) };
        for e in &src.d {
            put(&mut bits, &mut o, (e[c] >> (8 - w)) as u32, w);
        }
    }
    // The first stream is the 2-bit one.
    let streams = if mode == 4 && sel == 1 {
        [(&alpha, al), (&color, cl)]
    } else {
        [(&color, cl), (&alpha, al)]
    };
    for (f, levels) in streams {
        let width = if levels == 8 { 3 } else { 2 };
        for (i, &x) in f.idx.iter().enumerate() {
            put(&mut bits, &mut o, u32::from(x), width - u32::from(i == 0));
        }
    }
    debug_assert_eq!(o, 128);
    bits.to_le_bytes()
}

// Two-subset modes 1, 3 and 7.
fn pack2(mode: usize, partition: usize, mut s: [Fit; 2]) -> [u8; 16] {
    let m1 = MASKS2[partition][0];
    let masks = [!m1, m1];
    let levels = if mode == 1 { 8 } else { 4 };
    let width = [0, 6, 0, 7, 0, 0, 0, 5][mode];
    let channels = if mode == 7 { 4 } else { 3 };
    let anchors = [0, ANCHOR2[partition]];
    for k in 0..2 {
        anchor(&mut s[k], masks[k], anchors[k], levels);
    }
    let mut bits = 0u128;
    let mut o = 0;
    put(&mut bits, &mut o, 1 << mode, (mode + 1) as u32);
    put(&mut bits, &mut o, partition as u32, 6);
    for c in 0..channels {
        for f in &s {
            for e in &f.d {
                put(
                    &mut bits,
                    &mut o,
                    (e[c] >> (8 - width)) as u32,
                    width as u32,
                );
            }
        }
    }
    for f in &s {
        if mode == 1 {
            put(
                &mut bits,
                &mut o,
                ((f.d[0][0] >> (7 - width)) & 1) as u32,
                1,
            );
        } else {
            for e in &f.d {
                put(&mut bits, &mut o, ((e[0] >> (7 - width)) & 1) as u32, 1);
            }
        }
    }
    let ib = if levels == 8 { 3 } else { 2 };
    for i in 0..16 {
        let k = usize::from(m1 >> i & 1 == 1);
        put(
            &mut bits,
            &mut o,
            u32::from(s[k].idx[i]),
            ib - u32::from(anchors[k] == i),
        );
    }
    debug_assert_eq!(o, 128);
    bits.to_le_bytes()
}

// Count, channel sums, then the upper triangle of channel products.
#[derive(Clone, Copy, Default)]
struct Moments {
    m: [i32; 15],
}

const fn moment_len(ch: usize) -> usize {
    1 + ch + ch * (ch + 1) / 2
}

// Per-row prefix moments for every 4-pixel subset of each row.
fn moment_rows<const CH: usize>(px: &Block) -> [[Moments; 16]; 4] {
    let mut lut = [[Moments::default(); 16]; 4];
    for (row, table) in lut.iter_mut().enumerate() {
        for mask in 1usize..16 {
            let p = &px[row * 4 + mask.trailing_zeros() as usize];
            let mut m = table[mask & (mask - 1)];
            m.m[0] += 1;
            let mut k = 1 + CH;
            for c in 0..CH {
                let x = i32::from(p[c]);
                m.m[1 + c] += x;
                for e in c..CH {
                    m.m[k] += x * i32::from(p[e]);
                    k += 1;
                }
            }
            table[mask] = m;
        }
    }
    lut
}

// Off-axis residual and principal variance of one subset, in SSE units.
fn subset_estimate<const CH: usize>(m: &Moments) -> (f32, f32) {
    let n = m.m[0];
    let mut cov = [[0f32; 4]; 4];
    let mut trace = 0f32;
    let mut top = 0;
    let mut k = 1 + CH;
    for c in 0..CH {
        for e in c..CH {
            let v = (n * m.m[k] - m.m[1 + c] * m.m[1 + e]) as f32;
            cov[c][e] = v;
            cov[e][c] = v;
            k += 1;
        }
        trace += cov[c][c];
        if cov[c][c] > cov[top][top] {
            top = c;
        }
    }
    if trace <= 0.0 {
        return (0.0, 0.0);
    }
    // The Rayleigh quotient of w = C e_top is a lower bound on the largest
    // eigenvalue, close to it after this implicit power step.
    let mut w = [0f32; 4];
    for c in 0..CH {
        w[c] = cov[c][top];
    }
    let mut ww = 0f32;
    let mut wcw = 0f32;
    for c in 0..CH {
        let mut cw = 0f32;
        for e in 0..CH {
            cw += cov[c][e] * w[e];
        }
        ww += w[c] * w[c];
        wcw += w[c] * cw;
    }
    let lambda = (wcw / ww).min(trace);
    let nf = n as f32;
    ((trace - lambda) / nf, lambda / nf)
}

// Estimated (residual, principal variance) of every two-subset partition.
fn partition_estimates<const CH: usize>(px: &Block) -> [(f32, f32); 64] {
    let len = moment_len(CH);
    let lut = moment_rows::<CH>(px);
    let mut total = Moments::default();
    for row in &lut {
        for k in 0..len {
            total.m[k] += row[15].m[k];
        }
    }
    let mut out = [(0f32, 0f32); 64];
    for (p, slot) in out.iter_mut().enumerate() {
        let mask = MASKS2[p][0];
        let mut one = Moments::default();
        for (row, table) in lut.iter().enumerate() {
            let part = &table[usize::from((mask >> (4 * row)) & 15)];
            for k in 0..len {
                one.m[k] += part.m[k];
            }
        }
        let mut zero = total;
        for k in 0..len {
            zero.m[k] -= one.m[k];
        }
        let a = subset_estimate::<CH>(&zero);
        let b = subset_estimate::<CH>(&one);
        *slot = (a.0 + b.0, a.1 + b.1);
    }
    out
}

// The `count` partitions with the lowest estimate for a mode with `levels`
// selectors per subset, best first.
fn best_partitions(est: &[(f32, f32); 64], levels: f32, count: usize) -> [(f32, usize); 16] {
    let step = 1.0 / ((levels - 1.0) * (levels - 1.0));
    let mut top = [(f32::MAX, 0usize); 16];
    for (p, &(res, var)) in est.iter().enumerate() {
        let e = res + var * step;
        if e < top[count - 1].0 {
            let mut k = count - 1;
            while k > 0 && e < top[k - 1].0 {
                top[k] = top[k - 1];
                k -= 1;
            }
            top[k] = (e, p);
        }
    }
    top
}

fn fit_partition(
    px: &Block,
    mode: usize,
    partition: usize,
    opaque: bool,
    iters: usize,
    rounds: usize,
) -> [Fit; 2] {
    let m1 = MASKS2[partition][0];
    [!m1, m1].map(|mask| {
        let reps = Reps::subset(mask);
        match mode {
            1 => fit::<0, 3>(px, &reps, P1, &L3, iters, rounds, false),
            3 => fit::<0, 3>(px, &reps, P3, &L2, iters, rounds, false),
            _ => {
                let mut f = fit::<0, 4>(px, &reps, P7, &L2, iters, rounds, opaque);
                if opaque {
                    f.d[0][3] = 255;
                    f.d[1][3] = 255;
                }
                f
            }
        }
    })
}

// Error at or below which a refinable block searches mode 6 endpoints.
const AXIS_GATE: u32 = 256;
// Opaque mode 6 error at or below which a rotated mode 5 is also tried.
const ROTATION_GATE: u32 = 128;
// Error above which a block with varying alpha also tries mode 4.
const MODE4_GATE: u32 = 256;
// Distinct colours at or below which a block is refined.
const FEW_COLORS: usize = 8;
// Error above which a block also searches partitions.
const PARTITION_GATE: u32 = 512;

fn mode6(px: &Block, reps: &Reps, opaque: bool, refine: bool) -> Fit {
    if opaque {
        let mut f = fit::<0, 3>(px, reps, P6, &L4, 2, 1, true);
        if refine && f.err > 0 && f.err <= AXIS_GATE {
            axis_search::<0, 3>(px, reps, &mut f, &L4, 2, 4);
        }
        f.d[0][3] = 255;
        f.d[1][3] = 255;
        f
    } else {
        // Blocks with alpha also try modes 5 and 4, which recover more than
        // this search would for its cost.
        fit::<0, 4>(px, reps, P6, &L4, 2, 1, false)
    }
}

// Mode 5 (sel unused) or mode 4 with index selection `sel`.
fn dual(px: &Block, reps: &Reps, mode: usize, sel: usize, rounds: usize) -> (Fit, Fit) {
    if mode == 5 {
        (
            fit::<0, 3>(px, reps, P5C, &L2, 2, rounds, false),
            fit::<3, 4>(px, reps, P5A, &L2, 2, rounds, false),
        )
    } else {
        let (cl, al): (&Levels, &Levels) = if sel == 0 { (&L2, &L3) } else { (&L3, &L2) };
        (
            fit::<0, 3>(px, reps, P4C, cl, 2, rounds, false),
            fit::<3, 4>(px, reps, P4A, al, 2, rounds, false),
        )
    }
}

pub(super) fn encode(px: &Block) -> [u8; 16] {
    if let Some(block) = solid_rgba(px) {
        return block;
    }
    let opaque = px.iter().all(|p| p[3] == 255);
    let reps = Reps::distinct(px);
    let gray = px.iter().all(|p| p[0] == p[1] && p[1] == p[2]);
    let refine = reps.len <= FEW_COLORS || gray;
    let m6 = mode6(px, &reps, opaque, refine);
    let mut best_err = m6.err;
    let mut best = pack6(m6);
    if best_err == 0 {
        return best;
    }
    let rounds = usize::from(refine);
    if !opaque {
        // Mode 4 spends its extra selector precision on alpha or colour; it
        // runs only when alpha varies (mode 5 represents constant alpha
        // exactly) and the error so far is still high.
        let alpha_varies = px.iter().any(|p| p[3] != px[0][3]);
        for (mode, sel) in [(5, 0), (4, 0), (4, 1)] {
            if mode == 4 && (!alpha_varies || best_err <= MODE4_GATE) {
                continue;
            }
            let (c, a) = dual(px, &reps, mode, sel, rounds);
            if c.err + a.err < best_err {
                best_err = c.err + a.err;
                best = pack_dual(mode, 0, sel, c, a);
            }
        }
    } else if best_err <= ROTATION_GATE {
        // Opaque mode 6 keeps alpha at 255 only with odd endpoints. Mode 5
        // with the widest colour channel in the scalar slot reaches even
        // values, such as black, exactly.
        let mut lo = [255u8; 3];
        let mut hi = [0u8; 3];
        for p in px {
            for c in 0..3 {
                lo[c] = lo[c].min(p[c]);
                hi[c] = hi[c].max(p[c]);
            }
        }
        let r = (0..3).fold(0, |b, c| if hi[c] - lo[c] > hi[b] - lo[b] { c } else { b });
        let mut rotated = *px;
        for p in &mut rotated {
            p.swap(r, 3);
        }
        let (c, a) = dual(&rotated, &reps, 5, 0, rounds);
        if c.err + a.err < best_err {
            best_err = c.err + a.err;
            best = pack_dual(5, r + 1, 0, c, a);
        }
    }
    if best_err > PARTITION_GATE {
        let est = if opaque {
            partition_estimates::<3>(px)
        } else {
            partition_estimates::<4>(px)
        };
        let modes: &[(usize, f32)] = if opaque {
            &[(1, 8.0), (3, 4.0)]
        } else {
            &[(7, 4.0)]
        };
        for &(mode, levels) in modes {
            let (e, partition) = best_partitions(&est, levels, 1)[0];
            if e >= best_err as f32 {
                continue;
            }
            let s = fit_partition(px, mode, partition, opaque, 1, 0);
            let err = s[0].err + s[1].err;
            if err < best_err {
                best_err = err;
                best = pack2(mode, partition, s);
            }
        }
    }
    best
}

#[cfg(feature = "simd")]
pub(super) fn encode_batch(blocks: &[Block; 8], out: &mut [u8], effort: crate::Effort) {
    // Solid blocks return before using a seed, and nonopaque blocks use
    // the separate visibility-aware fit. Avoid work for batches where no
    // live block can consume the opaque seed.
    let wants_seed = blocks[..out.len() / 16]
        .iter()
        .any(|block| block.iter().all(|p| p[3] == 255) && solid_rgba(block).is_none());
    let seeds = if wants_seed {
        simd::seed_batch(blocks)
    } else {
        None
    };
    let cfg = if effort == crate::Effort::Balanced {
        &BALANCED
    } else {
        &QUALITY
    };
    for (slot, dst) in out.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        dst.copy_from_slice(&search::encode_seeded(
            &blocks[slot],
            cfg,
            seeds.as_ref().map(|t| t[slot]),
        ));
    }
}

#[cfg(feature = "simd")]
pub(super) fn batch_width() -> usize {
    simd::width()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded_sse(px: &Block, block: &[u8; 16]) -> u64 {
        let d = super::super::independent_decoder::decode(block);
        let mut e = 0u64;
        for (a, b) in px.iter().zip(d.iter()) {
            for c in 0..4 {
                let x = i64::from(a[c]) - i64::from(b[c]);
                e += (x * x) as u64;
            }
        }
        e
    }

    // Every candidate's reported error is its decoded SSE, so mode choices
    // compare real errors; opaque blocks keep alpha exactly 255.
    #[test]
    fn masked_representatives_preserve_subset_fits() {
        let mut state = 0x741e_5badu32;
        for case in 0..512 {
            let mut px = [[0; 4]; 16];
            for i in 0..16 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                px[i] = if i > 0 && case % 3 != 0 {
                    px[(state as usize >> 16) % i]
                } else {
                    state.to_le_bytes()
                };
            }
            let mask = (state as u16) | 1;
            let original = Reps::subset(mask);
            let compact = Reps::distinct(&px).masked(mask);
            for (prec, levels) in [(P1, &L3), (P3, &L2), (P7, &L2)] {
                let a = fit::<0, 4>(&px, &original, prec, levels, 2, 2, false);
                let b = fit::<0, 4>(&px, &compact, prec, levels, 2, 2, false);
                assert_eq!(a.err, b.err);
                assert_eq!(a.d, b.d);
                for i in 0..16 {
                    if mask >> i & 1 != 0 {
                        assert_eq!(a.idx[i], b.idx[i]);
                    }
                }
            }
        }
    }

    #[test]
    fn fast_candidates_report_their_decoded_error() {
        let mut state = 0x2545_f491u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for case in 0..3000 {
            let mut px = [[0u8; 4]; 16];
            let base = next();
            let palette = [next(), next(), next()];
            for (i, p) in px.iter_mut().enumerate() {
                let r = next();
                for c in 0..4 {
                    let spread = [255, 40, 8][case % 3];
                    let centre = (base >> (8 * c)) as u8;
                    let v = i32::from(centre) + (r >> (8 * c)) as i32 % (spread + 1) - spread / 2;
                    p[c] = v.clamp(0, 255) as u8;
                }
                if case % 7 == 3 {
                    *p = palette[r as usize % 3].to_le_bytes();
                }
                if case % 4 == 0 {
                    p[3] = 255;
                }
                if case % 5 == 1 && i % 3 == 0 {
                    p[0] = 255 - p[0];
                }
            }
            let opaque = px.iter().all(|p| p[3] == 255);
            let reps = Reps::distinct(&px);
            for refine in [false, true] {
                let m6 = mode6(&px, &reps, opaque, refine);
                assert_eq!(
                    u64::from(m6.err),
                    decoded_sse(&px, &pack6(m6)),
                    "case {case}"
                );
            }
            for (mode, sel) in [(5, 0), (4, 0), (4, 1)] {
                for rotation in 0..4 {
                    let mut src = px;
                    if rotation > 0 {
                        for p in &mut src {
                            p.swap(rotation - 1, 3);
                        }
                    }
                    let (c, a) = dual(&src, &reps, mode, sel, rotation % 2);
                    let block = pack_dual(mode, rotation, sel, c, a);
                    assert_eq!(u64::from(c.err + a.err), decoded_sse(&px, &block));
                }
            }
            let partition = case % 64;
            for mode in [1, 3, 7] {
                let s = fit_partition(&px, mode, partition, opaque && mode == 7, 1, 0);
                let block = pack2(mode, partition, s);
                if mode == 7 || opaque {
                    assert_eq!(u64::from(s[0].err + s[1].err), decoded_sse(&px, &block));
                }
            }
            if opaque {
                let d = super::super::independent_decoder::decode(&encode(&px));
                assert!(d.iter().all(|p| p[3] == 255), "case {case}");
            }
        }
    }
}
