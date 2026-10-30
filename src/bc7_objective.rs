//! Visible RGB and independent all-texel alpha fits for nonopaque searches.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct Objective {
    pub visible: u16,
    pub alpha: u8,
    pub alpha_weight: u8,
}

impl Objective {
    pub fn weight(self, i: usize, c: usize) -> i32 {
        if c == usize::from(self.alpha) {
            i32::from(self.alpha_weight)
        } else {
            i32::from(self.visible >> i & 1)
        }
    }
}

// The bounded objectives use only weights 0, 1 and 4; avoid libm in no_std.
fn root_weight(w: i32) -> f32 {
    match w {
        0 => 0.0,
        1 => 1.0,
        4 => 2.0,
        _ => unreachable!(),
    }
}

// Exhaustive selectors are bounded by 16 entries. Missing RGB never enters
// either the selector projection or the error; alpha remains observed.
pub(super) fn select<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    d: &Pair,
    lv: &Levels,
    idx: &mut [u8; 16],
    limit: u32,
) -> u32 {
    #[cfg(feature = "simd")]
    let objective = reps.objective.unwrap();
    #[cfg(feature = "simd")]
    if let Some(err) = simd::select::<C0, C1>(px, &reps.base, d, lv, idx, limit, Some(objective)) {
        return err;
    }
    select_scalar::<C0, C1>(px, reps, d, lv, idx, limit)
}

pub(super) fn select_scalar<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    d: &Pair,
    lv: &Levels,
    idx: &mut [u8; 16],
    limit: u32,
) -> u32 {
    let objective = reps.objective.unwrap();

    let mut pal = [[0; 4]; 16];
    for (j, &w) in lv.w.iter().enumerate() {
        for c in C0..C1 {
            pal[j][c] = ((64 - w) * d[0][c] + w * d[1][c] + 32) >> 6;
        }
    }
    let mut err = 0;
    for (i, count) in reps.iter() {
        let mut best = (i32::MAX, 0);
        for (j, entry) in pal[..lv.w.len()].iter().enumerate() {
            let mut e = 0;
            for c in C0..C1 {
                let x = i32::from(px[i][c]) - entry[c];
                e += objective.weight(i, c) * x * x;
            }
            if e < best.0 {
                best = (e, j);
            }
        }
        idx[i] = best.1 as u8;
        err += (count * best.0) as u32;
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

// Missing observations have zero centred value. Each observed channel has
// its own mean; cross moments use only jointly observed channels. This is a
// positive semidefinite seed covariance, not an invented invisible colour.
fn moments<const C0: usize, const C1: usize>(px: &Block, reps: &Reps) -> ([f32; 4], [[f32; 4]; 4]) {
    let o = reps.objective.unwrap();
    let mut mean = [0.0; 4];
    for c in C0..C1 {
        let mut n = 0;
        let mut sum = 0;
        for (i, count) in reps.iter() {
            let w = count * o.weight(i, c);
            n += w;
            sum += w * i32::from(px[i][c]);
        }
        if n != 0 {
            mean[c] = sum as f32 / n as f32;
        }
    }
    let mut cov = [[0.0; 4]; 4];
    for (i, count) in reps.iter() {
        let mut v = [0.0; 4];
        for c in C0..C1 {
            v[c] = (f32::from(px[i][c]) - mean[c]) * root_weight(o.weight(i, c));
        }
        for c in C0..C1 {
            for e in C0..C1 {
                cov[c][e] += count as f32 * v[c] * v[e];
            }
        }
    }
    (mean, cov)
}

pub(super) fn estimate<const C0: usize, const C1: usize>(px: &Block, reps: &Reps) -> (f32, f32) {
    let (_, cov) = moments::<C0, C1>(px, reps);
    let mut trace = 0.0;
    let mut top = C0;
    for c in C0..C1 {
        trace += cov[c][c];
        if cov[c][c] > cov[top][top] {
            top = c;
        }
    }
    if trace <= 0.0 {
        return (0.0, 0.0);
    }
    let mut ww = 0.0;
    let mut wcw = 0.0;
    for c in C0..C1 {
        let mut cw = 0.0;
        for e in C0..C1 {
            cw += cov[c][e] * cov[e][top];
        }
        ww += cov[c][top] * cov[c][top];
        wcw += cov[c][top] * cw;
    }
    let lambda = (wcw / ww).min(trace);
    (trace - lambda, lambda)
}

pub(super) fn targets<const C0: usize, const C1: usize>(px: &Block, reps: &Reps) -> Pair {
    let (mean, cov) = moments::<C0, C1>(px, reps);
    let o = reps.objective.unwrap();
    // Use observed per-channel extrema for a scalar, including empty subsets.
    if C1 - C0 == 1 {
        let mut lo = 255;
        let mut hi = 0;
        for (i, _) in reps.iter() {
            if o.weight(i, C0) != 0 {
                lo = lo.min(i32::from(px[i][C0]));
                hi = hi.max(i32::from(px[i][C0]));
            }
        }
        let mut t = [[0; 4]; 2];
        if lo <= hi {
            t[0][C0] = lo;
            t[1][C0] = hi;
        }
        return t;
    }
    let mut top = C0;
    for c in C0..C1 {
        if cov[c][c] > cov[top][top] {
            top = c;
        }
    }
    let mut v = [0.0; 4];
    for c in C0..C1 {
        v[c] = cov[c][top];
    }
    for _ in 0..3 {
        let mut next = [0.0f32; 4];
        let mut largest = 0.0f32;
        for c in C0..C1 {
            for e in C0..C1 {
                next[c] += cov[c][e] * v[e];
            }
            largest = largest.max(next[c].abs());
        }
        if largest == 0.0 {
            break;
        }
        for c in C0..C1 {
            v[c] = next[c] / largest;
        }
    }
    // Project each observation using only its observed channels.
    let mut lo = 0.0f32;
    let mut hi = 0.0f32;
    for (i, _) in reps.iter() {
        let mut dot = 0.0;
        let mut len = 0.0;
        for c in C0..C1 {
            let w = o.weight(i, c) as f32;
            let dir = v[c] / root_weight(o.weight(i, c).max(1));
            dot += w * (f32::from(px[i][c]) - mean[c]) * dir;
            len += w * dir * dir;
        }
        if len > 0.0 {
            lo = lo.min(dot / len);
            hi = hi.max(dot / len);
        }
    }
    let mut t = [[0; 4]; 2];
    for c in C0..C1 {
        let dir = v[c]
            / root_weight(if c == usize::from(o.alpha) {
                i32::from(o.alpha_weight)
            } else {
                1
            });
        t[0][c] = (mean[c] + lo * dir + 0.5).clamp(0.0, 255.0) as i32;
        t[1][c] = (mean[c] + hi * dir + 0.5).clamp(0.0, 255.0) as i32;
    }
    t
}

pub(super) fn least_squares<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    idx: &[u8; 16],
    lv: &Levels,
) -> Option<Pair> {
    let o = reps.objective.unwrap();
    let mut t = [[0; 4]; 2];
    let mut fitted = false;
    for c in C0..C1 {
        let (mut aa, mut ab, mut bb, mut ax, mut bx) = (0i64, 0i64, 0i64, 0i64, 0i64);
        let (mut sum, mut n) = (0i64, 0i64);
        for (i, count) in reps.iter() {
            let w = i64::from(count * o.weight(i, c));
            let b = i64::from(lv.w[usize::from(idx[i])]);
            let a = 64 - b;
            let x = i64::from(px[i][c]);
            aa += w * a * a;
            ab += w * a * b;
            bb += w * b * b;
            ax += w * a * x;
            bx += w * b * x;
            sum += w * x;
            n += w;
        }
        let det = aa * bb - ab * ab;
        if det != 0 {
            let scale = 64.0 / det as f64;
            let round = |x: i64| ((x as f64 * scale).clamp(0.0, 255.0) + 0.5) as i32;
            t[0][c] = round(ax * bb - bx * ab);
            t[1][c] = round(bx * aa - ax * ab);
            fitted = true;
        } else if n != 0 {
            t[0][c] = ((sum + n / 2) / n) as i32;
            t[1][c] = t[0][c];
        }
    }
    fitted.then_some(t)
}

pub(super) fn quantize<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    t: &Pair,
    prec: Prec,
    force_p1: bool,
) -> Pair {
    if reps.objective.is_none() {
        return super::quantize::<C0, C1>(t, prec, force_p1);
    }
    let o = reps.objective.unwrap();
    let error = |k: usize, p: usize| {
        let mut err = 0;
        for c in C0..C1 {
            let n: i32 = reps.iter().map(|(i, count)| count * o.weight(i, c)).sum();
            let x = t[k][c] - i32::from(QUANT[prec.bits][p][t[k][c] as usize]);
            err += n * x * x;
        }
        err
    };
    let choice = match prec.pbit {
        PBit::None => [2; 2],
        PBit::Unique => [0, 1].map(|k| usize::from(force_p1 || error(k, 1) < error(k, 0))),
        PBit::Shared => {
            [usize::from(force_p1 || error(0, 1) + error(1, 1) < error(0, 0) + error(1, 0)); 2]
        }
    };
    let _ = px;
    quantize_with::<C0, C1>(t, prec, choice)
}

pub(super) fn difficulty(px: &Block, reps: &Reps, error: u32) -> usize {
    if error == 0 {
        return 0;
    }
    let o = reps.objective.unwrap();
    let mut energy = 0u64;
    let mut noise = 0u64;
    for c in 0..4 {
        let (mut n, mut s, mut sq) = (0u64, 0u64, 0u64);
        for (i, count) in reps.iter() {
            let w = u64::from(count as u32) * u64::from(o.weight(i, c) as u32);
            let x = u64::from(px[i][c]);
            n += w;
            s += w * x;
            sq += w * x * x;
        }
        energy += (n * sq - s * s).checked_div(n).unwrap_or(0);
        noise += 4 * n;
    }
    1 + (4 * u64::from(error) / (energy + noise).max(1)).min(3) as usize
}

pub(super) fn polish<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    f: &mut Fit,
    prec: Prec,
    lv: &Levels,
) -> bool {
    let n = prec.bits as u32;
    let step = if prec.pbit == PBit::None { 1 } else { 2 };
    let max = (1i32 << n) - 1;
    #[cfg(feature = "simd")]
    if let Some(changed) = simd::polish::<C0, C1>(px, &reps.base, f, prec, lv, reps.objective) {
        return changed;
    }
    let channel_err = |c: usize, a: i32, b: i32| -> u32 {
        let mut e = 0;
        for (i, count) in reps.iter() {
            let w = lv.w[usize::from(f.idx[i])];
            let x = i32::from(px[i][c]) - (((64 - w) * a + w * b + 32) >> 6);
            e += x * x * count * reps.objective.unwrap().weight(i, c);
        }
        e as u32
    };
    let mut fixed = 0u32;
    let mut d = f.d;
    for c in C0..C1 {
        let x0 = d[0][c] >> (8 - n);
        let x1 = d[1][c] >> (8 - n);
        let mut best = (channel_err(c, d[0][c], d[1][c]), d[0][c], d[1][c]);
        for (a, b) in [
            (x0 - step, x1),
            (x0 + step, x1),
            (x0, x1 - step),
            (x0, x1 + step),
        ] {
            if !(0..=max).contains(&a) || !(0..=max).contains(&b) {
                continue;
            }
            let (va, vb) = (expand(a as u32, n) as i32, expand(b as u32, n) as i32);
            let e = channel_err(c, va, vb);
            if e < best.0 {
                best = (e, va, vb);
            }
        }
        fixed += best.0;
        d[0][c] = best.1;
        d[1][c] = best.2;
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

#[derive(Clone, Copy)]
pub(super) struct Reps {
    base: super::Reps,
    pub objective: Option<Objective>,
}
impl core::ops::Deref for Reps {
    type Target = super::Reps;
    fn deref(&self) -> &Self::Target {
        &self.base
    }
}
impl Reps {
    pub fn distinct(px: &Block) -> Self {
        Self {
            base: super::Reps::distinct(px),
            objective: None,
        }
    }
    pub fn masked(&self, mask: u16) -> Self {
        Self {
            base: self.base.masked(mask),
            objective: self.objective,
        }
    }
}
use super::search::{dual_layout, parity_of, Search, P0, P2};
pub(super) use targets as axis_targets;

pub(super) fn fit<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    prec: Prec,
    lv: &Levels,
    iters: usize,
    rounds: usize,
    force_p1: bool,
) -> Fit {
    let d = quantize::<C0, C1>(px, reps, &axis_targets::<C0, C1>(px, reps), prec, force_p1);
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
pub(super) fn improve<const C0: usize, const C1: usize>(
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
        let d = quantize::<C0, C1>(px, reps, &t, prec, force_p1);
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

pub(super) fn parity<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    f: &mut Fit,
    prec: Prec,
    lv: &Levels,
    iters: usize,
) {
    let combos: &[[usize; 2]] = match prec.pbit {
        PBit::None => return,
        PBit::Unique => &[[0, 0], [0, 1], [1, 0], [1, 1]],
        PBit::Shared => &[[0, 0], [1, 1]],
    };
    if f.err == 0 {
        return;
    }
    let Some(t) = least_squares::<C0, C1>(px, reps, &f.idx, lv) else {
        return;
    };
    let current = parity_of::<C0>(&f.d, prec);
    for &p in combos {
        if p == current {
            continue;
        }
        let d = quantize_with::<C0, C1>(&t, prec, p);
        let mut g = Fit {
            err: 0,
            d,
            idx: f.idx,
        };
        g.err = select::<C0, C1>(px, reps, &d, lv, &mut g.idx, u32::MAX);
        for _ in 0..iters {
            if g.err == 0 {
                break;
            }
            let Some(t) = least_squares::<C0, C1>(px, reps, &g.idx, lv) else {
                break;
            };
            let d = quantize_with::<C0, C1>(&t, prec, p);
            if d == g.d {
                break;
            }
            let mut idx = g.idx;
            let err = select::<C0, C1>(px, reps, &d, lv, &mut idx, g.err);
            if err >= g.err {
                break;
            }
            g = Fit { err, d, idx };
        }
        if g.err < f.err {
            *f = g;
        }
    }
}

pub(super) fn dual_fit(
    px: &Block,
    reps: &Reps,
    targets: &[Pair; 2],
    mode: usize,
    sel: usize,
    iters: usize,
    compact: bool,
) -> (Fit, Fit) {
    let (cp, cl, ap, al) = dual_layout(mode, sel);
    (
        if compact {
            rough_with::<0, 2>(px, reps, &targets[0], cp, cl, iters, false, u32::MAX)
        } else {
            rough_with::<0, 3>(px, reps, &targets[0], cp, cl, iters, false, u32::MAX)
        },
        rough_with::<3, 4>(px, reps, &targets[1], ap, al, iters, false, u32::MAX),
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "mode and selector precision are independent of search effort"
)]
pub(super) fn refine_dual(
    px: &Block,
    reps: &Reps,
    mode: usize,
    sel: usize,
    color: &mut Fit,
    alpha: &mut Fit,
    cfg: &Search,
    compact: bool,
) {
    let (cp, cl, ap, al) = dual_layout(mode, sel);
    let (iters, rounds) = (cfg.refine_iters, cfg.refine_rounds);
    if compact {
        improve::<0, 2>(px, reps, color, cp, cl, iters, rounds, false);
    } else {
        improve::<0, 3>(px, reps, color, cp, cl, iters, rounds, false);
    }
    improve::<3, 4>(px, reps, alpha, ap, al, iters, rounds, false);
}

// Endpoints of one subset from its axis targets, then `iters` least-squares
// passes. The error stops counting once it reaches `limit`.
#[allow(
    clippy::too_many_arguments,
    reason = "the fit parameters are independent knobs shared by every mode"
)]
pub(super) fn rough_with<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    t: &Pair,
    prec: Prec,
    lv: &Levels,
    iters: usize,
    force_p1: bool,
    limit: u32,
) -> Fit {
    let d = quantize::<C0, C1>(px, reps, t, prec, force_p1);
    let mut f = Fit {
        err: 0,
        d,
        idx: [0; 16],
    };
    f.err = select::<C0, C1>(px, reps, &d, lv, &mut f.idx, limit);
    if f.err < limit {
        improve::<C0, C1>(px, reps, &mut f, prec, lv, iters, 0, force_p1);
    }
    f
}

pub(super) fn rough(
    px: &Block,
    reps: &Reps,
    t: &Pair,
    mode: usize,
    opaque: bool,
    iters: usize,
    limit: u32,
) -> Fit {
    match mode {
        1 => rough_with::<0, 3>(px, reps, t, P1, &L3, iters, false, limit),
        3 => rough_with::<0, 3>(px, reps, t, P3, &L2, iters, false, limit),
        _ => {
            let mut f = rough_with::<0, 4>(px, reps, t, P7, &L2, iters, opaque, limit);
            if opaque {
                f.d[0][3] = 255;
                f.d[1][3] = 255;
            }
            f
        }
    }
}

// Extra passes on one subset of a two-subset candidate.
pub(super) fn refine_subset(
    px: &Block,
    mode: usize,
    reps: &Reps,
    f: &mut Fit,
    opaque: bool,
    cfg: &Search,
) {
    let (iters, rounds) = (cfg.refine_iters, cfg.refine_rounds);
    match mode {
        0 => {
            parity::<0, 3>(px, reps, f, P0, &L3, iters);
            improve::<0, 3>(px, reps, f, P0, &L3, iters, rounds, false);
        }
        2 => improve::<0, 3>(px, reps, f, P2, &L2, iters, rounds, false),
        1 => {
            parity::<0, 3>(px, reps, f, P1, &L3, iters);
            improve::<0, 3>(px, reps, f, P1, &L3, iters, rounds, false);
        }
        3 => {
            parity::<0, 3>(px, reps, f, P3, &L2, iters);
            improve::<0, 3>(px, reps, f, P3, &L2, iters, rounds, false);
        }
        _ => {
            if !opaque {
                parity::<0, 4>(px, reps, f, P7, &L2, iters);
            }
            improve::<0, 4>(px, reps, f, P7, &L2, iters, rounds, opaque);
            if opaque {
                f.d[0][3] = 255;
                f.d[1][3] = 255;
            }
        }
    }
}
