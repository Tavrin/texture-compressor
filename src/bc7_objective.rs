//! Visible RGB and independent all-texel alpha fits for nonopaque searches.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct Objective {
    pub visible: u16,
    pub alpha: u8,
    pub alpha_weight: u8,
    /// Weight of each visible colour channel; invisible colour weighs zero.
    pub colour_weight: u8,
}

impl Objective {
    #[inline]
    pub fn weight(self, i: usize, c: usize) -> i32 {
        if c == usize::from(self.alpha) {
            i32::from(self.alpha_weight)
        } else {
            self.colour(i)
        }
    }

    /// The weight every colour channel of texel `i` shares.
    #[inline]
    pub fn colour(self, i: usize) -> i32 {
        i32::from(self.visible >> i & 1) * i32::from(self.colour_weight)
    }

    /// Total weights `[colour, alpha]` of a representative set; every colour
    /// channel has the first.
    #[inline]
    fn totals(self, reps: &super::Reps) -> [i32; 2] {
        let mut n = [0; 2];
        for (i, count) in reps.iter() {
            n[0] += count * self.colour(i);
            n[1] += count * i32::from(self.alpha_weight);
        }
        n
    }
}

// The bounded objectives use only weights 0, 1, 2 and 4; avoid libm in no_std.
#[inline]
fn root_weight(w: i32) -> f32 {
    match w {
        0 => 0.0,
        1 => 1.0,
        2 => core::f32::consts::SQRT_2,
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
    let alpha = usize::from(o.alpha);
    // Weights per class [colour, alpha]: colour channels share one per texel.
    let mut n = [0; 2];
    let mut sum = [0; 4];
    for (i, count) in reps.iter() {
        let w = [count * o.colour(i), count * i32::from(o.alpha_weight)];
        n[0] += w[0];
        n[1] += w[1];
        for c in C0..C1 {
            sum[c] += w[usize::from(c == alpha)] * i32::from(px[i][c]);
        }
    }
    let mut mean = [0.0; 4];
    for c in C0..C1 {
        let n = n[usize::from(c == alpha)];
        if n != 0 {
            mean[c] = sum[c] as f32 / n as f32;
        }
    }
    // root_weight of a visible colour, an invisible colour (weight 0) and alpha.
    let roots = [
        root_weight(i32::from(o.colour_weight)),
        0.0,
        root_weight(i32::from(o.alpha_weight)),
    ];
    let mut cov = [[0.0; 4]; 4];
    for (i, count) in reps.iter() {
        let seen = o.visible >> i & 1 == 1;
        let mut v = [0.0; 4];
        for c in C0..C1 {
            let root = roots[if c == alpha { 2 } else { usize::from(!seen) }];
            v[c] = (f32::from(px[i][c]) - mean[c]) * root;
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
    principal::<C0, C1>(&cov)
}

// (Residual off the principal axis, principal variance) of a covariance.
fn principal<const C0: usize, const C1: usize>(cov: &[[f32; 4]; 4]) -> (f32, f32) {
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

// Integer sums of one texel set, additive over texels: visible count, visible
// R, G, B, A sums, the six visible colour products (rr rg rb gg gb bb), the
// three visible colour-alpha products, then the count, alpha sum and alpha
// square of all texels.
type Sums = [i32; 17];

// The observed covariance of a texel set from its sums, as `moments` defines
// it: colour moments over visible texels, alpha over all texels, each channel
// centred on its own mean and scaled by the root of its weight.
fn sums_covariance(s: &Sums, o: Objective) -> [[f32; 4]; 4] {
    let mut cov = [[0f32; 4]; 4];
    let cw = root_weight(i32::from(o.colour_weight));
    let aw = root_weight(i32::from(o.alpha_weight));
    let nv = i64::from(s[0]);
    if nv > 0 {
        let mut k = 5;
        for c in 0..3 {
            for e in c..3 {
                let v = (nv * i64::from(s[k]) - i64::from(s[1 + c]) * i64::from(s[1 + e])) as f32
                    / nv as f32
                    * cw
                    * cw;
                cov[c][e] = v;
                cov[e][c] = v;
                k += 1;
            }
        }
        for c in 0..3 {
            let v = (nv * i64::from(s[11 + c]) - i64::from(s[1 + c]) * i64::from(s[4])) as f32
                / nv as f32
                * cw
                * aw;
            cov[c][3] = v;
            cov[3][c] = v;
        }
    }
    let nt = i64::from(s[14]);
    if nt > 0 {
        cov[3][3] = (nt * i64::from(s[16]) - i64::from(s[15]) * i64::from(s[15])) as f32
            / nt as f32
            * aw
            * aw;
    }
    cov
}

/// Row lookup tables of texel sums for a block with alpha in channel 3: the
/// sums of any texel subset are four lookups, so partition estimates need no
/// per-partition moment passes.
pub(super) struct SumTable {
    rows: [[Sums; 16]; 4],
    o: Objective,
}

impl SumTable {
    pub(super) fn new(px: &Block, o: Objective) -> Self {
        debug_assert_eq!(o.alpha, 3);
        let mut rows = [[[0i32; 17]; 16]; 4];
        for (row, table) in rows.iter_mut().enumerate() {
            for mask in 1usize..16 {
                let i = row * 4 + mask.trailing_zeros() as usize;
                let p = px[i].map(i32::from);
                let mut s = table[mask & (mask - 1)];
                if o.visible >> i & 1 == 1 {
                    s[0] += 1;
                    for c in 0..4 {
                        s[1 + c] += p[c];
                    }
                    let mut k = 5;
                    for c in 0..3 {
                        for e in c..3 {
                            s[k] += p[c] * p[e];
                            k += 1;
                        }
                    }
                    for c in 0..3 {
                        s[11 + c] += p[c] * p[3];
                    }
                }
                s[14] += 1;
                s[15] += p[3];
                s[16] += p[3] * p[3];
                table[mask] = s;
            }
        }
        Self { rows, o }
    }

    fn sums(&self, mask: u16) -> Sums {
        let mut s = [0i32; 17];
        for (row, table) in self.rows.iter().enumerate() {
            let part = &table[usize::from((mask >> (4 * row)) & 15)];
            for k in 0..17 {
                s[k] += part[k];
            }
        }
        s
    }

    /// Estimates of all 64 two-subset partitions for the RGB modes (`[0]`)
    /// and mode 7 (`[1]`).
    pub(super) fn rankings(&self) -> [[(f32, f32); 64]; 2] {
        let total = self.sums(0xffff);
        #[cfg(feature = "simd")]
        {
            let mut out = [[(0f32, 0f32); 64]; 2];
            if self.rankings_simd(&total, &mut out) {
                return out;
            }
        }
        self.rankings_scalar(&total)
    }

    fn rankings_scalar(&self, total: &Sums) -> [[(f32, f32); 64]; 2] {
        let total = *total;
        let mut out = [[(0f32, 0f32); 64]; 2];
        for p in 0..64 {
            let one = self.sums(MASKS2[p][0]);
            let mut zero = total;
            for k in 0..17 {
                zero[k] -= one[k];
            }
            for s in [zero, one] {
                let cov = sums_covariance(&s, self.o);
                let rgb = principal::<0, 3>(&cov);
                let rgba = principal::<0, 4>(&cov);
                out[0][p].0 += rgb.0;
                out[0][p].1 += rgb.1;
                out[1][p].0 += rgba.0;
                out[1][p].1 += rgba.1;
            }
        }
        out
    }

    /// RGB estimates of the first `count` three-subset partitions given by
    /// their subset masks; the rest stay unranked.
    // The scalar estimates of `rankings`, a vector of partitions at a time;
    // false (with `out` untouched) on the scalar backend.
    #[cfg(feature = "simd")]
    fn rankings_simd(&self, total: &Sums, out: &mut [[(f32, f32); 64]; 2]) -> bool {
        let (cw, aw) = self.roots();
        let width = simd::width();
        for base in (0..64).step_by(width) {
            let mut one = [[0i32; 17]; 8];
            let mut zero = [[0i32; 17]; 8];
            for l in 0..width {
                one[l] = self.sums(MASKS2[base + l][0]);
                for k in 0..17 {
                    zero[l][k] = total[k] - one[l][k];
                }
            }
            let (Some(z), Some(o)) = (
                simd::weighted_estimates::<true>(&zero, cw, aw),
                simd::weighted_estimates::<true>(&one, cw, aw),
            ) else {
                return false;
            };
            for l in 0..width {
                for k in 0..2 {
                    let slot = &mut out[k][base + l];
                    slot.0 = 0.0 + z[k][l].0 + o[k][l].0;
                    slot.1 = 0.0 + z[k][l].1 + o[k][l].1;
                }
            }
        }
        true
    }

    // `rankings3` a vector of subsets at a time; false on the scalar backend.
    #[cfg(feature = "simd")]
    fn rankings3_simd(
        &self,
        count: usize,
        masks: &impl Fn(usize) -> [u16; 3],
        out: &mut [(f32, f32); 64],
    ) -> bool {
        let (cw, aw) = self.roots();
        let width = simd::width();
        let mut estimates = [(0f32, 0f32); 3 * 64];
        let subsets = 3 * count.min(64);
        for base in (0..subsets).step_by(width) {
            let mut sums = [[0i32; 17]; 8];
            for l in 0..width.min(subsets - base) {
                let k = base + l;
                sums[l] = self.sums(masks(k / 3)[k % 3]);
            }
            let Some(e) = simd::weighted_estimates::<false>(&sums, cw, aw) else {
                return false;
            };
            let filled = width.min(subsets - base);
            estimates[base..base + filled].copy_from_slice(&e[0][..filled]);
        }
        for (p, slot) in out.iter_mut().enumerate().take(count) {
            *slot = (0.0, 0.0);
            for e in &estimates[3 * p..3 * p + 3] {
                slot.0 += e.0;
                slot.1 += e.1;
            }
        }
        true
    }

    #[cfg(feature = "simd")]
    fn roots(&self) -> (f32, f32) {
        (
            root_weight(i32::from(self.o.colour_weight)),
            root_weight(i32::from(self.o.alpha_weight)),
        )
    }

    pub(super) fn rankings3(
        &self,
        count: usize,
        masks: impl Fn(usize) -> [u16; 3],
    ) -> [(f32, f32); 64] {
        #[cfg(feature = "simd")]
        {
            let mut out = [(f32::MAX, 0f32); 64];
            if self.rankings3_simd(count, &masks, &mut out) {
                return out;
            }
        }
        self.rankings3_scalar(count, &masks)
    }

    fn rankings3_scalar(
        &self,
        count: usize,
        masks: &impl Fn(usize) -> [u16; 3],
    ) -> [(f32, f32); 64] {
        let mut out = [(f32::MAX, 0f32); 64];
        for (p, slot) in out.iter_mut().enumerate().take(count) {
            *slot = (0.0, 0.0);
            for mask in masks(p) {
                let e = principal::<0, 3>(&sums_covariance(&self.sums(mask), self.o));
                slot.0 += e.0;
                slot.1 += e.1;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every backend's partition rankings carry the scalar estimates' bits,
    // for any visibility, alpha range and colour weight the objective allows.
    #[test]
    fn vector_rankings_equal_scalar_bits() {
        let mut state = 0x9e37_79b9u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for trial in 0..400 {
            let spread = [255, 64, 7, 1][trial % 4];
            let px: Block = core::array::from_fn(|_| {
                core::array::from_fn(|_| (next() % 256).min(spread + next() % 3) as u8)
            });
            let o = Objective {
                visible: if trial % 3 == 0 {
                    u16::MAX
                } else {
                    next() as u16
                },
                alpha: 3,
                alpha_weight: [1, 2, 4][trial % 3],
                colour_weight: [1, 0, 2, 4][(trial / 3) % 4],
            };
            let table = SumTable::new(&px, o);
            let total = table.sums(0xffff);
            let pairs = |a: &[(f32, f32)], b: &[(f32, f32)]| {
                for (x, y) in a.iter().zip(b) {
                    assert_eq!(
                        (x.0.to_bits(), x.1.to_bits()),
                        (y.0.to_bits(), y.1.to_bits())
                    );
                }
            };
            let (vector, scalar) = (table.rankings(), table.rankings_scalar(&total));
            for k in 0..2 {
                pairs(&vector[k], &scalar[k]);
            }
            let masks = |p: usize| {
                let [m1, m2] = crate::bc7::MASKS3[p];
                [!(m1 | m2), m1, m2]
            };
            pairs(
                &table.rankings3(16, masks),
                &table.rankings3_scalar(16, &masks),
            );
        }
    }
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
    // Project each observation using only its observed channels. A texel's
    // channel weights depend only on its visibility: [visible, invisible].
    let mut weights = [[0.0f32; 4]; 2];
    let mut dirs = [[0.0f32; 4]; 2];
    for k in 0..2 {
        for c in C0..C1 {
            let w = if c == usize::from(o.alpha) {
                i32::from(o.alpha_weight)
            } else {
                i32::from(k == 0) * i32::from(o.colour_weight)
            };
            weights[k][c] = w as f32;
            dirs[k][c] = v[c] / root_weight(w.max(1));
        }
    }
    let mut lo = 0.0f32;
    let mut hi = 0.0f32;
    for (i, _) in reps.iter() {
        let k = usize::from(o.visible >> i & 1 == 0);
        let (weights, dirs) = (&weights[k], &dirs[k]);
        let mut dot = 0.0;
        let mut len = 0.0;
        for c in C0..C1 {
            let w = weights[c];
            let dir = dirs[c];
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
                i32::from(o.colour_weight).max(1)
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
    let alpha = usize::from(o.alpha);
    // Every colour channel shares one weight per texel, so the selector
    // moments (aa, ab, bb, n) are summed once per weight class: [colour,
    // alpha]. Integer sums over at most 16 texels of weight at most 64 per
    // representative stay far inside i32; `det` widens to i64.
    let mut shared = [[0i32; 4]; 2];
    let mut sums = [[0i32; 3]; 4];
    for (i, count) in reps.iter() {
        let b = lv.w[usize::from(idx[i])];
        let a = 64 - b;
        let w = [count * o.colour(i), count * i32::from(o.alpha_weight)];
        for k in 0..2 {
            shared[k][0] += w[k] * a * a;
            shared[k][1] += w[k] * a * b;
            shared[k][2] += w[k] * b * b;
            shared[k][3] += w[k];
        }
        for c in C0..C1 {
            let w = w[usize::from(c == alpha)];
            let x = i32::from(px[i][c]);
            sums[c][0] += w * a * x;
            sums[c][1] += w * b * x;
            sums[c][2] += w * x;
        }
    }
    let mut t = [[0; 4]; 2];
    let mut fitted = false;
    for c in C0..C1 {
        let [aa, ab, bb, n] = shared[usize::from(c == alpha)].map(i64::from);
        let [ax, bx, sum] = sums[c].map(i64::from);
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
    let totals = o.totals(reps);
    let error = |k: usize, p: usize| {
        let mut err = 0;
        for c in C0..C1 {
            let n = totals[usize::from(c == usize::from(o.alpha))];
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
use super::search::{dual_layout, parity_of, Search, P0};
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
) -> u32 {
    // Passes performed, counted as in the unweighted `improve`.
    let mut passes = 0;
    for _ in 0..iters {
        if best.err == 0 {
            break;
        }
        passes += 1;
        let Some(t) = least_squares::<C0, C1>(px, reps, &best.idx, lv) else {
            break;
        };
        let d = quantize::<C0, C1>(px, reps, &t, prec, force_p1);
        if d == best.d {
            break;
        }
        passes += 1;
        let mut idx = best.idx;
        let err = select::<C0, C1>(px, reps, &d, lv, &mut idx, best.err);
        if err >= best.err {
            break;
        }
        *best = Fit { err, d, idx };
    }
    for _ in 0..rounds {
        if best.err == 0 {
            break;
        }
        passes += 3;
        if !polish::<C0, C1>(px, reps, best, prec, lv) {
            break;
        }
    }
    passes
}

pub(super) fn parity<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    f: &mut Fit,
    prec: Prec,
    lv: &Levels,
    iters: usize,
    tries: usize,
) -> u32 {
    // Passes performed: each least-squares solve and selector pass is one.
    if prec.pbit == PBit::None || f.err == 0 || tries == 0 {
        return 0;
    }
    let mut passes = 1;
    let Some(t) = least_squares::<C0, C1>(px, reps, &f.idx, lv) else {
        return passes;
    };
    let current = parity_of::<C0>(&f.d, prec);
    let (ranked, len) = super::search::ranked_parities::<C0, C1>(&t, prec, current);
    for &p in &ranked[..len.min(tries)] {
        let d = quantize_with::<C0, C1>(&t, prec, p);
        let mut g = Fit {
            err: 0,
            d,
            idx: f.idx,
        };
        passes += 1;
        g.err = select::<C0, C1>(px, reps, &d, lv, &mut g.idx, u32::MAX);
        for _ in 0..iters {
            if g.err == 0 {
                break;
            }
            passes += 1;
            let Some(t) = least_squares::<C0, C1>(px, reps, &g.idx, lv) else {
                break;
            };
            let d = quantize_with::<C0, C1>(&t, prec, p);
            if d == g.d {
                break;
            }
            passes += 1;
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
    passes
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
) -> u32 {
    let (cp, cl, ap, al) = dual_layout(mode, sel);
    let (iters, rounds) = (cfg.refine_iters, cfg.refine_rounds);
    let colour = if compact {
        improve::<0, 2>(px, reps, color, cp, cl, iters, rounds, false)
    } else {
        improve::<0, 3>(px, reps, color, cp, cl, iters, rounds, false)
    };
    colour + improve::<3, 4>(px, reps, alpha, ap, al, iters, rounds, false)
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
) -> u32 {
    let (iters, rounds, tries) = (cfg.refine_iters, cfg.refine_rounds, cfg.pbit_tries);
    match mode {
        0 => {
            parity::<0, 3>(px, reps, f, P0, &L3, iters, tries)
                + improve::<0, 3>(px, reps, f, P0, &L3, iters, rounds, false)
        }
        1 => {
            parity::<0, 3>(px, reps, f, P1, &L3, iters, tries)
                + improve::<0, 3>(px, reps, f, P1, &L3, iters, rounds, false)
        }
        3 => {
            parity::<0, 3>(px, reps, f, P3, &L2, iters, tries)
                + improve::<0, 3>(px, reps, f, P3, &L2, iters, rounds, false)
        }
        _ => {
            let mut passes = 0;
            if !opaque {
                passes += parity::<0, 4>(px, reps, f, P7, &L2, iters, tries);
            }
            passes += improve::<0, 4>(px, reps, f, P7, &L2, iters, rounds, opaque);
            if opaque {
                f.d[0][3] = 255;
                f.d[1][3] = 255;
            }
            passes
        }
    }
}
