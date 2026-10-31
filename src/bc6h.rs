// Independent implementation of Khronos Data Format 1.4 section 20.2.
// DirectXTex's endpoint perturbation search informed the search strategy only:
// https://github.com/microsoft/DirectXTex/blob/main/DirectXTex/BC6HBC7.cpp
use crate::Effort;
#[path = "bc6h_layout.rs"]
mod layout;
use crate::bc7::tables as partitions;
const MODES: [u8; 14] = [0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15];
const PREC: [u32; 14] = [10, 7, 11, 11, 11, 9, 8, 8, 8, 6, 10, 11, 12, 16];
const DELTA: [[u32; 3]; 14] = [
    [5; 3],
    [6; 3],
    [5, 4, 4],
    [4, 5, 4],
    [4, 4, 5],
    [5; 3],
    [6, 5, 5],
    [5, 6, 5],
    [5, 5, 6],
    [6; 3],
    [10; 3],
    [9; 3],
    [8; 3],
    [4; 3],
];
const W3: [i32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const W4: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];
fn sign(x: i32, bits: u32) -> i32 {
    let bit = 1 << (bits - 1);
    (x ^ bit) - bit
}
fn transformed(m: usize) -> bool {
    m != 9 && m != 10
}
fn unquant(x: i32, bits: u32, signed: bool) -> i32 {
    if signed {
        if bits >= 16 {
            return x;
        }
        let a = x.abs();
        let q = if a == 0 {
            0
        } else if a >= (1 << (bits - 1)) - 1 {
            32767
        } else {
            ((a << 15) + 16384) >> (bits - 1)
        };
        if x < 0 {
            -q
        } else {
            q
        }
    } else if bits >= 15 {
        x
    } else if x == 0 {
        0
    } else if x == (1 << bits) - 1 {
        65535
    } else {
        ((x << 15) + 16384) >> (bits - 1)
    }
}
fn finish(x: i32, signed: bool) -> u16 {
    let v = if signed {
        ((x.abs() * 31) >> 5) | if x < 0 { 32768 } else { 0 }
    } else {
        (x * 31) >> 6
    };
    v as u16
}
pub(crate) fn half(x: u16) -> f32 {
    let a = u32::from(x & 32767);
    let exponent = a >> 10;
    let mantissa = a & 1023;
    let v = if exponent == 0 {
        (mantissa as f32) * (1.0 / 16777216.0)
    } else {
        f32::from_bits(((exponent + 112) << 23) | (mantissa << 13))
    };
    let v = if exponent == 31 {
        if mantissa == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        v
    };
    if x & 32768 != 0 {
        -v
    } else {
        v
    }
}
fn pattern(m: usize, p: usize, i: usize) -> usize {
    if m >= 10 {
        0
    } else {
        ((partitions::PART2[p] >> (2 * i)) & 1) as usize
    }
}
fn anchor(m: usize, p: usize, i: usize) -> bool {
    i == 0 || (m < 10 && i == partitions::ANCHOR2[p])
}
fn weights(m: usize) -> &'static [i32] {
    if m < 10 {
        &W3
    } else {
        &W4
    }
}
fn palette(ep: &[[i32; 3]; 4], m: usize, signed: bool) -> [[[u16; 3]; 16]; 2] {
    let mut out = [[[0; 3]; 16]; 2];
    for s in 0..if m < 10 { 2 } else { 1 } {
        for (i, &w) in weights(m).iter().enumerate() {
            for c in 0..3 {
                out[s][i][c] = finish(
                    (ep[2 * s][c] * (64 - w) + ep[2 * s + 1][c] * w + 32) >> 6,
                    signed,
                );
            }
        }
    }
    out
}
pub(crate) fn decode(block: &[u8; 16], signed: bool) -> [[u16; 3]; 16] {
    let bits = u128::from_le_bytes(*block);
    let tag = if bits & 3 < 2 { bits & 3 } else { bits & 31 };
    let Some(m) = MODES.iter().position(|&x| u128::from(x) == tag) else {
        return [[0; 3]; 16];
    };
    let mut raw = [0i32; 12];
    for (pos, &(e, b)) in layout::LAYOUT[m].iter().enumerate() {
        if e != 255 && bits & (1u128 << pos) != 0 {
            raw[usize::from(e)] |= 1 << b;
        }
    }
    let mut ep = [[0; 3]; 4];
    for e in 0..if m < 10 { 4 } else { 2 } {
        for c in 0..3 {
            let mut v = raw[e * 3 + c];
            if e == 0 {
                if signed {
                    v = sign(v, PREC[m]);
                }
            } else {
                if signed || transformed(m) {
                    v = sign(v, DELTA[m][c]);
                }
                if transformed(m) {
                    v = (v + ep[0][c]) & ((1 << PREC[m]) - 1);
                    if signed {
                        v = sign(v, PREC[m]);
                    }
                }
            }
            ep[e][c] = v;
        }
    }
    for endpoint in &mut ep {
        for c in endpoint {
            *c = unquant(*c, PREC[m], signed);
        }
    }
    let pal = palette(&ep, m, signed);
    let p = if m < 10 {
        ((bits >> 77) & 31) as usize
    } else {
        0
    };
    let mut pos = if m < 10 { 82 } else { 65 };
    let mut out = [[0; 3]; 16];
    for (i, pixel) in out.iter_mut().enumerate() {
        let n = if m < 10 { 3 } else { 4 } - usize::from(anchor(m, p, i));
        let idx = ((bits >> pos) & ((1 << n) - 1)) as usize;
        *pixel = pal[pattern(m, p, i)][idx];
        pos += n;
    }
    out
}
// libm avoids platform libc variation; all search comparisons are f64.
pub(crate) fn perceptual(x: f32) -> f64 {
    let a = libm::log2(1.0 + 16.0 * f64::from(x.abs()));
    if x < 0.0 {
        -a
    } else {
        a
    }
}
include!(concat!(env!("OUT_DIR"), "/hdr_log.rs"));
fn half_log(h: u16) -> f64 {
    let value = HALF_LOG[usize::from(h & 32767)];
    if h & 32768 != 0 {
        -value
    } else {
        value
    }
}
fn quant(x: i32, m: usize, signed: bool) -> i32 {
    let bits = PREC[m];
    let (lo, hi) = if signed {
        (-(1 << (bits - 1)) + 1, (1 << (bits - 1)) - 1)
    } else {
        (0, (1 << bits) - 1)
    };
    if bits >= if signed { 16 } else { 15 } {
        return x.clamp(lo, hi);
    }
    // Interior reconstruction is (2*q+1)*2^(15-bits), with a
    // symmetric signed magnitude. The inverse is within one integer of
    // this shift; explicitly compare neighbours for saturation and low ties.
    let magnitude = x.abs() >> (16 - bits);
    let guess = (if x < 0 { -magnitude } else { magnitude }).clamp(lo, hi);
    let mut best = (guess - 1).max(lo);
    let mut error = (unquant(best, bits, signed) - x).abs();
    for q in guess..=(guess + 1).min(hi) {
        let next = (unquant(q, bits, signed) - x).abs();
        if next < error {
            error = next;
            best = q;
        }
    }
    best
}
// Reject transformed deltas without constructing a complete trial bitstream.
fn admissible(ep: &[[i32; 3]; 4], m: usize) -> bool {
    if !transformed(m) {
        return true;
    }
    for e in 1..if m < 10 { 4 } else { 2 } {
        for c in 0..3 {
            let d = sign((ep[e][c] - ep[0][c]) & ((1 << PREC[m]) - 1), PREC[m]);
            let bound = 1 << (DELTA[m][c] - 1);
            if d < -bound || d >= bound {
                return false;
            }
        }
    }
    true
}
fn packed(
    ep: &[[i32; 3]; 4],
    m: usize,
    p: usize,
    idx: &[usize; 16],
    _signed: bool,
) -> Option<[u8; 16]> {
    let mut raw = [0i32; 12];
    for e in 0..if m < 10 { 4 } else { 2 } {
        for c in 0..3 {
            let mask = (1 << PREC[m]) - 1;
            let q = ep[e][c];
            let value = if e > 0 && transformed(m) {
                let d = sign((q - ep[0][c]) & mask, PREC[m]);
                let bound = 1 << (DELTA[m][c] - 1);
                if d < -bound || d >= bound {
                    return None;
                }
                d & ((1 << DELTA[m][c]) - 1)
            } else {
                q & mask
            };
            raw[e * 3 + c] = value;
        }
    }
    let mut bits = u128::from(MODES[m]);
    for (pos, &(e, b)) in layout::LAYOUT[m].iter().enumerate() {
        if e != 255 && raw[usize::from(e)] & (1 << b) != 0 {
            bits |= 1 << pos;
        }
    }
    if m < 10 {
        bits |= u128::try_from(p).ok()? << 77;
    }
    let mut pos = if m < 10 { 82 } else { 65 };
    for (i, &v) in idx.iter().enumerate() {
        let n = if m < 10 { 3 } else { 4 } - usize::from(anchor(m, p, i));
        if v >= 1 << n {
            return None;
        }
        bits |= u128::try_from(v).ok()? << pos;
        pos += n;
    }
    Some(bits.to_le_bytes())
}
fn score(
    ep: &[[i32; 3]; 4],
    src: &[[f64; 3]; 16],
    m: usize,
    p: usize,
    signed: bool,
) -> (f64, [usize; 16]) {
    let mut unq = *ep;
    for e in &mut unq {
        for c in e {
            *c = unquant(*c, PREC[m], signed);
        }
    }
    let pal = palette(&unq, m, signed);
    let mut log = [[[0.0; 3]; 16]; 2];
    for s in 0..if m < 10 { 2 } else { 1 } {
        for k in 0..weights(m).len() {
            for c in 0..3 {
                log[s][k][c] = half_log(pal[s][k][c]);
            }
        }
    }
    let mut total = 0.0;
    let mut idx = [0; 16];
    for i in 0..16 {
        let s = pattern(m, p, i);
        let mut best = f64::INFINITY;
        for (k, color) in log[s]
            .iter()
            .enumerate()
            .take(weights(m).len() >> usize::from(anchor(m, p, i)))
        {
            let mut error = 0.0;
            for c in 0..3 {
                let d = src[i][c] - color[c];
                error += d * d;
            }
            if error < best {
                best = error;
                idx[i] = k;
            }
        }
        total += best;
    }
    (total, idx)
}
// Damped Gauss-Newton fitting in the decoded perceptual objective. The
// Jacobian is measured in the integer interpolation domain, including half
// reconstruction; reassignment after each accepted step handles selector moves.
fn fit(
    ep: &mut [[i32; 3]; 4],
    src: &[[f64; 3]; 16],
    m: usize,
    p: usize,
    signed: bool,
    rounds: usize,
) {
    let mut current = score(ep, src, m, p, signed);
    for _ in 0..rounds {
        let mut unq = *ep;
        for e in &mut unq {
            for c in e {
                *c = unquant(*c, PREC[m], signed);
            }
        }
        let mut deltas = [[0.0; 3]; 4];
        for s in 0..if m < 10 { 2 } else { 1 } {
            for c in 0..3 {
                let (mut aa, mut ab, mut bb, mut ra, mut rb) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for (i, pixel) in src.iter().enumerate() {
                    if pattern(m, p, i) != s {
                        continue;
                    }
                    let w = weights(m)[current.1[i]];
                    let u = (unq[2 * s][c] * (64 - w) + unq[2 * s + 1][c] * w + 32) >> 6;
                    let lo = (u - 32).max(if signed { -32767 } else { 0 });
                    let hi = (u + 32).min(if signed { 32767 } else { 65535 });
                    let derivative = (half_log(finish(hi, signed)) - half_log(finish(lo, signed)))
                        / f64::from(hi - lo);
                    let a = f64::from(64 - w) / 64.0 * derivative;
                    let b = f64::from(w) / 64.0 * derivative;
                    let r = pixel[c] - half_log(finish(u, signed));
                    aa += a * a;
                    ab += a * b;
                    bb += b * b;
                    ra += a * r;
                    rb += b * r;
                }
                let determinant = aa * bb - ab * ab;
                if determinant > 1e-30 {
                    deltas[2 * s][c] = (ra * bb - rb * ab) / determinant;
                    deltas[2 * s + 1][c] = (rb * aa - ra * ab) / determinant;
                }
            }
        }
        let mut accepted = false;
        for damping in [1.0, 0.5, 0.25] {
            let mut trial = *ep;
            for e in 0..if m < 10 { 4 } else { 2 } {
                for c in 0..3 {
                    let target =
                        (f64::from(unq[e][c]) + damping * deltas[e][c]).clamp(-65535.0, 65535.0);
                    trial[e][c] = quant(libm::round(target) as i32, m, signed);
                }
            }
            if !admissible(&trial, m) {
                continue;
            }
            let next = score(&trial, src, m, p, signed);
            if next.0 < current.0 {
                *ep = trial;
                current = next;
                accepted = true;
                break;
            }
        }
        if !accepted {
            break;
        }
    }
}
type Candidate = ([[i32; 3]; 4], usize, usize);
// Eight blocks share a structure-of-arrays distance pass. Endpoint search is
// still independent per block, retaining the scalar candidate and tie order.
pub(crate) fn encode_batch(
    pixels: &[[[f32; 3]; 16]; 8],
    signed: bool,
    effort: Effort,
    out: &mut [[u8; 16]],
) {
    let source = pixels.map(|block| block.map(|p| p.map(perceptual)));
    let soa = core::array::from_fn(|c| {
        core::array::from_fn(|i| core::array::from_fn(|b| source[b][i][c]))
    });
    #[cfg(feature = "simd")]
    let distances = crate::bc7::hdr_distances(&soa);
    #[cfg(not(feature = "simd"))]
    let distances = distance_batch(&soa);
    for (b, dst) in out.iter_mut().enumerate() {
        *dst = search(&pixels[b], &source[b], &distances[b], signed, effort).0;
    }
}
pub(crate) fn distance_batch(src: &[[[f64; 8]; 16]; 3]) -> [[[f64; 16]; 16]; 8] {
    let mut out = [[[0.0; 16]; 16]; 8];
    for a in 0..16 {
        for b in a..16 {
            for (l, block) in out.iter_mut().enumerate() {
                let mut d = 0.0;
                for channel in src {
                    let v = channel[a][l] - channel[b][l];
                    d += v * v;
                }
                block[a][b] = d;
            }
        }
    }
    out
}
fn search(
    pixels: &[[f32; 3]; 16],
    src: &[[f64; 3]; 16],
    distances: &[[f64; 16]; 16],
    signed: bool,
    effort: Effort,
) -> ([u8; 16], Option<Candidate>) {
    let domain = pixels.map(|p| {
        p.map(|x| {
            // Invert final scaling in the integer half domain. Binary search avoids
            // any dependence on the host's float-to-half instruction rounding.
            let mut lo = 0u16;
            let mut hi = 31743u16;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if half(mid) < x.abs() {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let a = i32::from(lo) * if signed { 32 } else { 64 } / 31;
            if x < 0.0 {
                -a
            } else {
                a
            }
        })
    });
    let mut best = f64::INFINITY;
    let mut result = [0; 16];
    let mut winner = None;
    // Cache the endpoint pair and a cheap line-residual estimate once per
    // partition, independent of endpoint precision and orientation.
    const fn masks() -> [[u16; 2]; 33] {
        let mut out = [[0; 2]; 33];
        let mut p = 0;
        while p < 32 {
            let mut i = 0;
            while i < 16 {
                let subset = ((partitions::PART2[p] >> (2 * i)) & 1) as usize;
                out[p][subset] |= 1 << i;
                i += 1;
            }
            p += 1;
        }
        out[32][0] = u16::MAX;
        out
    }
    const MASKS: [[u16; 2]; 33] = masks();
    let mut pairs = [[(0, 0); 2]; 33];
    let mut order = [(0.0, 0usize); 32];
    for (p, pair) in pairs.iter_mut().enumerate() {
        let mut estimate = 0.0;
        for (subset, ends) in pair
            .iter_mut()
            .enumerate()
            .take(if p == 32 { 1 } else { 2 })
        {
            let mut dist = -1.0;
            let mut remaining = MASKS[p][subset];
            while remaining != 0 {
                let a = remaining.trailing_zeros() as usize;
                let mut others = remaining;
                while others != 0 {
                    let b = others.trailing_zeros() as usize;
                    let d = distances[a][b];
                    if d > dist {
                        dist = d;
                        *ends = (a, b);
                    }
                    others &= others - 1;
                }
                remaining &= remaining - 1;
            }
            let (a, b) = *ends;
            let mut remaining = MASKS[p][subset];
            while remaining != 0 {
                let i = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                let mut norm = 0.0;
                let mut dot = 0.0;
                for ((&value, &lo), &hi) in src[i].iter().zip(&src[a]).zip(&src[b]) {
                    let v = value - lo;
                    norm += v * v;
                    dot += v * (hi - lo);
                }
                estimate += if dist > 0.0 {
                    (norm - dot * dot / dist).max(0.0)
                } else {
                    norm
                };
            }
        }
        if p < 32 {
            order[p] = (estimate, p);
        }
    }
    order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    // Nested, fixed budgets reuse preparation and retain the weaker preset's
    // fitted winner. Only newly admitted partitions need initial evaluation.
    let stages = match effort {
        Effort::Fast => 1,
        Effort::Balanced => 2,
        Effort::Quality => 3,
    };
    for &(start, keep, rounds) in [(0, 1, 1), (1, 2, 1), (2, 4, 1)].iter().take(stages) {
        for p in (if start == 0 { Some(32) } else { None })
            .into_iter()
            .chain(order.iter().take(keep).skip(start).map(|x| x.1))
        {
            for m in 0..14 {
                if (m >= 10) != (p == 32) {
                    continue;
                }
                let partition = if p == 32 { 0 } else { p };
                let mut endpoints = [[0; 3]; 4];
                for s in 0..if m < 10 { 2 } else { 1 } {
                    let pair = pairs[p][s];
                    endpoints[2 * s] = domain[pair.0].map(|x| quant(x, m, signed));
                    endpoints[2 * s + 1] = domain[pair.1].map(|x| quant(x, m, signed));
                }
                let p = partition;
                // All endpoint orientations are considered before delta admission.
                for flip in 0..if m < 10 { 4 } else { 2 } {
                    let mut ep = endpoints;
                    for s in 0..if m < 10 { 2 } else { 1 } {
                        if flip & (1 << s) != 0 {
                            ep.swap(2 * s, 2 * s + 1);
                        }
                    }
                    if !admissible(&ep, m) {
                        continue;
                    }
                    let (error, idx) = score(&ep, src, m, p, signed);
                    if error < best {
                        if let Some(block) = packed(&ep, m, p, &idx, signed) {
                            best = error;
                            result = block;
                            winner = Some((ep, m, p));
                        }
                    }
                }
            }
        }
        // Bounded coordinate refinement of the winning quantized endpoints.
        if let Some((mut ep, m, p)) = winner {
            fit(&mut ep, src, m, p, signed, rounds);
            let (error, idx) = score(&ep, src, m, p, signed);
            if error < best {
                if let Some(block) = packed(&ep, m, p, &idx, signed) {
                    best = error;
                    result = block;
                }
            }
            winner = Some((ep, m, p));
        }
    }
    (result, winner)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn integer_quantizer_matches_binary_search_exhaustively() {
        for signed in [false, true] {
            for m in 0..14 {
                for x in -65535..=65535 {
                    assert_eq!(
                        quant(x, m, signed),
                        quant_binary(x, m, signed),
                        "x={x} mode={m} signed={signed}"
                    );
                }
            }
        }
    }
    fn quant_binary(x: i32, m: usize, signed: bool) -> i32 {
        let (mut lo, mut hi) = if signed {
            (-(1 << (PREC[m] - 1)) + 1, (1 << (PREC[m] - 1)) - 1)
        } else {
            (0, (1 << PREC[m]) - 1)
        };
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if unquant(mid, PREC[m], signed) < x {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo > if signed { -(1 << (PREC[m] - 1)) + 1 } else { 0 }
            && (unquant(lo - 1, PREC[m], signed) - x).abs()
                <= (unquant(lo, PREC[m], signed) - x).abs()
        {
            lo - 1
        } else {
            lo
        }
    }
    #[test]
    fn all_mode_partition_packers_are_admitted() {
        for signed in [false, true] {
            for (m, _) in PREC.iter().enumerate() {
                for p in 0..if m < 10 { 32 } else { 1 } {
                    let ep = [[if signed { -2 } else { 2 }; 3]; 4];
                    let idx = [0; 16];
                    let b = packed(&ep, m, p, &idx, signed).expect("representable equal endpoints");
                    let expected = [finish(unquant(ep[0][0], PREC[m], signed), signed); 3];
                    assert_eq!(decode(&b, signed), [expected; 16]);
                }
            }
        }
    }
}

#[cfg(test)]
#[test]
fn khronos_signed_zero_restores_pre_scale_sign() {
    // Section 20.2: (abs(i)*31)>>5, with the sign bit taken from i.
    assert_eq!(finish(-1, true), 0x8000);
    let ep = [[-1; 3]; 4];
    let block = packed(&ep, 13, 0, &[0; 16], true).expect("mode 15 represents -1 exactly");
    assert_eq!(decode(&block, true), [[0x8000; 3]; 16]);
}

#[cfg(test)]
#[test]
fn decoded_half_objective_table_is_exact() {
    for h in 0..31744u16 {
        for sign in [0, 32768] {
            let h = h | sign;
            assert_eq!(half_log(h), perceptual(half(h)));
        }
    }
}
