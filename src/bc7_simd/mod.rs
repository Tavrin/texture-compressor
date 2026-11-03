#![allow(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
//! Checked dispatch and fixed-array kernels. See SAFETY.md.
use super::*;
use core::sync::atomic::{AtomicU8, Ordering};
#[macro_use]
#[allow(unused_macros)]
mod kernels;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse41;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;

static LEVEL: AtomicU8 = AtomicU8::new(255);
// 0 scalar, 1 SSE4.1, 2 AVX2, 3 NEON, 4 wasm simd128. No mutable
// diagnostic override: selection is immutable for the process and its workers.
// Inlined into every dispatching caller; detection runs once, out of line.
#[inline]
fn level() -> u8 {
    let cached = LEVEL.load(Ordering::Relaxed);
    if cached != 255 {
        return cached;
    }
    detect_level()
}

#[cold]
#[inline(never)]
fn detect_level() -> u8 {
    #[allow(unused_mut)]
    let mut detected = detect();
    #[cfg(feature = "simd-diagnostics")]
    if let Ok(force) = std::env::var("TEXENC_SIMD") {
        let requested = match force.as_str() {
            "scalar" => 0,
            "sse41" => 1,
            "avx2" => 2,
            "neon" => 3,
            "wasm" => 4,
            _ => panic!("invalid TEXENC_SIMD backend"),
        };
        assert!(
            requested == 0
                || requested == detected
                || (requested == 1 && detected == 2 && sse41_detected()),
            "unsupported TEXENC_SIMD backend"
        );
        detected = requested;
    }
    LEVEL.store(detected, Ordering::Relaxed);
    detected
}

/// Whether SSE4.1 itself is reported, checked separately from AVX2 detection
/// before a forced or test-selected SSE4.1 call.
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[cfg_attr(not(any(test, feature = "simd-diagnostics")), allow(dead_code))]
fn sse41_detected() -> bool {
    std::arch::is_x86_feature_detected!("sse4.1")
}

#[cfg(not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"))))]
#[allow(dead_code)]
fn sse41_detected() -> bool {
    false
}

// Every unsafe call below is guarded by level() or the explicit SSE4.1
// feature check in select(), which establishes the exact target_feature. Array-only arguments cannot carry invalid extents.
macro_rules! dispatch {
    ($level:expr, $fun:ident $(::<$($c:ident),*>)?, $($arg:expr),*) => {{
        match $level {
            #[cfg(any(target_arch="x86",target_arch="x86_64"))]
            1 => Some(unsafe {sse41::$fun $(::<$($c),*>)? ($($arg),*)}),
            #[cfg(any(target_arch="x86",target_arch="x86_64"))]
            2 => Some(unsafe {avx2::$fun $(::<$($c),*>)? ($($arg),*)}),
            #[cfg(target_arch="aarch64")]
            3 => Some(unsafe {neon::$fun $(::<$($c),*>)? ($($arg),*)}),
            #[cfg(all(target_arch="wasm32",target_feature="simd128"))]
            4 => Some(wasm::$fun $(::<$($c),*>)? ($($arg),*)),
            _ => None,
        }
    }};
}
pub(super) fn width() -> usize {
    if level() == 2 {
        8
    } else {
        4
    }
}
pub(super) fn seed_batch(blocks: &[Block; 8]) -> Option<[Fit; 8]> {
    let _ = blocks;
    dispatch!(level(), seed, blocks)
}
#[inline]
pub(super) fn select<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    d: &Pair,
    lv: &Levels,
    idx: &mut [u8; 16],
    limit: u32,
    objective: Option<objective::Objective>,
) -> Option<u32> {
    // Every fit length uses the vector selector: one batch already covers
    // up to 4 or 8 representatives (short fits measured faster too).
    let _ = (&px, &reps, &d, &lv, &idx, &limit, &objective);
    dispatch!(
        level(),
        select::<C0, C1>,
        px,
        reps,
        d,
        lv,
        idx,
        limit,
        objective
    )
}

// Selector tables: the weights padded to 16 entries, and the midpoint
// thresholds w[m-1] + w[m] at index m - 1 (m = 1..N-1) with 255 at N - 1.
// Twice the projection exceeds threshold m exactly when entry m is nearer
// than entry m - 1, so the nearest entry counts the thresholds exceeded.
#[cfg_attr(
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
const fn padded<const N: usize>(w: [i32; N]) -> [i32; 16] {
    let mut out = [0; 16];
    let mut j = 0;
    while j < N {
        out[j] = w[j];
        j += 1;
    }
    out
}
const fn thresholds<const N: usize>(w: [i32; N]) -> [u8; 16] {
    let mut out = [0; 16];
    let mut m = 1;
    while m < N {
        out[m - 1] = (w[m - 1] + w[m]) as u8;
        m += 1;
    }
    out[N - 1] = 255;
    out
}
// The kernels guess g = clamp((k(N-1) + 32) >> 6, 1, N-1) and count only
// thresholds g and g + 1; this holds when the nearest entry (the `nearest`
// table the scalar selector reads) is within one of g for every k.
const fn guess_within_one<const N: usize>(w: [i32; N]) -> bool {
    let table = nearest(w);
    let t = thresholds(w);
    let mut k = 0;
    while k <= 64 {
        let mut g = (k * (N - 1) + 32) >> 6;
        if g < 1 {
            g = 1;
        }
        if g > N - 1 {
            g = N - 1;
        }
        let near = table[k] as usize;
        if near + 1 < g || near > g + 1 {
            return false;
        }
        let j = g - 1 + (2 * k > t[g - 1] as usize) as usize + (2 * k > t[g] as usize) as usize;
        if j != near {
            return false;
        }
        k += 1;
    }
    true
}
const _: () = assert!(guess_within_one(W2) && guess_within_one(W3) && guess_within_one(W4));
#[cfg_attr(
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
static WEIGHTS: [[i32; 16]; 3] = [padded(W2), padded(W3), padded(W4)];
#[cfg_attr(
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
static THRESHOLDS: [[u8; 16]; 3] = [thresholds(W2), thresholds(W3), thresholds(W4)];
#[cfg_attr(
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
fn level_tables(n: usize) -> (&'static [i32; 16], &'static [u8; 16]) {
    let slot = match n {
        4 => 0,
        8 => 1,
        _ => 2,
    };
    (&WEIGHTS[slot], &THRESHOLDS[slot])
}
// The weights as bytes (each is at most 64), padded to 16 entries.
const fn weight_bytes<const N: usize>(w: [i32; N]) -> [u8; 16] {
    let mut out = [0; 16];
    let mut j = 0;
    while j < N {
        out[j] = w[j] as u8;
        j += 1;
    }
    out
}
static WEIGHT_BYTES: [[u8; 16]; 3] = [weight_bytes(W2), weight_bytes(W3), weight_bytes(W4)];

/// The integer sums of a least-squares endpoint solve (see `least_squares`).
#[cfg_attr(
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
pub(super) struct LsSums {
    pub(super) aa: i32,
    pub(super) ab: i32,
    pub(super) bb: i32,
    pub(super) ax: [i32; 4],
    pub(super) bx: [i32; 4],
}

/// `least_squares` sums for the selectors `idx` on `lv`; `None` on the
/// scalar backend.
#[inline]
pub(super) fn ls_sums<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    idx: &[u8; 16],
    lv: &Levels,
) -> Option<LsSums> {
    let weights = &WEIGHT_BYTES[match lv.w.len() {
        4 => 0,
        8 => 1,
        _ => 2,
    }];
    let _ = (&px, &reps, &idx, &weights);
    dispatch!(level(), ls_sums::<C0, C1>, px, reps, idx, weights)
}
// Visibility bit i of a u16 mask, split by byte: bit i of the low byte for
// texels 0..8 and of the high byte for texels 8..16.
#[cfg_attr(
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
static VISIBLE_LOW: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0];
#[cfg_attr(
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )),
    allow(dead_code)
)]
static VISIBLE_HIGH: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, 128];
#[inline]
pub(super) fn polish<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    f: &mut Fit,
    prec: Prec,
    lv: &Levels,
    objective: Option<objective::Objective>,
) -> Option<bool> {
    // Short unweighted fits do not amortize SIMD staging and dispatch.
    // The caller retains the same scalar error, tie and cutoff rules.
    if objective.is_none() && reps.len <= 4 {
        return None;
    }
    let _ = (&px, &reps, &f, &prec, &lv, &objective);
    dispatch!(level(), polish::<C0, C1>, px, reps, f, prec, lv, objective)
}
// Every batch reads eight lanes in place; the padded arrays hold them.
const LANES: usize = super::search::PARTITION_LANES;
const _: () = assert!(LANES >= 64 - 4 + 8);
pub(super) fn partition_estimates<const CH: usize>(
    out: &mut [(f32, f32); 64],
    zero: &[Moments; LANES],
    one: &[Moments; LANES],
) -> bool {
    let selected = level();
    if selected == 0 {
        return false;
    }
    let width = if selected == 2 { 8 } else { 4 };
    for base in (0..64).step_by(width) {
        let (Some(z), Some(o)) = (
            zero[base..].first_chunk::<8>(),
            one[base..].first_chunk::<8>(),
        ) else {
            return false;
        };
        let _ = (&z, &o);
        let a: Option<[(f32, f32); 8]> = dispatch!(selected, estimates::<CH>, z);
        let b: Option<[(f32, f32); 8]> = dispatch!(selected, estimates::<CH>, o);
        let (Some(a), Some(b)) = (a, b) else {
            return false;
        };
        for l in 0..width {
            out[base + l] = (a[l].0 + b[l].0, a[l].1 + b[l].1);
        }
    }
    true
}

/// `subset_estimate::<CH>` of each of the first `count` subset moments, in
/// place in `out`; `false` (with `out` unspecified) on the scalar backend.
/// Every batch reads eight lanes, so `N` must exceed `count` by at least
/// four (by eight when `count` is not a multiple of four).
pub(super) fn subset_estimates<const CH: usize, const N: usize>(
    out: &mut [(f32, f32); N],
    moments: &[Moments; N],
    count: usize,
) -> bool {
    let selected = level();
    if selected == 0 {
        return false;
    }
    let width = if selected == 2 { 8 } else { 4 };
    for base in (0..count).step_by(width) {
        let Some(m) = moments[base..].first_chunk::<8>() else {
            return false;
        };
        let _ = &m;
        let e: Option<[(f32, f32); 8]> = dispatch!(selected, estimates::<CH>, m);
        let Some(e) = e else {
            return false;
        };
        let lanes = width.min(count - base);
        out[base..base + lanes].copy_from_slice(&e[..lanes]);
    }
    true
}

/// RGB and (when `RGBA`) RGBA principal estimates of the first `width()`
/// weighted texel sets, equal to `objective::SumTable`'s scalar estimates;
/// `None` on the scalar backend.
pub(super) fn weighted_estimates<const RGBA: bool>(
    sums: &[[i32; 17]; 8],
    cw: f32,
    aw: f32,
) -> Option<[[(f32, f32); 8]; 2]> {
    let _ = (&sums, &cw, &aw);
    dispatch!(level(), weighted_estimates::<RGBA>, sums, cw, aw)
}

pub(crate) fn hdr_distances(src: &[[[f64; 8]; 16]; 3]) -> [[[f64; 16]; 16]; 8] {
    match level() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        // level() establishes the exact ISA; all loads/stores use fixed arrays.
        1 => unsafe { sse41::hdr_distances(src) },
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        2 => unsafe { avx2::hdr_distances(src) },
        _ => crate::bc6h::distance_batch(src),
    }
}

fn detect() -> u8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "std")]
        {
            if std::is_x86_feature_detected!("avx2") {
                return 2;
            } else if std::is_x86_feature_detected!("sse4.1") {
                return 1;
            }
        }
        #[cfg(not(feature = "std"))]
        {
            if cfg!(target_feature = "avx2") {
                return 2;
            } else if cfg!(target_feature = "sse4.1") {
                return 1;
            }
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return 3;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return 4;
    }

    #[allow(unreachable_code)]
    0
}

#[cfg(all(
    test,
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )
))]
mod tests {
    use super::*;
    #[test]
    fn hdr_distances_match_scalar_bits() {
        let mut state = 17u32;
        for trial in 0..32 {
            let src = core::array::from_fn(|_| {
                core::array::from_fn(|_| {
                    core::array::from_fn(|_| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        if trial == 0 {
                            0.0
                        } else {
                            f64::from(state) / 1234567.0 - 1700.0
                        }
                    })
                })
            });
            let scalar = crate::bc6h::distance_batch(&src);
            let vector = hdr_distances(&src);
            for (a, b) in scalar
                .iter()
                .flatten()
                .flatten()
                .zip(vector.iter().flatten().flatten())
            {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }
    #[test]
    fn kernels_match_scalar_order_and_cutoffs() {
        let selected = level();
        let levels: &[u8] = if selected == 2 && sse41_detected() {
            &[1, 2]
        } else {
            core::slice::from_ref(&selected)
        };
        let mut rng = 7u32;
        for trial in 0..96 {
            let blocks = core::array::from_fn(|_| {
                core::array::from_fn(|_| {
                    core::array::from_fn(|_| {
                        rng ^= rng << 13;
                        rng ^= rng >> 17;
                        rng ^= rng << 5;
                        if trial == 0 {
                            0
                        } else if trial == 1 {
                            255
                        } else if trial % 3 == 2 {
                            // Few distinct colours: duplicate representatives.
                            rng as u8 & 0x80
                        } else {
                            rng as u8
                        }
                    })
                })
            });
            for &backend in levels {
                if backend == 0 {
                    continue;
                }
                let targets = dispatch!(backend, axis, &blocks).unwrap();
                let seeds = dispatch!(backend, seed, &blocks).unwrap();
                for l in 0..if backend == 2 { 8 } else { 4 } {
                    let expected =
                        fit::<0, 3>(&blocks[l], &Reps::distinct(&blocks[l]), P6, &L4, 1, 0, true);
                    assert_eq!(seeds[l].err, expected.err);
                    assert_eq!(seeds[l].d, expected.d);
                    assert_eq!(seeds[l].idx, expected.idx);
                }

                for l in 0..if backend == 2 { 8 } else { 4 } {
                    let reps = Reps::distinct(&blocks[l]);
                    assert_eq!(targets[l], axis_targets::<0, 3>(&blocks[l], &reps));
                }
                for px in &blocks {
                    let reps =
                        Reps::distinct(px).masked(if trial % 2 == 0 { 0xffff } else { 0x5317 });
                    for d in pairs(px) {
                        for lv in [&L2, &L3, &L4] {
                            for limit in [0, 1, 257, 100000, u32::MAX] {
                                let mut scalar = [91; 16];
                                let mut vector = scalar;
                                let a =
                                    select_scalar::<0, 4>(px, &reps, &d, lv, &mut scalar, limit);
                                let b = dispatch!(
                                    backend,
                                    select::<C0, C1>,
                                    px,
                                    &reps,
                                    &d,
                                    lv,
                                    &mut vector,
                                    limit,
                                    None
                                );
                                // The generic arguments are constants bound
                                // locally to keep the shared dispatch macro simple.
                                assert_eq!(Some(a), b);
                                assert_eq!(scalar, vector);
                                let mut scalar = [91; 16];
                                let mut vector = scalar;
                                let a =
                                    select_scalar::<0, 3>(px, &reps, &d, lv, &mut scalar, limit);
                                let b = dispatch!(
                                    backend,
                                    select::<C0, CH>,
                                    px,
                                    &reps,
                                    &d,
                                    lv,
                                    &mut vector,
                                    limit,
                                    None
                                );
                                assert_eq!(Some(a), b);
                                assert_eq!(scalar, vector);
                            }
                        }
                    }
                    // Least-squares sums equal their scalar loop.
                    for (slot, lv) in [&L2, &L3, &L4].into_iter().enumerate() {
                        let idx: [u8; 16] =
                            core::array::from_fn(|i| ((i * 7 + trial) % lv.w.len()) as u8);
                        let s = dispatch!(
                            backend,
                            ls_sums::<C0, CH>,
                            px,
                            &reps,
                            &idx,
                            &WEIGHT_BYTES[slot]
                        )
                        .unwrap();
                        let (mut aa, mut ab, mut bb) = (0, 0, 0);
                        let (mut ax, mut bx) = ([0i32; 4], [0i32; 4]);
                        for (i, n) in reps.iter() {
                            let b = lv.w[usize::from(idx[i])];
                            let a = 64 - b;
                            aa += a * a * n;
                            ab += a * b * n;
                            bb += b * b * n;
                            for c in 0..3 {
                                ax[c] += a * i32::from(px[i][c]) * n;
                                bx[c] += b * i32::from(px[i][c]) * n;
                            }
                        }
                        assert_eq!((s.aa, s.ab, s.bb, s.ax, s.bx), (aa, ab, bb, ax, bx));
                    }
                }
                for px in &blocks {
                    let mut reps = objective::Reps::distinct(px);
                    let weights = [0, 1, 2, 4];
                    let o = objective::Objective {
                        visible: 0x57a3 ^ (trial as u16).wrapping_mul(0x1111),
                        alpha: (trial % 4) as u8,
                        alpha_weight: weights[trial % 4],
                        colour_weight: weights[(trial / 4) % 3 + 1],
                    };
                    reps.objective = Some(o);
                    for d in pairs(px) {
                        for lv in [&L2, &L3, &L4] {
                            for limit in [0, 1, 1000, u32::MAX] {
                                let mut scalar = [91; 16];
                                let mut vector = scalar;
                                let a = objective::select_scalar::<0, 4>(
                                    px,
                                    &reps,
                                    &d,
                                    lv,
                                    &mut scalar,
                                    limit,
                                );
                                let b = dispatch!(
                                    backend,
                                    select::<C0, C1>,
                                    px,
                                    &reps,
                                    &d,
                                    lv,
                                    &mut vector,
                                    limit,
                                    Some(o)
                                );
                                assert_eq!(Some(a), b);
                                assert_eq!(scalar, vector);
                            }
                        }
                    }
                }
                let moments = blocks.map(|px| {
                    let rows = moment_rows::<3>(&px);
                    let mut m = Moments::default();
                    for row in rows {
                        for k in 0..10 {
                            m.m[k] += row[15].m[k];
                        }
                    }
                    m
                });
                let estimates = dispatch!(backend, estimates::<CH>, &moments).unwrap();
                for l in 0..if backend == 2 { 8 } else { 4 } {
                    let expected = subset_estimate::<3>(&moments[l]);
                    assert_eq!(estimates[l].0.to_bits(), expected.0.to_bits());
                    assert_eq!(estimates[l].1.to_bits(), expected.1.to_bits());
                }
            }
        }
    }
    // A fixed pair, a pair drawn from the block, and a zero-length pair.
    fn pairs(px: &Block) -> [Pair; 3] {
        let pixel = |i: usize| px[i].map(i32::from);
        [
            [[0, 31, 127, 255], [255, 77, 11, 0]],
            [pixel(3), pixel(9)],
            [pixel(5), pixel(5)],
        ]
    }
    const C0: usize = 0;
    const C1: usize = 4;
    const CH: usize = 3;
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
pub(crate) fn byte_lookup(values: &[u8; 16], min: u8, table: &[u8; 8]) -> Option<[u8; 16]> {
    if !std::arch::is_x86_feature_detected!("ssse3") {
        return None;
    }
    // SAFETY: this exact feature was checked; all extents are fixed arrays.
    Some(unsafe { byte_lookup_ssse3(values, min, table) })
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
fn byte_lookup_ssse3(values: &[u8; 16], min: u8, table: &[u8; 8]) -> [u8; 16] {
    use core::arch::x86_64::*;
    let mut out = [0; 16];
    // SAFETY: the loads read exactly sixteen and eight bytes respectively,
    // and the store writes exactly sixteen bytes to its fixed output array.
    unsafe {
        let v = _mm_loadu_si128(values.as_ptr().cast());
        let t = _mm_loadl_epi64(table.as_ptr().cast());
        let offsets = _mm_sub_epi8(v, _mm_set1_epi8(i8::from_ne_bytes([min])));
        _mm_storeu_si128(out.as_mut_ptr().cast(), _mm_shuffle_epi8(t, offsets));
    }
    out
}

/// Exact unsigned byte bounds for range fitting in other block formats.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn byte_bounds(values: &[u8; 16]) -> (u8, u8) {
    use core::arch::x86_64::*;
    // SAFETY: SSE2 is part of the x86_64 baseline. The fixed array provides
    // sixteen readable bytes; loadu imposes no additional alignment requirement.
    // All remaining operations are register-only, and only the low byte is read.
    unsafe {
        let v = _mm_loadu_si128(values.as_ptr().cast());
        let mut lo = _mm_min_epu8(v, _mm_srli_si128::<8>(v));
        let mut hi = _mm_max_epu8(v, _mm_srli_si128::<8>(v));
        lo = _mm_min_epu8(lo, _mm_srli_si128::<4>(lo));
        hi = _mm_max_epu8(hi, _mm_srli_si128::<4>(hi));
        lo = _mm_min_epu8(lo, _mm_srli_si128::<2>(lo));
        hi = _mm_max_epu8(hi, _mm_srli_si128::<2>(hi));
        lo = _mm_min_epu8(lo, _mm_srli_si128::<1>(lo));
        hi = _mm_max_epu8(hi, _mm_srli_si128::<1>(hi));
        (
            _mm_cvtsi128_si32(lo).to_le_bytes()[0],
            _mm_cvtsi128_si32(hi).to_le_bytes()[0],
        )
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
pub(crate) fn byte_search(values: &[u8; 16], bounds: &[u8; 7]) -> Option<[u8; 16]> {
    if !std::arch::is_x86_feature_detected!("ssse3") {
        return None;
    }
    // SAFETY: SSSE3 was explicitly checked, and extents are fixed arrays.
    Some(unsafe { byte_search_ssse3(values, bounds) })
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
fn byte_search_ssse3(values: &[u8; 16], bounds: &[u8; 7]) -> [u8; 16] {
    use core::arch::x86_64::*;
    let limits = [
        bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5], bounds[6], 0,
    ];
    let mut out = [0; 16];
    // SAFETY: fixed arrays provide the exact 16/8-byte input loads and the
    // 16-byte output store. PSHUFB only indexes registers. Wide-range bounds
    // are nonzero, so comparing x > bound-1 implements x >= bound exactly.
    unsafe {
        let sign = _mm_set1_epi8(i8::MIN);
        let one = _mm_set1_epi8(1);
        let x = _mm_xor_si128(_mm_loadu_si128(values.as_ptr().cast()), sign);
        let b = _mm_xor_si128(
            _mm_sub_epi8(_mm_loadl_epi64(limits.as_ptr().cast()), one),
            sign,
        );
        let c = _mm_cmpgt_epi8(x, _mm_shuffle_epi8(b, _mm_set1_epi8(3)));
        let mut lo = _mm_and_si128(c, _mm_set1_epi8(4));
        let c = _mm_cmpgt_epi8(x, _mm_shuffle_epi8(b, _mm_add_epi8(lo, one)));
        lo = _mm_add_epi8(lo, _mm_and_si128(c, _mm_set1_epi8(2)));
        let c = _mm_cmpgt_epi8(x, _mm_shuffle_epi8(b, lo));
        lo = _mm_add_epi8(lo, _mm_and_si128(c, one));
        let selectors = _mm_setr_epi8(1, 7, 6, 5, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        _mm_storeu_si128(out.as_mut_ptr().cast(), _mm_shuffle_epi8(selectors, lo));
    }
    out
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
pub(crate) fn range_channel(values: &[u8; 16]) -> Option<[u8; 8]> {
    if !std::arch::is_x86_feature_detected!("ssse3") {
        return None;
    }
    // SAFETY: SSSE3 was checked, and the reference has exactly sixteen bytes.
    Some(unsafe { range_channel_ssse3(values) })
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
fn range_channel_ssse3(values: &[u8; 16]) -> [u8; 8] {
    use core::arch::x86_64::*;
    let (min, max) = byte_bounds(values);
    let span = usize::from(max - min);
    // SAFETY: fixed-array extents permit the 16-byte source load, eight-byte
    // table loads and sixteen-byte packed-word store. All lookup is in registers.
    unsafe {
        let v = _mm_loadu_si128(values.as_ptr().cast());
        let minimum = _mm_set1_epi8(i8::from_ne_bytes([min]));
        let selectors = if span <= 7 {
            let table = &crate::bc4::NARROW_SELECTORS[span];
            _mm_shuffle_epi8(
                _mm_loadl_epi64(table.as_ptr().cast()),
                _mm_sub_epi8(v, minimum),
            )
        } else {
            let t = &crate::bc4::RANGE_BOUND_OFFSETS[span];
            let offsets = [t[0], t[1], t[2], t[3], t[4], t[5], t[6], 0];
            let one = _mm_set1_epi8(1);
            let sign = _mm_set1_epi8(i8::MIN);
            let x = _mm_xor_si128(v, sign);
            let limits = _mm_add_epi8(_mm_loadl_epi64(offsets.as_ptr().cast()), minimum);
            let b = _mm_xor_si128(_mm_sub_epi8(limits, one), sign);
            let c = _mm_cmpgt_epi8(x, _mm_shuffle_epi8(b, _mm_set1_epi8(3)));
            let mut pos = _mm_and_si128(c, _mm_set1_epi8(4));
            let c = _mm_cmpgt_epi8(x, _mm_shuffle_epi8(b, _mm_add_epi8(pos, one)));
            pos = _mm_add_epi8(pos, _mm_and_si128(c, _mm_set1_epi8(2)));
            let c = _mm_cmpgt_epi8(x, _mm_shuffle_epi8(b, pos));
            pos = _mm_add_epi8(pos, _mm_and_si128(c, one));
            let map = _mm_setr_epi8(1, 7, 6, 5, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            _mm_shuffle_epi8(map, pos)
        };
        let mut packed = _mm_or_si128(selectors, _mm_srli_epi64::<5>(selectors));
        packed = _mm_and_si128(packed, _mm_set1_epi64x(0x003f_003f_003f_003f));
        packed = _mm_or_si128(packed, _mm_srli_epi64::<10>(packed));
        packed = _mm_and_si128(packed, _mm_set1_epi64x(0x0000_0fff_0000_0fff));
        packed = _mm_or_si128(packed, _mm_srli_epi64::<20>(packed));
        packed = _mm_and_si128(packed, _mm_set1_epi64x(0x00ff_ffff));
        let mut words = [0u64; 2];
        _mm_storeu_si128(words.as_mut_ptr().cast(), packed);
        let bits = (words[0] | (words[1] << 24)).to_le_bytes();
        [
            max, min, bits[0], bits[1], bits[2], bits[3], bits[4], bits[5],
        ]
    }
}

// A checked capability lets block rows share one immutable ISA decision.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub(crate) struct RangeEncoder {
    _private: (),
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub(crate) fn range_encoder() -> Option<RangeEncoder> {
    if !std::arch::is_x86_feature_detected!("ssse3") {
        return None;
    }
    Some(RangeEncoder { _private: () })
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
impl RangeEncoder {
    #[inline]
    pub(crate) fn encode(&self, values: &[u8; 16]) -> [u8; 8] {
        // SAFETY: only range_encoder can construct this private capability,
        // after checking SSSE3. CPU features remain valid for its lifetime.
        unsafe { range_channel_ssse3(values) }
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3,sse4.1")]
fn range_channel_sse41(values: &[u8; 16]) -> [u8; 8] {
    use core::arch::x86_64::*;
    // SAFETY: the source is a fixed sixteen-byte array and both target
    // features were established by the checked capability below.
    let v = unsafe { _mm_loadu_si128(values.as_ptr().cast()) };
    fit_sse41(v).to_le_bytes()
}

// Fits one channel held in a register: endpoints in the low two bytes, then
// the 48 selector bits. The register never round-trips through memory.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3,sse4.1")]
#[inline]
fn fit_sse41(v: core::arch::x86_64::__m128i) -> u64 {
    use core::arch::x86_64::*;
    let (min, max) = {
        let upper = _mm_srli_si128::<8>(v);
        let lo = _mm_cvtepu8_epi16(_mm_min_epu8(v, upper));
        let hi = _mm_cvtepu8_epi16(_mm_xor_si128(_mm_max_epu8(v, upper), _mm_set1_epi8(-1)));
        let min = _mm_cvtsi128_si32(_mm_minpos_epu16(lo)).to_le_bytes()[0];
        let inverse_max = _mm_cvtsi128_si32(_mm_minpos_epu16(hi)).to_le_bytes()[0];
        (min, 255 - inverse_max)
    };
    let span = usize::from(max - min);
    // SAFETY: fixed-array extents permit the eight-byte table loads.
    // All lookup is in registers.
    unsafe {
        let minimum = _mm_set1_epi8(i8::from_ne_bytes([min]));
        let selectors = if span <= 7 {
            let table = &crate::bc4::NARROW_SELECTORS[span];
            _mm_shuffle_epi8(
                _mm_loadl_epi64(table.as_ptr().cast()),
                _mm_sub_epi8(v, minimum),
            )
        } else {
            let offsets = &crate::bc4::RANGE_BOUND_VECTORS[span];
            let one = _mm_set1_epi8(1);
            let sign = _mm_set1_epi8(i8::MIN);
            let x = _mm_xor_si128(v, sign);
            let limits = _mm_add_epi8(_mm_loadl_epi64(offsets.as_ptr().cast()), minimum);
            let b = _mm_xor_si128(_mm_sub_epi8(limits, one), sign);
            let c = _mm_cmpgt_epi8(x, _mm_shuffle_epi8(b, _mm_set1_epi8(3)));
            let mut pos = _mm_and_si128(c, _mm_set1_epi8(4));
            let c = _mm_cmpgt_epi8(x, _mm_shuffle_epi8(b, _mm_add_epi8(pos, one)));
            pos = _mm_add_epi8(pos, _mm_and_si128(c, _mm_set1_epi8(2)));
            let c = _mm_cmpgt_epi8(x, _mm_shuffle_epi8(b, pos));
            pos = _mm_add_epi8(pos, _mm_and_si128(c, one));
            let map = _mm_setr_epi8(1, 7, 6, 5, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            _mm_shuffle_epi8(map, pos)
        };
        let mut packed = _mm_or_si128(selectors, _mm_srli_epi64::<5>(selectors));
        packed = _mm_and_si128(packed, _mm_set1_epi64x(0x003f_003f_003f_003f));
        packed = _mm_or_si128(packed, _mm_srli_epi64::<10>(packed));
        packed = _mm_and_si128(packed, _mm_set1_epi64x(0x0000_0fff_0000_0fff));
        packed = _mm_or_si128(packed, _mm_srli_epi64::<20>(packed));
        packed = _mm_and_si128(packed, _mm_set1_epi64x(0x00ff_ffff));
        let low = _mm_cvtsi128_si64(packed).cast_unsigned();
        let high = _mm_extract_epi64::<1>(packed).cast_unsigned();
        u64::from(max) | (u64::from(min) << 8) | ((low | (high << 24)) << 16)
    }
}

// Fits every complete block of one block row in a single feature-enabled
// frame, deinterleaving R and G with byte shuffles straight into registers.
// `CHANNELS` is 2 (RG8) or 4 (RGBA8); `rows` are the byte offsets of the four
// source rows. Output is identical to gathering R and G into arrays and
// fitting each with `fit_sse41`.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3,sse4.1")]
fn peak_row_sse41<const CHANNELS: usize>(
    data: &[u8],
    rows: [usize; 4],
    blocks: usize,
    dst: &mut [u8],
) {
    use core::arch::x86_64::*;
    for (bx, out) in dst
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .take(blocks)
        .enumerate()
    {
        let start = bx * 4 * CHANNELS;
        let [r0, r1, r2, r3] = rows.map(|row| &data[row + start..row + start + 4 * CHANNELS]);
        // SAFETY: each source slice holds exactly 4 * CHANNELS bytes, which
        // covers the 8-byte (RG8) or 16-byte (RGBA8) unaligned load; the
        // remaining intrinsics are register-only under the enabled features.
        let (red, green) = unsafe {
            if CHANNELS == 2 {
                let split = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
                let rows01 = _mm_shuffle_epi8(
                    _mm_unpacklo_epi64(
                        _mm_loadl_epi64(r0.as_ptr().cast()),
                        _mm_loadl_epi64(r1.as_ptr().cast()),
                    ),
                    split,
                );
                let rows23 = _mm_shuffle_epi8(
                    _mm_unpacklo_epi64(
                        _mm_loadl_epi64(r2.as_ptr().cast()),
                        _mm_loadl_epi64(r3.as_ptr().cast()),
                    ),
                    split,
                );
                // Each half is [R row a, R row b | G row a, G row b].
                (
                    _mm_unpacklo_epi64(rows01, rows23),
                    _mm_unpackhi_epi64(rows01, rows23),
                )
            } else {
                let split = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
                let row = |p: &[u8]| _mm_shuffle_epi8(_mm_loadu_si128(p.as_ptr().cast()), split);
                // Dwords are [R, G, B, A] of four texels in one source row.
                let rows01 = _mm_unpacklo_epi32(row(r0), row(r1));
                let rows23 = _mm_unpacklo_epi32(row(r2), row(r3));
                (
                    _mm_unpacklo_epi64(rows01, rows23),
                    _mm_unpackhi_epi64(rows01, rows23),
                )
            }
        };
        out[..8].copy_from_slice(&fit_sse41(red).to_le_bytes());
        out[8..].copy_from_slice(&fit_sse41(green).to_le_bytes());
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub(crate) struct Sse41RangeEncoder {
    _private: (),
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub(crate) fn range_encoder_sse41() -> Option<Sse41RangeEncoder> {
    if !std::arch::is_x86_feature_detected!("ssse3")
        || !std::arch::is_x86_feature_detected!("sse4.1")
    {
        return None;
    }
    Some(Sse41RangeEncoder { _private: () })
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
impl Sse41RangeEncoder {
    #[inline]
    pub(crate) fn encode(&self, values: &[u8; 16]) -> [u8; 8] {
        // SAFETY: private construction establishes both SSSE3 and SSE4.1.
        unsafe { range_channel_sse41(values) }
    }

    /// Fits the first `blocks` complete 4x4 blocks of one block row.
    ///
    /// `rows` are the byte offsets of the four source rows in `data`, `CHANNELS`
    /// is 2 (RG8) or 4 (RGBA8), and `dst` receives 16 bytes per block.
    #[inline]
    pub(crate) fn encode_row<const CHANNELS: usize>(
        &self,
        data: &[u8],
        rows: [usize; 4],
        blocks: usize,
        dst: &mut [u8],
    ) {
        // SAFETY: private construction establishes both SSSE3 and SSE4.1.
        unsafe { peak_row_sse41::<CHANNELS>(data, rows, blocks, dst) }
    }
}
