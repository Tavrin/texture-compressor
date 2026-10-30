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
fn level() -> u8 {
    let cached = LEVEL.load(Ordering::Relaxed);
    if cached != 255 {
        return cached;
    }
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
pub(super) fn select<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    d: &Pair,
    lv: &Levels,
    idx: &mut [u8; 16],
    limit: u32,
    objective: Option<objective::Objective>,
) -> Option<u32> {
    // Short unweighted fits do not amortize SIMD staging and dispatch.
    // The caller retains the same scalar error, tie and cutoff rules.
    if objective.is_none() && reps.len <= 4 {
        return None;
    }
    let _ = (&px, &reps, &d, &lv, &idx, &limit, &objective);
    let selected = level();
    // An unweighted selector checks at most three neighbouring entries.
    // Four lanes suffice; retain eight lanes for weighted full palettes.
    // AVX2 is not used as evidence of SSE4.1: check that feature explicitly.
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    if selected == 2 && objective.is_none() && sse41_detected() {
        return dispatch!(1, select::<C0, C1>, px, reps, d, lv, idx, limit, objective);
    }
    dispatch!(
        selected,
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
pub(super) fn partition_estimates<const CH: usize>(
    out: &mut [(f32, f32); 64],
    subsets: impl Fn(usize) -> [Moments; 2],
) -> bool {
    let selected = level();
    if selected == 0 {
        return false;
    }
    let width = if selected == 2 { 8 } else { 4 };
    for base in (0..64).step_by(width) {
        let parts: [_; 8] = core::array::from_fn(|l| subsets((base + l).min(63)));
        let zero = parts.map(|p| p[0]);
        let one = parts.map(|p| p[1]);
        let _ = (&zero, &one);
        let a: [(f32, f32); 8] = dispatch!(selected, estimates::<CH>, &zero).unwrap();
        let b: [(f32, f32); 8] = dispatch!(selected, estimates::<CH>, &one).unwrap();
        for l in 0..width {
            out[base + l] = (a[l].0 + b[l].0, a[l].1 + b[l].1);
        }
    }
    true
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
                    let d = [[0, 31, 127, 255], [255, 77, 11, 0]];
                    for lv in [&L2, &L3, &L4] {
                        for limit in [0, 1, 257, 100000, u32::MAX] {
                            let mut scalar = [91; 16];
                            let mut vector = scalar;
                            let a = select_scalar::<0, 4>(px, &reps, &d, lv, &mut scalar, limit);
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
                            // The generic arguments below are constants bound
                            // locally to keep the shared dispatch macro simple.
                            assert_eq!(Some(a), b);
                            assert_eq!(scalar, vector);
                        }
                    }
                }
                for px in &blocks {
                    let mut reps = objective::Reps::distinct(px);
                    let o = objective::Objective {
                        visible: 0x57a3,
                        alpha: 2,
                        alpha_weight: 4,
                    };
                    reps.objective = Some(o);
                    let d = [[0, 31, 127, 255], [255, 77, 11, 0]];
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
    const C0: usize = 0;
    const C1: usize = 4;
    const CH: usize = 3;
}
