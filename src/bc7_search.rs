//! Balanced and Quality BC7 searches.
//!
//! Both presets reuse the Fast fits and widen the search: ranked rotations of
//! modes 4 and 5, several two-subset partitions per mode ranked by the
//! moment estimate (and three-subset partitions for Quality), p-bit choices
//! for the modes that store them, and extra least-squares and polish passes
//! on the candidates that are kept. Estimates only decide what is fitted;
//! retained candidates carry their exact decoded error.
#![allow(
    clippy::needless_range_loop,
    reason = "channel loops index several per-channel arrays over const-generic channel ranges"
)]

use super::super::tables::{ANCHOR3A, ANCHOR3B};
use super::super::MASKS3;
use super::*;

// Moments of both subsets of every two-subset partition, kept so fitted
// subsets reuse the sums that ranked them. Lanes 64.. repeat partition 63
// so that SIMD batches of four or eight never leave the arrays.
struct Partitions {
    est: [(f32, f32); 64],
    rows: [[Moments; 16]; 4],
    zero: [Moments; PARTITION_LANES],
    one: [Moments; PARTITION_LANES],
}

/// Two-subset partitions plus padding up to a whole SIMD batch.
pub(super) const PARTITION_LANES: usize = 68;

impl Partitions {
    fn new<const CH: usize>(px: &Block) -> Self {
        let len = moment_len(CH);
        let rows = moment_rows::<CH>(px);
        let mut total = Moments::default();
        for row in &rows {
            for k in 0..len {
                total.m[k] += row[15].m[k];
            }
        }
        // Integer sums: the order of the additions cannot change a moment.
        // Every entry is written once, without clearing the arrays first.
        let one: [Moments; PARTITION_LANES] = core::array::from_fn(|p| {
            let mask = MASKS2[p.min(63)][0];
            let mut one = rows[0][usize::from(mask & 15)];
            for row in 1..4 {
                let part = &rows[row][usize::from((mask >> (4 * row)) & 15)];
                for k in 0..len {
                    one.m[k] += part.m[k];
                }
            }
            one
        });
        let zero = core::array::from_fn(|p| {
            let mut zero = Moments::default();
            for k in 0..len {
                zero.m[k] = total.m[k] - one[p].m[k];
            }
            zero
        });
        let mut data = Self {
            est: [(0.0, 0.0); 64],
            rows,
            zero,
            one,
        };
        #[cfg(feature = "simd")]
        if simd::partition_estimates::<CH>(&mut data.est, &data.zero, &data.one) {
            return data;
        }
        for p in 0..64 {
            let a = subset_estimate::<CH>(&data.zero[p]);
            let b = subset_estimate::<CH>(&data.one[p]);
            data.est[p] = (a.0 + b.0, a.1 + b.1);
        }
        data
    }

    fn subsets<const CH: usize>(&self, partition: usize) -> [Moments; 2] {
        [self.zero[partition], self.one[partition]]
    }

    fn targets<const CH: usize>(
        &self,
        px: &Block,
        reps: &[Reps; 2],
        partition: usize,
    ) -> [Pair; 2] {
        let subsets = self.subsets::<CH>(partition);
        [0, 1].map(|s| {
            let m = &subsets[s].m;
            let n = m[0];
            let mut mean = [0.0; 4];
            let mut cov = [[0.0; 4]; 4];
            let mut k = 1 + CH;
            for c in 0..CH {
                mean[c] = m[1 + c] as f32 / n as f32;
                for e in c..CH {
                    let v = (n * m[k] - m[1 + c] * m[1 + e]) as f32;
                    cov[c][e] = v;
                    cov[e][c] = v;
                    k += 1;
                }
            }
            axis_endpoints::<0, CH>(px, &reps[s], &mean, &cov)
        })
    }
}

// `best_partitions` for modes 3 (4 selector levels) and 1 (8 levels) in one
// pass over the shared estimates; each list is ordered exactly as there.
fn best_partitions_31(est: &[(f32, f32); 64], counts: [usize; 2]) -> [[(f32, usize); 16]; 2] {
    let steps = [1.0 / 9.0, 1.0 / 49.0];
    let mut top = [[(f32::MAX, 0usize); 16]; 2];
    for (p, &(res, var)) in est.iter().enumerate() {
        for r in 0..2 {
            let count = counts[r].min(16);
            if count == 0 {
                continue;
            }
            let e = res + var * steps[r];
            let top = &mut top[r];
            if e < top[count - 1].0 {
                let mut k = count - 1;
                while k > 0 && e < top[k - 1].0 {
                    top[k] = top[k - 1];
                    k -= 1;
                }
                top[k] = (e, p);
            }
        }
    }
    top
}

// Mode 0: four stored bits plus a unique p-bit. No preset searches the
// other three-subset mode (2): its ranking does not fit a block budget,
// and Intel basic does not search it either.
pub(super) const P0: Prec = Prec {
    bits: 5,
    pbit: PBit::Unique,
};

// The three subset masks of a three-subset partition.
fn masks3(partition: usize) -> [u16; 3] {
    let [m1, m2] = MASKS3[partition];
    [!(m1 | m2), m1, m2]
}

// Three-subset partitions ranked by `partition_estimates3`, and their subset
// moments plus padding up to a whole SIMD batch.
const RANKED3: usize = 16;
const RANKED3_LANES: usize = 3 * RANKED3 + 8;

// Estimated (residual, principal variance) of the first `count` (at most
// RANKED3) three-subset partitions from the RGB row moments of the
// two-subset ranking; the rest are left unranked.
fn partition_estimates3(lut: &[[Moments; 16]; 4], count: usize) -> [(f32, f32); 64] {
    let count = count.min(RANKED3);
    let len = moment_len(3);
    let mut total = Moments::default();
    for row in lut {
        for k in 0..len {
            total.m[k] += row[15].m[k];
        }
    }
    // Subsets zero, one and two of partition p at 3p, 3p + 1 and 3p + 2.
    let mut subsets = [Moments::default(); RANKED3_LANES];
    for p in 0..count {
        let mut parts = [Moments::default(); 2];
        for (s, part) in parts.iter_mut().enumerate() {
            let mask = MASKS3[p][s];
            for (row, table) in lut.iter().enumerate() {
                let m = &table[usize::from((mask >> (4 * row)) & 15)];
                for k in 0..len {
                    part.m[k] += m.m[k];
                }
            }
        }
        let mut zero = total;
        for k in 0..len {
            zero.m[k] -= parts[0].m[k] + parts[1].m[k];
        }
        subsets[3 * p] = zero;
        subsets[3 * p + 1] = parts[0];
        subsets[3 * p + 2] = parts[1];
    }
    let mut est = [(0f32, 0f32); RANKED3_LANES];
    #[cfg(feature = "simd")]
    let done = simd::subset_estimates::<3, RANKED3_LANES>(&mut est, &subsets, 3 * count);
    #[cfg(not(feature = "simd"))]
    let done = false;
    if !done {
        for (e, m) in est.iter_mut().zip(&subsets).take(3 * count) {
            *e = subset_estimate::<3>(m);
        }
    }
    let mut out = [(f32::MAX, 0f32); 64];
    for (slot, e) in out.iter_mut().zip(est.as_chunks::<3>().0).take(count) {
        let mut sum = (0f32, 0f32);
        for &(res, var) in e {
            sum.0 += res;
            sum.1 += var;
        }
        *slot = sum;
    }
    out
}

// Mode 0 fit of each subset of a three-subset partition.
fn fit_partition3(
    px: &Block,
    all: &Reps,
    partition: usize,
    iters: usize,
    rounds: usize,
) -> [Fit; 3] {
    masks3(partition).map(|mask| fit::<0, 3>(px, &all.masked(mask), P0, &L3, iters, rounds, false))
}

// Three-subset mode 0.
fn pack3(partition: usize, mut s: [Fit; 3]) -> [u8; 16] {
    // Four stored endpoint bits plus a unique p-bit, 3-bit selectors.
    let width = 4;
    let masks = masks3(partition);
    let anchors = [0, ANCHOR3A[partition], ANCHOR3B[partition]];
    for k in 0..3 {
        anchor(&mut s[k], masks[k], anchors[k], 8);
    }
    let mut bits = 0u128;
    let mut o = 0;
    put(&mut bits, &mut o, 1, 1);
    put(&mut bits, &mut o, partition as u32, 4);
    for c in 0..3 {
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
        for e in &f.d {
            put(&mut bits, &mut o, ((e[0] >> (7 - width)) & 1) as u32, 1);
        }
    }
    for i in 0..16 {
        let k = masks.iter().position(|m| m >> i & 1 == 1).unwrap_or(0);
        put(
            &mut bits,
            &mut o,
            u32::from(s[k].idx[i]),
            3 - u32::from(anchors[k] == i),
        );
    }
    debug_assert_eq!(o, 128);
    bits.to_le_bytes()
}

/// Ceilings for the general search. Sparse opaque precision may reuse the
/// Quality allowance; block-local difficulty reduces each chosen allowance.
#[derive(Clone, Copy)]
pub(crate) struct Search {
    quality: bool,
    axis_rounds: usize,
    duals: [usize; 2],
    rough_iters: usize,
    mode4_iters: usize,
    parts: [usize; 3],
    /// Three-subset mode 0 partitions fitted.
    parts3: usize,
    pub(super) refine_iters: usize,
    pub(super) refine_rounds: usize,
    /// Other p-bit assignments refitted during refinement, best quantized first.
    pub(super) pbit_tries: usize,
    /// Whether a near-tied second partition per mode is also refined.
    runner_up: bool,
    /// Work ceiling of one block in representative passes (see [`Work`]).
    /// It is the same for every block; allowances above only order and
    /// limit the candidates that compete for it.
    budget: u32,
    /// Whether colour under zero alpha is unscored (opt-in policy); by
    /// default every texel's colour counts (straight alpha).
    hide_transparent: bool,
}

pub(crate) const BALANCED: Search = Search {
    quality: false,
    axis_rounds: 1,
    duals: [1, 2],
    rough_iters: BALANCED_ROUGH_ITERS,
    mode4_iters: 1,
    parts: [
        4 * BALANCED_PARTS_PER_UNIT,
        4 * BALANCED_PARTS_PER_UNIT,
        8 * BALANCED_PARTS_PER_UNIT,
    ],
    parts3: 1,
    refine_iters: BALANCED_REFINE_ITERS,
    refine_rounds: 1,
    pbit_tries: BALANCED_PBIT_TRIES,
    runner_up: BALANCED_RUNNER_UP,
    budget: BALANCED_BUDGET,
    hide_transparent: false,
};

pub(crate) const QUALITY: Search = Search {
    quality: true,
    axis_rounds: 2,
    duals: [4, 4],
    rough_iters: 1,
    mode4_iters: 2,
    parts: [8, 8, 16],
    parts3: 4,
    refine_iters: 4,
    refine_rounds: 2,
    pbit_tries: 3,
    runner_up: true,
    budget: QUALITY_BUDGET,
    hide_transparent: false,
};

const BALANCED_BUDGET: u32 = 1300;
const QUALITY_BUDGET: u32 = 2000;
const BALANCED_PBIT_TRIES: usize = 3;
const BALANCED_RUNNER_UP: bool = true;
const BALANCED_REFINE_ITERS: usize = 3;
const ESCALATE_BREADTH: bool = true;
// Two-subset partitions per allowance unit, and least-squares passes per rough
// fit, in Balanced.
const BALANCED_PARTS_PER_UNIT: usize = 1;
const BALANCED_ROUGH_ITERS: usize = 1;
// At least this many mode 5 rotations and mode 4 rotation/selection pairs
// are fitted for nonopaque Balanced blocks.
const BALANCED_VISIBLE_DUALS: [usize; 2] = [0, 0];
// Percent of the work left after mode 6 that only refinement may spend.
const REFINE_SHARE: u32 = 40;

// Visible colour error weight relative to alpha in nonopaque blocks.
const COLOUR_WEIGHT: u8 = 1;
// Whether the dual-mode estimate of a nonopaque block charges alpha
// selector steps only to texels strictly between its extreme alphas.
const VISIBLE_EXTREME_ALPHA: bool = true;
// Peak-aware final choice: a block whose winner has a channel error of at
// least PEAK_FLOOR takes the finalist with the lowest peak if that peak is
// at least PEAK_GAIN lower and its squared error is at most 1/PEAK_SLACK
// above the winner's. A texture's peak error is its worst block's.
const PEAK_FLOOR: u8 = 6;
const PEAK_GAIN: u8 = 1;
const PEAK_SLACK: u32 = 8;

// Peak second look of an opaque block (see `peak_look`): a winner with a
// channel error of at least PEAK_LOOK_FLOOR re-ranks its rough candidates by
// peak and refines the PEAK_LOOK_REFINE lowest; one of them replaces the
// winner if its peak is lower and its squared error at most 1/PEAK_LOOK_SLACK
// above the winner's. Measured on the development set, fitting further
// partitions for the look, a lower floor and a look for nonopaque blocks
// did not lower guard failures (the last raised other channels' peaks).
const PEAK_LOOK: bool = true;
const PEAK_LOOK_FLOOR: u8 = 10;
const PEAK_LOOK_REFINE: usize = 2;
const PEAK_LOOK_SLACK: u32 = 4;
// Refined finalists of an opaque winner whose peak reaches PEAK_LOOK_FLOOR
// compete under the look's bound too (see `peak_choice`).
const PEAK_WIDE_FINALISTS: bool = true;
// A lower peak may not raise the colour peak (see `peak_groups_hold`).
const PEAK_GROUPS: bool = true;
// Rough fits are refined while within a quarter of the best refined error;
// up to LOW_REFINE_EXTRA more while within LOW_REFINE_SLACK of it (see the
// refinement loop). Measured on the development set, slacks of 32, 64 and
// 128 gave the same guard results; one extra candidate kept them at about
// the earlier Quality time, while no count limit cost about 10% on the
// slowest opaque textures.
const LOW_REFINE_SLACK: u32 = 32;
const LOW_REFINE_EXTRA: usize = 1;

// Up to eight refined candidates of one block, with their errors.
type Finalists = [Option<(u32, Cand)>; 8];
// Rough fits of one opaque block: at most one per planned fit.
const ROUGH_SLOTS: usize = 32;
// Candidates of the peak second look, with their errors.
type Look = [Option<(u32, Cand)>; PEAK_LOOK_REFINE];

fn push_finalist(finalists: &mut Finalists, candidate: (u32, Cand)) {
    if let Some(slot) = finalists.iter_mut().find(|f| f.is_none()) {
        *slot = Some(candidate);
    }
}

// Squared error stays the objective; the per-block peak decides only among
// near-equal finalists, and only where the peak is large enough to set a
// texture's maximum error. Second-look candidates compete under their own,
// wider error bound.
// `peak` is the winner's. `wide_finalists` (opaque blocks) gives finalists
// the look's bound at PEAK_LOOK_FLOOR; `protect_colour` applies
// `peak_groups_hold`.
fn peak_choice(
    px: &Block,
    best: (u32, Cand),
    peak: u8,
    finalists: &Finalists,
    look: &Look,
    wide_finalists: bool,
    protect_colour: bool,
) -> (u32, Cand) {
    if peak < PEAK_FLOOR {
        return best;
    }
    let near = best.0.saturating_add(best.0 / PEAK_SLACK);
    let wide = best.0.saturating_add(best.0 / PEAK_LOOK_SLACK);
    // A winner whose peak reaches the second-look floor lets its refined
    // finalists compete under the second look's bound as well: the look
    // skips candidates the search already refined.
    let finalist_limit = if PEAK_WIDE_FINALISTS && wide_finalists && peak >= PEAK_LOOK_FLOOR {
        wide
    } else {
        near
    };
    let mut pick = (peak, best);
    let tiers = [(finalist_limit, &finalists[..]), (wide, &look[..])];
    // Colour and alpha peaks of the winner; see `peak_groups_hold`.
    let held = best.1.peaks(px);
    for (limit, list) in tiers {
        for &(err, cand) in list.iter().flatten() {
            if err > limit {
                continue;
            }
            let groups = cand.peaks(px);
            let p = groups[0].max(groups[1]);
            // `peak - p >= PEAK_GAIN` without u8 overflow when p is 255.
            if peak.checked_sub(p).is_some_and(|gain| gain >= PEAK_GAIN)
                && (!protect_colour || peak_groups_hold(held, groups))
                && (p, err) < (pick.0, pick.1 .0)
            {
                pick = (p, (err, cand));
            }
        }
    }
    pick.1
}

// Colour and alpha peaks of a texture differ in scale: an alpha edge is a
// step that every encoder misses by more than smooth colour, so a block's
// alpha peak rarely sets the texture's alpha maximum while its colour peak
// can set the colour maximum. A lower block peak therefore may not move the
// error into colour: the candidate's colour peak may rise above the winner's
// only up to the candidate's own alpha peak. Lowering the colour peak at the
// cost of alpha stays allowed. Opaque blocks have no alpha error.
// `[colour, alpha]` peaks, as `Cand::peaks` returns them.
fn peak_groups_hold(winner: [u8; 2], candidate: [u8; 2]) -> bool {
    !PEAK_GROUPS || candidate[0] <= winner[0].max(candidate[1])
}

// The block with channel c read from channel map[c] (a dual-mode rotation).
#[inline]
fn rotate(px: &Block, map: [usize; 4]) -> Block {
    let mut out = [[0u8; 4]; 16];
    for (o, p) in out.iter_mut().zip(px) {
        for c in 0..4 {
            o[c] = p[map[c]];
        }
    }
    out
}

// Identity of a rough candidate: its mode and partition, or its mode,
// rotation and index selection.
fn rough_key(r: &Rough) -> (usize, usize, usize) {
    match *r {
        Rough::Dual((_, mode, rotation, sel, _, _)) => (mode, rotation, sel),
        Rough::Two((_, mode, partition, _)) => (mode, partition, 0),
        Rough::Three((_, partition, _)) => (0, partition, 0),
    }
}

// The candidate a rough fit stands for, unrefined.
fn rough_candidate(px: &Block, r: Rough, flat_extreme_alpha: bool) -> (u32, Cand) {
    match r {
        Rough::Dual((err, mode, rotation, sel, color, alpha)) => {
            let compact = flat_extreme_alpha && rotation > 0;
            (
                err,
                Cand::Dual {
                    mode,
                    rotation,
                    sel,
                    color: restore_color(color, rotation, compact, px[0][3]),
                    alpha,
                },
            )
        }
        Rough::Two((err, mode, partition, s)) => (err, Cand::Two { mode, partition, s }),
        Rough::Three((err, partition, s)) => (err, Cand::Three { partition, s }),
    }
}

// Refine one rough candidate of an opaque block while the remaining work
// pays for it; otherwise it competes with its rough fit and exact error.
fn refine_rough(
    px: &Block,
    reps: &Reps,
    r: Rough,
    cfg: &Search,
    work: &mut Work,
    flat_extreme_alpha: bool,
) -> (u32, Cand) {
    let n = reps.len as u32;
    match r {
        Rough::Dual((_, mode, rotation, sel, mut color, mut alpha)) => {
            let compact = flat_extreme_alpha && rotation > 0;
            let charge = 2 * improve_work(n, cfg.refine_iters, cfg.refine_rounds);
            if work.take(charge) {
                let map = dual_map(rotation, compact);
                let rotated = rotate(px, map);
                let passes = refine_dual(
                    &rotated, reps, mode, sel, &mut color, &mut alpha, cfg, compact,
                );
                work.refund(charge, passes * n);
            }
            (
                color.err + alpha.err,
                Cand::Dual {
                    mode,
                    rotation,
                    sel,
                    color: restore_color(color, rotation, compact, px[0][3]),
                    alpha,
                },
            )
        }
        Rough::Two((_, mode, partition, mut s)) => {
            let m1 = MASKS2[partition][0];
            let charge = refine_work(mode, subset_len(reps, &[!m1, m1]), cfg, true);
            if work.take(charge) {
                let mut used = 0;
                for (k, mask) in [!m1, m1].into_iter().enumerate() {
                    let r = reps.masked(mask);
                    used += r.len as u32 * refine_subset(px, mode, &r, &mut s[k], true, cfg);
                }
                work.refund(charge, used);
            }
            (s[0].err + s[1].err, Cand::Two { mode, partition, s })
        }
        Rough::Three((_, partition, mut s)) => {
            let masks = masks3(partition);
            let charge = refine_work(0, subset_len(reps, &masks), cfg, true);
            if work.take(charge) {
                let mut used = 0;
                for (k, mask) in masks.into_iter().enumerate() {
                    let r = reps.masked(mask);
                    used += r.len as u32 * refine_subset(px, 0, &r, &mut s[k], true, cfg);
                }
                work.refund(charge, used);
            }
            (s[0].err + s[1].err + s[2].err, Cand::Three { partition, s })
        }
    }
}

// The squared-error search keeps few rough candidates per mode, so a block
// whose winner leaves one large channel error rarely refines the partition
// or mode that would avoid it. When the winner's peak reaches
// PEAK_LOOK_FLOOR, every rough fit of the search that was not refined is
// re-ranked by its peak (then its error); the PEAK_LOOK_REFINE lowest peaks
// below the winner's, with a rough error at most a quarter above the
// winner's, are refined. Peaks are paid as one pass each and refinement as
// in the refinement pool, from the block's remaining work (`refine` refines
// one rough candidate and charges it). `n` is the block's representatives.
#[allow(
    clippy::too_many_arguments,
    reason = "the second look reads the search state of one block"
)]
fn peak_look(
    px: &Block,
    n: u32,
    (best_err, peak): (u32, u8),
    roughs: &[Option<Rough>],
    refined: &[Option<Rough>],
    work: &mut Work,
    flat_extreme_alpha: bool,
    mut refine: impl FnMut(Rough, &mut Work) -> (u32, Cand),
) -> Look {
    let mut look: Look = [None; PEAK_LOOK_REFINE];
    if !PEAK_LOOK || PEAK_LOOK_REFINE == 0 || peak < PEAK_LOOK_FLOOR {
        return look;
    }
    let limit = best_err.saturating_add(best_err / 4);
    // (peak, rough error, slot), ordered by peak then error; ties keep the
    // search order.
    let mut ranked = [(0u8, 0u32, 0usize); ROUGH_SLOTS];
    let mut len = 0;
    for (slot, r) in roughs.iter().enumerate().take(ROUGH_SLOTS) {
        let Some(r) = r else {
            continue;
        };
        let key = rough_key(r);
        if r.err() > limit || refined.iter().flatten().any(|k| rough_key(k) == key) {
            continue;
        }
        if !work.take(n) {
            break;
        }
        let p = rough_candidate(px, *r, flat_extreme_alpha).1.peak(px);
        if p >= peak {
            continue;
        }
        let mut at = len;
        while at > 0 && (p, r.err()) < (ranked[at - 1].0, ranked[at - 1].1) {
            ranked[at] = ranked[at - 1];
            at -= 1;
        }
        ranked[at] = (p, r.err(), slot);
        len += 1;
    }
    for (out, &(_, _, slot)) in look.iter_mut().zip(&ranked[..len]) {
        if let Some(r) = roughs[slot] {
            *out = Some(refine(r, work));
        }
    }
    look
}

/// Remaining work of one block, counted in representative passes: one unit
/// is one distinct texel evaluated by one selector pass. Every optional step
/// is charged its worst case before it runs and is skipped when the
/// remaining work cannot pay for it, so the search of any block performs at
/// most the fixed seed and ranking work plus the preset budget, whatever the
/// content. Early exits inside a paid step only leave work unspent.
struct Work(u32);

impl Work {
    fn take(&mut self, cost: u32) -> bool {
        if cost <= self.0 {
            self.0 -= cost;
            true
        } else {
            false
        }
    }

    // Return the unused part of a charge once the step reports the passes it
    // performed; the bound still holds because a step never exceeds its charge.
    fn refund(&mut self, charged: u32, used: u32) {
        self.0 += charged - used.min(charged);
    }
}

// Worst-case charges, in representative passes over `n` representatives.
// A selector pass costs n, a least-squares solve n, a polish pass (five
// endpoint sums per channel, then one selector pass) 3n and a principal
// axis (covariance, power iterations and projections) 2n.
const fn axis_work(n: u32) -> u32 {
    2 * n
}

const fn improve_work(n: u32, iters: usize, rounds: usize) -> u32 {
    n * (2 * iters as u32 + 3 * rounds as u32)
}

const fn rough_work(n: u32, iters: usize) -> u32 {
    n * (1 + 2 * iters as u32)
}

const fn parity_work(n: u32, prec: Prec, cfg: &Search) -> u32 {
    let others = match prec.pbit {
        PBit::None => return 0,
        PBit::Unique => 3,
        PBit::Shared => 1,
    };
    let tries = if cfg.pbit_tries < others {
        cfg.pbit_tries as u32
    } else {
        others as u32
    };
    if tries == 0 {
        return 0;
    }
    n * (1 + tries * (1 + 2 * cfg.refine_iters as u32))
}

// Parity and improvement of one subset, as in `refine_subset`.
fn refine_work(mode: usize, n: u32, cfg: &Search, opaque: bool) -> u32 {
    let parity = match mode {
        0 => parity_work(n, P0, cfg),
        1 => parity_work(n, P1, cfg),
        3 => parity_work(n, P3, cfg),
        7 if !opaque => parity_work(n, P7, cfg),
        _ => 0,
    };
    parity + improve_work(n, cfg.refine_iters, cfg.refine_rounds)
}

// The representatives of each subset of a partition.
#[inline]
fn subset_len(reps: &Reps, masks: &[u16]) -> u32 {
    let mut len = 0;
    for &m in masks {
        len += reps.masked_len(m) as u32;
    }
    len
}

// Representatives of every subset of an opaque candidate, per channel group.
fn candidate_len(reps: &Reps, candidate: &Cand) -> u32 {
    match candidate {
        Cand::Two { partition, .. } => {
            let m = MASKS2[*partition][0];
            subset_len(reps, &[!m, m])
        }
        Cand::Three { partition, .. } => subset_len(reps, &masks3(*partition)),
        _ => reps.len as u32,
    }
}

// Endpoint-lattice moves of `precision`: each colour channel (and the scalar
// channel of a dual-stream candidate) tries four or eight moves.
fn precision_work(reps: &Reps, candidate: &Cand, diagonals: bool) -> u32 {
    let moves = if diagonals { 4 } else { 8 };
    let channels = if matches!(candidate, Cand::Dual { .. }) {
        4
    } else {
        3
    };
    moves * channels * candidate_len(reps, candidate)
}

// A second principal-axis seed per subset (two axes are compared), its
// rough fit and refinement, as in `alternate_seed`.
fn alternate_work(reps: &Reps, candidate: &Cand, cfg: &Search) -> u32 {
    let n = candidate_len(reps, candidate);
    match candidate {
        Cand::Six(_) => {
            axis_work(n)
                + rough_work(n, cfg.refine_iters)
                + improve_work(n, cfg.refine_iters, cfg.refine_rounds)
        }
        Cand::Two { mode, .. } => {
            2 * axis_work(n) + rough_work(n, cfg.rough_iters) + refine_work(*mode, n, cfg, true)
        }
        Cand::Three { .. } => {
            2 * axis_work(n) + rough_work(n, cfg.rough_iters) + refine_work(0, n, cfg, true)
        }
        Cand::Dual { .. } => 0,
    }
}

impl Search {
    /// These ceilings with colour under zero alpha scored (`false`, straight
    /// alpha) or ignored (`true`).
    pub(crate) const fn hiding_transparent(self, hide: bool) -> Self {
        Self {
            hide_transparent: hide,
            ..self
        }
    }

    fn for_opaque_block(self, units: usize) -> Self {
        let mut local = self.for_block(units);
        // A small seed residual can conceal a missed partition on fine detail.
        // Spend at most four rough fits per two-subset mode in that case:
        // mode 1 recall decides low-residual blocks where mode 6 is close.
        if units == 1 {
            local.parts[0] = if local.quality { 4 } else { 2 };
            local.parts[1] = 4;
        }
        local
    }

    fn for_block(self, units: usize) -> Self {
        debug_assert!((1..=4).contains(&units));
        let scale = 1 + usize::from(self.quality);
        Self {
            // Low-residual fits spend their small allowance on endpoint
            // precision; larger residuals spend it on additional partitions.
            axis_rounds: if units == 1 { self.axis_rounds } else { 0 },
            // Sparse opaque Quality blocks can restore the full mode-5
            // allowance after this ordinary rotation ranking limit.
            duals: [self.duals[0].min(2), (scale + units / 2).min(self.duals[1])],
            rough_iters: self.rough_iters,
            parts: if self.quality {
                [scale * units, scale * units, 2 * scale * units]
            } else {
                let per = BALANCED_PARTS_PER_UNIT;
                [per * units, per * units, 2 * per * units]
            },
            parts3: if self.quality { units } else { self.parts3 },
            refine_iters: (scale + 1 + units / 2).min(self.refine_iters),
            ..self
        }
    }
}

#[derive(Clone, Copy)]
enum Cand {
    Six(Fit),
    Dual {
        mode: usize,
        rotation: usize,
        sel: usize,
        color: Fit,
        alpha: Fit,
    },
    Two {
        mode: usize,
        partition: usize,
        s: [Fit; 2],
    },
    /// Three-subset mode 0.
    Three {
        partition: usize,
        s: [Fit; 3],
    },
}

impl Cand {
    fn pack(self) -> [u8; 16] {
        match self {
            Cand::Six(f) => pack6(f),
            Cand::Dual {
                mode,
                rotation,
                sel,
                color,
                alpha,
            } => pack_dual(mode, rotation, sel, color, alpha),
            Cand::Two { mode, partition, s } => pack2(mode, partition, s),
            Cand::Three { partition, s } => pack3(partition, s),
        }
    }

    /// The decoded texels, from the dequantized endpoints and selectors that
    /// [`Cand::pack`] stores (anchor flips leave them unchanged).
    fn decode(self) -> Block {
        let lerp = |f: &Fit, lv: &Levels, i: usize, c: usize| {
            let w = lv.w[usize::from(f.idx[i])];
            (((64 - w) * f.d[0][c] + w * f.d[1][c] + 32) >> 6) as u8
        };
        core::array::from_fn(|i| match self {
            Cand::Six(f) => core::array::from_fn(|c| lerp(&f, &L4, i, c)),
            Cand::Dual {
                mode,
                rotation,
                sel,
                color,
                alpha,
            } => {
                let (_, cl, _, al) = dual_layout(mode, sel);
                let r: [u8; 4] = core::array::from_fn(|c| {
                    if c == 3 {
                        lerp(&alpha, al, i, 3)
                    } else {
                        lerp(&color, cl, i, c)
                    }
                });
                let map = dual_map(rotation, false);
                let mut p = [0; 4];
                for c in 0..4 {
                    p[map[c]] = r[c];
                }
                p
            }
            Cand::Two { mode, partition, s } => {
                let f = &s[usize::from(MASKS2[partition][0] >> i & 1 == 1)];
                let lv = if mode == 1 { &L3 } else { &L2 };
                core::array::from_fn(|c| {
                    if c == 3 && mode != 7 {
                        255
                    } else {
                        lerp(f, lv, i, c)
                    }
                })
            }
            Cand::Three { partition, s } => {
                let k = masks3(partition)
                    .iter()
                    .position(|m| m >> i & 1 == 1)
                    .unwrap_or(0);
                core::array::from_fn(|c| if c == 3 { 255 } else { lerp(&s[k], &L3, i, c) })
            }
        })
    }

    /// Largest absolute byte error of any channel of any texel.
    fn peak(self, px: &Block) -> u8 {
        let [colour, alpha] = self.peaks(px);
        colour.max(alpha)
    }

    /// Largest absolute byte error of the colour channels and of alpha.
    fn peaks(self, px: &Block) -> [u8; 2] {
        let d = self.decode();
        let mut peaks = [0; 2];
        for (a, b) in px.iter().zip(d.iter()) {
            for c in 0..4 {
                let g = usize::from(c == 3);
                peaks[g] = peaks[g].max(a[c].abs_diff(b[c]));
            }
        }
        peaks
    }
}

// (error, mode, partition, subsets) of a two-subset candidate.
type RankedTwo = (u32, usize, usize, [Fit; 2]);
// (error, partition, subsets) of a three-subset mode 0 candidate.
type RankedThree = (u32, usize, [Fit; 3]);

fn retain_partition(kept: &mut [Option<RankedTwo>; 2], candidate: RankedTwo) {
    if kept[0].is_none_or(|k| candidate.0 < k.0) {
        kept[1] = kept[0];
        kept[0] = Some(candidate);
    } else if kept[1].is_none_or(|k| candidate.0 < k.0) {
        kept[1] = Some(candidate);
    }
}

type RankedDual = (u32, usize, usize, usize, Fit, Fit);

// A rough fit the opaque search may spend breadth on.
#[derive(Clone, Copy)]
enum Plan {
    Dual {
        mode: usize,
        rotation: usize,
        slot: usize,
    },
    Two {
        mode: usize,
        partition: usize,
    },
    Three {
        partition: usize,
    },
}

// Planned fits ordered by estimate, lowest first; ties keep insertion order.
// The preset ceilings propose at most 6 dual, 16 two-subset and 8
// three-subset fits.
struct Ranking<const N: usize> {
    items: [(f32, Plan); N],
    len: usize,
}

impl<const N: usize> Default for Ranking<N> {
    fn default() -> Self {
        Self {
            items: [(
                f32::MAX,
                Plan::Two {
                    mode: 0,
                    partition: 0,
                },
            ); N],
            len: 0,
        }
    }
}

impl<const N: usize> Ranking<N> {
    #[inline]
    fn insert(&mut self, estimate: f32, plan: Plan) {
        debug_assert!(self.len < self.items.len());
        if self.len == self.items.len() {
            return;
        }
        let mut at = self.len;
        while at > 0 && estimate < self.items[at - 1].0 {
            self.items[at] = self.items[at - 1];
            at -= 1;
        }
        self.items[at] = (estimate, plan);
        self.len += 1;
    }

    fn best(&self, count: usize) -> &[(f32, Plan)] {
        &self.items[..count.min(self.len)]
    }
}

// A rough candidate awaiting refinement, with its rough decoded error.
#[derive(Clone, Copy)]
enum Rough {
    Dual(RankedDual),
    Two(RankedTwo),
    Three(RankedThree),
}

impl Rough {
    fn err(&self) -> u32 {
        match self {
            Rough::Dual(r) => r.0,
            Rough::Two(r) => r.0,
            Rough::Three(r) => r.0,
        }
    }
}

// Estimated error of mode 5 ([r][0]) and mode 4 with each index selection
// ([r][1], [r][2]) after rotation r: the colour channels' off-axis residual
// plus a selector-step term for the principal variance, and a selector-step
// term for the scalar channel's variance (SSE units, as for partitions).
struct DualEstimates {
    error: [[f32; 3]; 4],
    mean: [f32; 4],
    cov: [[f32; 4]; 4],
}

fn dual_estimates(px: &Block, reps: &Reps) -> DualEstimates {
    let mut n = 0i32;
    let mut s = [0i32; 4];
    let mut prod = [[0i32; 4]; 4];
    for (i, count) in reps.iter() {
        n += count;
        for c in 0..4 {
            let x = i32::from(px[i][c]);
            s[c] += x * count;
            for e in c..4 {
                prod[c][e] += x * i32::from(px[i][e]) * count;
            }
        }
    }
    let nf = n as f32;
    let mut cov = [[0f32; 4]; 4];
    let mut raw_cov = cov;
    for c in 0..4 {
        for e in c..4 {
            let raw = (n * prod[c][e] - s[c] * s[e]) as f32;
            raw_cov[c][e] = raw;
            raw_cov[e][c] = raw;
            let v = raw / nf;
            cov[c][e] = v;
            cov[e][c] = v;
        }
    }
    let step = |levels: f32| 1.0 / ((levels - 1.0) * (levels - 1.0));
    let mut out = [[0f32; 3]; 4];
    for (r, slot) in out.iter_mut().enumerate() {
        let scalar = if r == 0 { 3 } else { r - 1 };
        let mut colour = [0usize, 1, 2];
        if r > 0 {
            colour[r - 1] = 3;
        }
        let mut trace = 0f32;
        let mut top = colour[0];
        for &c in &colour {
            trace += cov[c][c];
            if cov[c][c] > cov[top][top] {
                top = c;
            }
        }
        let lambda = if trace <= 0.0 {
            0.0
        } else {
            let mut ww = 0f32;
            let mut wcw = 0f32;
            for &c in &colour {
                let mut cw = 0f32;
                for &e in &colour {
                    cw += cov[c][e] * cov[e][top];
                }
                ww += cov[c][top] * cov[c][top];
                wcw += cov[c][top] * cw;
            }
            (wcw / ww).min(trace)
        };
        let residual = trace - lambda;
        let var = cov[scalar][scalar];
        *slot = [
            residual + (lambda + var) * step(4.0),
            residual + lambda * step(4.0) + var * step(8.0),
            residual + lambda * step(8.0) + var * step(4.0),
        ];
    }
    DualEstimates {
        error: out,
        mean: s.map(|v| v as f32 / nf),
        cov: raw_cov,
    }
}

// The stored low bit of each endpoint for `prec`.
pub(super) fn parity_of<const C0: usize>(d: &Pair, prec: Prec) -> [usize; 2] {
    let shift = 8 - prec.bits;
    [0, 1].map(|k| ((d[k][C0] >> shift) & 1) as usize)
}

// The other p-bit assignments of `prec`, ranked by how closely they quantize
// the least-squares targets `t` (ties keep the fixed order).
pub(super) fn ranked_parities<const C0: usize, const C1: usize>(
    t: &Pair,
    prec: Prec,
    current: [usize; 2],
) -> ([[usize; 2]; 3], usize) {
    let combos: &[[usize; 2]] = match prec.pbit {
        PBit::None => &[],
        PBit::Unique => &[[0, 0], [0, 1], [1, 0], [1, 1]],
        PBit::Shared => &[[0, 0], [1, 1]],
    };
    let mut ranked = [(0i32, [0usize; 2]); 3];
    let mut len = 0;
    for &p in combos {
        if p == current {
            continue;
        }
        let d = quantize_with::<C0, C1>(t, prec, p);
        let mut e = 0;
        for k in 0..2 {
            for c in C0..C1 {
                e += (t[k][c] - d[k][c]).pow(2);
            }
        }
        let mut at = len;
        while at > 0 && e < ranked[at - 1].0 {
            ranked[at] = ranked[at - 1];
            at -= 1;
        }
        ranked[at] = (e, p);
        len += 1;
    }
    (ranked.map(|r| r.1), len)
}

// Fit again under the `tries` other p-bit assignments that best quantize the
// least-squares targets of the current selectors; keep the best.
fn parity<const C0: usize, const C1: usize>(
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
    let (ranked, len) = ranked_parities::<C0, C1>(&t, prec, current);
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

// Colour and scalar precision and selector levels of mode 5, or of mode 4
// with index selection `sel`.
pub(super) fn dual_layout(
    mode: usize,
    sel: usize,
) -> (Prec, &'static Levels, Prec, &'static Levels) {
    match (mode, sel) {
        (5, _) => (P5C, &L2, P5A, &L2),
        (_, 0) => (P4C, &L2, P4A, &L3),
        _ => (P4C, &L3, P4A, &L2),
    }
}

// Compact a constant zero or 255 channel introduced by rotation into slot 2.
// Both dual-stream modes represent these values exactly.
fn dual_map(rotation: usize, compact: bool) -> [usize; 4] {
    let mut map = [0, 1, 2, 3];
    if rotation > 0 {
        map.swap(rotation - 1, 3);
    }
    if compact {
        // Preserve variable-channel order, including equal-variance axis ties.
        for c in rotation - 1..2 {
            map.swap(c, c + 1);
        }
    }
    map
}

fn restore_color(mut color: Fit, rotation: usize, compact: bool, value: u8) -> Fit {
    if compact {
        for endpoint in &mut color.d {
            endpoint[2] = i32::from(value);
            for c in (rotation - 1..2).rev() {
                endpoint.swap(c, c + 1);
            }
        }
    }
    color
}

fn scalar_targets(px: &Block, reps: &Reps) -> Pair {
    let mut t = [[0; 4]; 2];
    t[0][3] = 255;
    for (i, _) in reps.iter() {
        let x = i32::from(px[i][3]);
        t[0][3] = t[0][3].min(x);
        t[1][3] = t[1][3].max(x);
    }
    t
}

fn dual_fit(
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
fn refine_dual(
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
fn rough_with<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    t: &Pair,
    prec: Prec,
    lv: &Levels,
    iters: usize,
    force_p1: bool,
    limit: u32,
) -> Fit {
    let d = quantize::<C0, C1>(t, prec, force_p1);
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

fn rough(
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
// Returns the passes performed (see `improve`).
fn refine_subset(
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

// Search the endpoint lattice with selectors free to change. Simultaneous
// endpoint moves can escape a fixed-selector polish minimum. The source
// block's allocation bounds this extra precision work to low residuals.
fn lattice<const C0: usize, const C1: usize>(
    px: &Block,
    reps: &Reps,
    f: &mut Fit,
    prec: Prec,
    lv: &Levels,
    diagonals: bool,
) {
    if f.err == 0 {
        return;
    }
    let step = if prec.pbit == PBit::None { 1 } else { 2 };
    let n = prec.bits as u32;
    let max = (1 << n) - 1;
    for c in C0..C1 {
        let centre = f.d;
        let a = centre[0][c] >> (8 - n);
        let b = centre[1][c] >> (8 - n);
        let moves: &[(i32, i32)] = if diagonals {
            &[(-1, -1), (-1, 1), (1, -1), (1, 1)]
        } else {
            &[
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        };
        for &(da, db) in moves {
            let (x, y) = (a + da * step, b + db * step);
            if !(0..=max).contains(&x) || !(0..=max).contains(&y) {
                continue;
            }
            let mut d = centre;
            d[0][c] = expand(x as u32, n) as i32;
            d[1][c] = expand(y as u32, n) as i32;
            let mut idx = f.idx;
            let err = select::<C0, C1>(px, reps, &d, lv, &mut idx, f.err);
            if err < f.err {
                *f = Fit { err, d, idx };
            }
        }
    }
}

fn precision(px: &Block, reps: &Reps, candidate: &mut Cand, diagonals: bool) -> u32 {
    match candidate {
        Cand::Six(f) => {
            lattice::<0, 3>(px, reps, f, P6, &L4, diagonals);
            f.err
        }
        Cand::Two { mode, partition, s } => {
            let mask = MASKS2[*partition][0];
            let (prec, lv) = if *mode == 1 { (P1, &L3) } else { (P3, &L2) };
            for (f, mask) in s.iter_mut().zip([!mask, mask]) {
                lattice::<0, 3>(px, &reps.masked(mask), f, prec, lv, diagonals);
            }
            s.iter().map(|f| f.err).sum()
        }
        Cand::Three { partition, s } => {
            for (f, mask) in s.iter_mut().zip(masks3(*partition)) {
                lattice::<0, 3>(px, &reps.masked(mask), f, P0, &L3, diagonals);
            }
            s.iter().map(|f| f.err).sum()
        }
        Cand::Dual {
            mode,
            rotation,
            sel,
            color,
            alpha,
        } => {
            let rotated = px.map(|p| dual_map(*rotation, false).map(|c| p[c]));
            let (cp, cl, ap, al) = dual_layout(*mode, *sel);
            lattice::<0, 3>(&rotated, reps, color, cp, cl, diagonals);
            lattice::<3, 4>(&rotated, reps, alpha, ap, al, diagonals);
            color.err + alpha.err
        }
    }
}

fn recall_axis_targets<const C0: usize, const C1: usize>(px: &Block, reps: &Reps) -> Pair {
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
    recall_axis_endpoints::<C0, C1>(px, reps, &mean, &cov)
}

// Project onto an axis from previously computed moments. Covariance uses
// the unnormalized integer moment scale, matching fresh subset fits exactly.
fn recall_axis_endpoints<const C0: usize, const C1: usize>(
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
    for _ in 0..8 {
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

// Refit a retained partition from a more converged PCA seed, preserving
// the existing fit whenever the alternative does not lower decoded error.
fn alternate_seed(px: &Block, reps: &Reps, candidate: &mut Cand, cfg: &Search) -> u32 {
    match candidate {
        Cand::Six(f) => {
            let t = recall_axis_targets::<0, 3>(px, reps);
            let mut g = rough_with::<0, 3>(px, reps, &t, P6, &L4, cfg.refine_iters, true, u32::MAX);
            improve::<0, 3>(
                px,
                reps,
                &mut g,
                P6,
                &L4,
                cfg.refine_iters,
                cfg.refine_rounds,
                true,
            );
            g.d[0][3] = 255;
            g.d[1][3] = 255;
            if g.err < f.err {
                *f = g;
            }
            f.err
        }
        Cand::Two { mode, partition, s } => {
            let mask = MASKS2[*partition][0];
            for (f, mask) in s.iter_mut().zip([!mask, mask]) {
                let r = reps.masked(mask);
                let t = recall_axis_targets::<0, 3>(px, &r);
                // Identical targets reproduce the already refined fit; its
                // subsequent lattice pass can only have lowered the error.
                if t == axis_targets::<0, 3>(px, &r) {
                    continue;
                }
                let mut g = rough(px, &r, &t, *mode, true, cfg.rough_iters, u32::MAX);
                refine_subset(px, *mode, &r, &mut g, true, cfg);
                if g.err < f.err {
                    *f = g;
                }
            }
            s.iter().map(|f| f.err).sum()
        }
        Cand::Three { partition, s } => {
            for (f, mask) in s.iter_mut().zip(masks3(*partition)) {
                let r = reps.masked(mask);
                let t = recall_axis_targets::<0, 3>(px, &r);
                // Identical targets reproduce the already refined fit; its
                // subsequent lattice pass can only have lowered the error.
                if t == axis_targets::<0, 3>(px, &r) {
                    continue;
                }
                let mut g =
                    rough_with::<0, 3>(px, &r, &t, P0, &L3, cfg.rough_iters, false, u32::MAX);
                refine_subset(px, 0, &r, &mut g, true, cfg);
                if g.err < f.err {
                    *f = g;
                }
            }
            s.iter().map(|f| f.err).sum()
        }
        Cand::Dual { color, alpha, .. } => color.err + alpha.err,
    }
}

/// Encode one block with the given search breadth.
pub(crate) fn encode_with(px: &Block, cfg: &Search) -> [u8; 16] {
    match solid_rgba(px) {
        Some(block) => block,
        None => search(px, cfg).1.pack(),
    }
}

pub(super) fn preserving_zero_alpha(px: &Block, cfg: &Search) -> Option<[u8; 16]> {
    let candidate = search(px, cfg).1;
    px.iter()
        .zip(candidate.decode())
        .all(|(source, decoded)| source[3] != 0 || decoded[3] == 0)
        .then(|| candidate.pack())
}

// The best candidate found for a block and its exact decoded error.
fn search(px: &Block, cfg: &Search) -> (u32, Cand) {
    search_seeded(px, cfg, None)
}

fn search_seeded(px: &Block, cfg: &Search, seed: Option<Fit>) -> (u32, Cand) {
    if px.iter().any(|p| p[3] != 255) {
        return visible_search(px, cfg);
    }
    // The early return above routes every nonopaque block separately.
    let opaque = true;
    let reps = Reps::distinct(px);
    // The preset's ceiling, fixed before any allowance is chosen.
    let mut work = Work(cfg.budget);
    let n = reps.len as u32;
    let mut m6 = if opaque {
        seed.unwrap_or_else(|| fit::<0, 3>(px, &reps, P6, &L4, 1, 0, true))
    } else {
        fit::<0, 4>(px, &reps, P6, &L4, 1, 0, false)
    };
    if opaque {
        m6.d[0][3] = 255;
        m6.d[1][3] = 255;
    }
    // Freeze the allowance from the seed, before subsequent improvements.
    // No candidate may expand it because the remaining error is large.
    let units = crate::budget::difficulty(px, m6.err, if opaque { 3 } else { 4 }, None);
    if units == 0 {
        return (0, Cand::Six(m6));
    }
    // Precision is most useful when the block has few colours or its
    // colour variation lies in a smaller channel space. These tests use
    // only source structure, independently of the current winning mode.
    let constant_channel = (0..3).any(|c| px.iter().all(|p| p[c] == px[0][c]));
    let grey = px.iter().all(|p| p[0] == p[1] && p[1] == p[2]);
    let detail = constant_channel || grey;
    let low = core::array::from_fn::<_, 3, _>(|c| px.iter().map(|p| p[c]).min().unwrap());
    let high = core::array::from_fn::<_, 3, _>(|c| px.iter().map(|p| p[c]).max().unwrap());
    let saturated = high.iter().any(|&v| v <= 16) && (0..3).any(|c| high[c] - low[c] >= 32);
    // Sparse ramps and gradients with a near-zero channel need endpoint
    // precision even when their seed residual earns only one search unit.
    let extra_precision = !cfg.quality && (reps.len <= 4 || saturated);
    // Escalated Balanced blocks use Quality's allowance with Balanced axis
    // rounds.
    let deep = Search {
        axis_rounds: cfg.for_opaque_block(units).axis_rounds,
        ..QUALITY
            .hiding_transparent(cfg.hide_transparent)
            .for_opaque_block(units)
    };
    let cfg = &if cfg.quality && !cfg.hide_transparent && reps.len <= 2 {
        // Collinear colours can tie every mode-5 rotation's variance estimate.
        // Endpoint quantization breaks that tie: only the scalar stream has
        // eight-bit endpoints. Fit all rotations before rejecting an exact fit.
        let mut local = cfg.for_opaque_block(units);
        local.duals[0] = cfg.duals[0];
        local
    } else if !cfg.quality && reps.len <= 4 {
        deep
    } else if !cfg.quality && (grey || saturated) {
        // Quality's refinement depth and alternate seed; breadth stays
        // Balanced unless the preset escalates it too.
        if ESCALATE_BREADTH {
            deep
        } else {
            let local = cfg.for_opaque_block(units);
            Search {
                duals: local.duals,
                parts: local.parts,
                parts3: local.parts3,
                ..deep
            }
        }
    } else {
        cfg.for_opaque_block(units)
    };
    // Endpoint precision of low-residual and sparse or saturated blocks runs
    // last; hold its work back so that breadth cannot spend it first.
    // Only the first lattice pass is held back; a second one runs if work
    // remains.
    let diagonals = !cfg.quality && !detail;
    let mut reserve = if units == 1 && diagonals {
        12 * n
    } else if units == 1 || extra_precision {
        24 * n
    } else {
        0
    };
    if !work.take(reserve) {
        reserve = 0;
    }
    if opaque {
        let charge = improve_work(n, cfg.refine_iters, cfg.refine_rounds);
        if work.take(charge) {
            let passes = improve::<0, 3>(
                px,
                &reps,
                &mut m6,
                P6,
                &L4,
                cfg.refine_iters,
                cfg.refine_rounds,
                true,
            );
            work.refund(charge, passes * n);
        }
        if work.take(8 * n * cfg.axis_rounds as u32) {
            axis_search::<0, 3>(px, &reps, &mut m6, &L4, 2, cfg.axis_rounds);
        }
    } else {
        parity::<0, 4>(
            px,
            &reps,
            &mut m6,
            P6,
            &L4,
            cfg.refine_iters,
            cfg.pbit_tries,
        );
        improve::<0, 4>(
            px,
            &reps,
            &mut m6,
            P6,
            &L4,
            cfg.refine_iters,
            cfg.refine_rounds,
            false,
        );
    }
    if opaque {
        m6.d[0][3] = 255;
        m6.d[1][3] = 255;
    }
    let mut best_err = m6.err;
    let mut best = Cand::Six(m6);
    let mut finalists: Finalists = [None; 8];
    finalists[0] = Some((best_err, best));
    if best_err == 0 {
        return (0, best);
    }
    // Rough fits of every mode family compete for one refinement pool.
    // Breadth may spend only the work above the refinement share; whatever
    // it leaves joins that share.
    let share = work.0 / 100 * REFINE_SHARE;
    let mut broad = Work(work.0 - share);
    work.0 = share;
    let mut pool: [Option<Rough>; 7] = [None; 7];
    // Every family proposes its candidates with its cheap estimate, all in
    // block SSE units: modes 5 and 4 by rotation and index selection, the
    // two-subset partitions of modes 3 and 1 and the three-subset partitions
    // of mode 0. One ranking orders them; rough fits run best first
    // while breadth lasts. Rotation 0 of an opaque block keeps alpha in the
    // scalar slot, where mode 5 is exact, so mode 4 skips it.
    let alpha_varies = false;
    let flat_extreme_alpha = opaque || (!alpha_varies && px[0][3] == 0);
    let est = dual_estimates(px, &reps);
    let partitions = Partitions::new::<3>(px);
    let mut plan = Ranking::<ROUGH_SLOTS>::default();
    for (mode, count) in [(5usize, cfg.duals[0]), (4, cfg.duals[1])] {
        // Four rotations, or three rotations with two index selections.
        let mut options = Ranking::<6>::default();
        for rotation in 0..4 {
            for slot in if mode == 5 { 0..1 } else { 1..3 } {
                if mode == 4 && rotation == 0 && !alpha_varies {
                    continue;
                }
                options.insert(
                    est.error[rotation][slot],
                    Plan::Dual {
                        mode,
                        rotation,
                        slot,
                    },
                );
            }
        }
        for &(e, p) in options.best(count) {
            plan.insert(e, p);
        }
    }
    let counts = [cfg.parts[0], cfg.parts[1]];
    let ranked = best_partitions_31(&partitions.est, counts);
    for ((mode, count), best) in [3usize, 1].into_iter().zip(counts).zip(&ranked) {
        for &(e, partition) in &best[..count] {
            plan.insert(e, Plan::Two { mode, partition });
        }
    }
    // Mode 0 ranks its 16 partitions; a ranked three-subset estimate costs
    // about eight passes (three subset moment sums and estimates).
    let (count, ranked) = (cfg.parts3, 16);
    if count > 0 && broad.take(8 * ranked as u32) {
        let est3 = partition_estimates3(&partitions.rows, ranked);
        for &(e, partition) in &best_partitions(&est3, 8.0, count)[..count] {
            plan.insert(e, Plan::Three { partition });
        }
    }
    // Modes 4 and 5 share principal-axis targets for each rotation; modes 3
    // and 1 share them for each partition.
    let mut targets = [None; 4];
    let mut shared: [Option<(usize, [Pair; 2])>; 4] = [None; 4];
    let mut kept_dual: [Option<RankedDual>; 2] = [None; 2];
    let mut kept: [[Option<RankedTwo>; 2]; 2] = [[None; 2]; 2];
    let mut kept3: Option<RankedThree> = None;
    // Every rough fit, in plan order, for the peak second look.
    let mut roughs: [Option<Rough>; ROUGH_SLOTS] = [None; ROUGH_SLOTS];
    for (slot, &(_, step)) in plan.best(plan.len).iter().enumerate() {
        if let Plan::Two { mode, partition } = step {
            let m1 = MASKS2[partition][0];
            let sub = [reps.masked(!m1), reps.masked(m1)];
            let k = (sub[0].len + sub[1].len) as u32;
            let known = shared
                .iter()
                .flatten()
                .find(|s| s.0 == partition)
                .map(|s| s.1);
            let axis = if known.is_none() { axis_work(k) } else { 0 };
            if !broad.take(axis + rough_work(k, cfg.rough_iters)) {
                continue;
            }
            let t = known.unwrap_or_else(|| {
                let t = partitions.targets::<3>(px, &sub, partition);
                shared.rotate_right(1);
                shared[0] = Some((partition, t));
                t
            });
            let s = [
                rough(px, &sub[0], &t[0], mode, true, cfg.rough_iters, u32::MAX),
                rough(px, &sub[1], &t[1], mode, true, cfg.rough_iters, u32::MAX),
            ];
            let err = s[0].err + s[1].err;
            if err == 0 {
                return (0, Cand::Two { mode, partition, s });
            }
            roughs[slot] = Some(Rough::Two((err, mode, partition, s)));
            retain_partition(&mut kept[usize::from(mode == 1)], (err, mode, partition, s));
            continue;
        }
        if let Plan::Three { partition } = step {
            let k = subset_len(&reps, &masks3(partition));
            if !broad.take(axis_work(k) + rough_work(k, cfg.rough_iters)) {
                continue;
            }
            let s = fit_partition3(px, &reps, partition, cfg.rough_iters, 0);
            let err = s[0].err + s[1].err + s[2].err;
            if err == 0 {
                return (0, Cand::Three { partition, s });
            }
            if kept3.is_none_or(|k| err < k.0) {
                kept3 = Some((err, partition, s));
            }
            roughs[slot] = Some(Rough::Three((err, partition, s)));
            continue;
        }
        let Plan::Dual {
            mode,
            rotation,
            slot,
        } = step
        else {
            continue;
        };
        {
            let iters = if mode == 4 {
                cfg.mode4_iters
            } else {
                cfg.rough_iters
            };
            let axis = if targets[rotation].is_none() {
                axis_work(n)
            } else {
                0
            };
            if !broad.take(axis + 2 * rough_work(n, iters)) {
                continue;
            }
            let sel = slot.saturating_sub(1);
            let compact = flat_extreme_alpha && rotation > 0;
            let map = dual_map(rotation, compact);
            let rotated = rotate(px, map);
            let t = targets[rotation].get_or_insert_with(|| {
                let mut mean = [0f32; 4];
                let mut cov = [[0f32; 4]; 4];
                for c in 0..4 {
                    mean[c] = est.mean[map[c]];
                    for e in 0..4 {
                        cov[c][e] = est.cov[map[c]][map[e]];
                    }
                }
                [
                    if compact {
                        axis_endpoints::<0, 2>(&rotated, &reps, &mean, &cov)
                    } else {
                        axis_endpoints::<0, 3>(&rotated, &reps, &mean, &cov)
                    },
                    scalar_targets(&rotated, &reps),
                ]
            });
            let (color, alpha) = dual_fit(&rotated, &reps, t, mode, sel, iters, compact);
            let err = color.err + alpha.err;
            if err == 0 {
                return (
                    0,
                    Cand::Dual {
                        mode,
                        rotation,
                        sel,
                        color: restore_color(color, rotation, compact, px[0][3]),
                        alpha,
                    },
                );
            }
            let kept = &mut kept_dual[usize::from(mode == 4)];
            if kept.is_none_or(|k| err < k.0) {
                *kept = Some((err, mode, rotation, sel, color, alpha));
            }
            roughs[slot] = Some(Rough::Dual((err, mode, rotation, sel, color, alpha)));
        }
    }
    for (slot, kept) in kept_dual.into_iter().enumerate() {
        pool[slot] = kept.map(Rough::Dual);
    }
    // Each two-subset mode keeps its winner and a near-tied runner-up:
    // initial endpoint error can exceed a cutoff even when least squares
    // would recover a better candidate.
    for list in &mut kept {
        if !cfg.runner_up
            || list[0]
                .zip(list[1])
                .is_some_and(|(first, second)| second.0 > first.0.saturating_add(first.0 / 4))
        {
            list[1] = None;
        }
    }
    let [[a, b], [c, d]] = kept;
    for (slot, kept) in [a, b, c, d].into_iter().enumerate() {
        pool[2 + slot] = kept.map(Rough::Two);
    }
    pool[6] = kept3.map(Rough::Three);
    // Refine the rough candidates, lowest error first, while work remains.
    // A rough fit more than a quarter above the best refined error rarely
    // overtakes it. Unrefined candidates still compete with their rough fit.
    work.0 += broad.0;
    let mut order = [0usize; 7];
    let mut len = 0;
    let err = |slot: usize| pool[slot].map_or(u32::MAX, |r| r.err());
    for slot in 0..pool.len() {
        if pool[slot].is_none() {
            continue;
        }
        let mut at = len;
        while at > 0 && err(slot) < err(order[at - 1]) {
            order[at] = order[at - 1];
            at -= 1;
        }
        order[at] = slot;
        len += 1;
    }
    let mut refined: [Option<Rough>; 7] = [None; 7];
    // In a low-error block, endpoint quantization dominates a rough fit's
    // error and refinement (other p-bits, polish) removes most of it, so up
    // to LOW_REFINE_EXTRA candidates beyond the quarter are refined while
    // within LOW_REFINE_SLACK. The opt-in policy keeps the earlier rule.
    let (slack, mut extra) = if cfg.hide_transparent {
        (0, 0)
    } else {
        (LOW_REFINE_SLACK, LOW_REFINE_EXTRA)
    };
    for &slot in &order[..len] {
        let Some(r) = pool[slot] else {
            continue;
        };
        if r.err() > best_err.saturating_add(best_err / 4) {
            if extra == 0 || r.err() > best_err.saturating_add(slack) {
                continue;
            }
            extra -= 1;
        }
        refined[slot] = Some(r);
        let candidate = refine_rough(px, &reps, r, cfg, &mut work, flat_extreme_alpha);
        push_finalist(&mut finalists, candidate);
        if candidate.0 < best_err {
            (best_err, best) = candidate;
        }
    }
    work.0 += reserve;
    if units == 1 && best_err > 0 && work.take(precision_work(&reps, &best, diagonals)) {
        best_err = precision(px, &reps, &mut best, diagonals);
    }
    if cfg.quality && best_err > 0 && work.take(alternate_work(&reps, &best, cfg)) {
        best_err = alternate_seed(px, &reps, &mut best, cfg);
    }
    if extra_precision && best_err > 0 && work.take(precision_work(&reps, &best, false)) {
        best_err = precision(px, &reps, &mut best, false);
    }
    // The opt-in policy reproduces the earlier output on every block.
    if cfg.hide_transparent {
        return (best_err, best);
    }
    let peak = best.peak(px);
    let look = peak_look(
        px,
        n,
        (best_err, peak),
        &roughs,
        &refined,
        &mut work,
        flat_extreme_alpha,
        |r, work| refine_rough(px, &reps, r, cfg, work, flat_extreme_alpha),
    );
    peak_choice(px, (best_err, best), peak, &finalists, &look, true, false)
}

// Nonopaque blocks carry original-channel observation weights through every
// rotation and subset. Its search allowance is independent of opaque recall.
fn visible_search(source: &Block, cfg: &Search) -> (u32, Cand) {
    use objective::{
        axis_targets, dual_fit, fit, improve, parity, refine_dual, refine_subset, rough, Reps,
    };

    // Straight alpha scores every texel's colour. Only the opt-in policy
    // drops colour under zero alpha, which then decodes black.
    let mut px = *source;
    let mut visible = u16::MAX;
    if cfg.hide_transparent {
        for (i, p) in px.iter_mut().enumerate() {
            if p[3] == 0 {
                p[..3].fill(0);
                visible &= !(1 << i);
            }
        }
    }
    let px = &px;
    let mut reps = Reps::distinct(px);
    reps.objective = Some(objective::Objective {
        visible,
        alpha: 3,
        alpha_weight: 1,
        colour_weight: COLOUR_WEIGHT,
    });
    let mut m6 = fit::<0, 4>(px, &reps, P6, &L4, 1, 0, false);
    let units = objective::difficulty(px, &reps, m6.err);
    if units == 0 {
        return (0, Cand::Six(m6));
    }
    let mut work = Work(cfg.budget);
    let n = reps.len as u32;
    let lens = |masks: &[u16]| -> u32 { masks.iter().map(|&m| reps.masked(m).len as u32).sum() };
    let mut local = cfg.for_block(units);
    if !cfg.quality {
        // Cutout colour depends on the dual-stream rotation and index
        // selection; their fits are charged like any other.
        for k in 0..2 {
            local.duals[k] = local.duals[k].max(BALANCED_VISIBLE_DUALS[k]);
        }
    }
    let cfg = &local;
    let charge = parity_work(n, P6, cfg) + improve_work(n, cfg.refine_iters, cfg.refine_rounds);
    if work.take(charge) {
        let passes = parity::<0, 4>(
            px,
            &reps,
            &mut m6,
            P6,
            &L4,
            cfg.refine_iters,
            cfg.pbit_tries,
        ) + improve::<0, 4>(
            px,
            &reps,
            &mut m6,
            P6,
            &L4,
            cfg.refine_iters,
            cfg.refine_rounds,
            false,
        );
        work.refund(charge, passes * n);
    }
    let mut best = (m6.err, Cand::Six(m6));
    let mut finalists: Finalists = [None; 8];
    finalists[0] = Some(best);
    // Texels at the lowest or highest alpha are exact at the scalar
    // endpoints of modes 4 and 5 (a cutout's alpha needs no selector
    // steps); only the share of texels between them pays the step term.
    let (low, high) = px
        .iter()
        .fold((255u8, 0u8), |(l, h), p| (l.min(p[3]), h.max(p[3])));
    let between = px.iter().filter(|p| p[3] != low && p[3] != high).count();
    let alpha_steps = between as f32 / 16.0;
    let mut estimates = [[0.0; 3]; 4];
    let rotated = [0, 1, 2, 3].map(|rotation| {
        let map = dual_map(rotation, false);
        let pixels = px.map(|p| map.map(|c| p[c]));
        let mut r = reps;
        r.objective.as_mut().unwrap().alpha = map.iter().position(|&c| c == 3).unwrap() as u8;
        let (res, var) = objective::estimate::<0, 3>(&pixels, &r);
        let (_, mut scalar) = objective::estimate::<3, 4>(&pixels, &r);
        if rotation == 0 && VISIBLE_EXTREME_ALPHA {
            scalar *= alpha_steps;
        }
        estimates[rotation] = [
            res + (var + scalar) / 9.0,
            res + var / 9.0 + scalar / 49.0,
            res + var / 49.0 + scalar / 9.0,
        ];
        (pixels, r)
    });
    let mut tried = [[false; 3]; 4];
    let mut kept: [Option<RankedDual>; 2] = [None; 2];
    let mixed_alpha = px.iter().any(|p| p[3] != px[0][3]);
    // One extra alpha champion across already fitted dual candidates. It is
    // refined only if distinct from both retained mode winners.
    let mut alpha_frontier: Option<((u32, u32), RankedDual)> = None;
    for (mode, count) in [(5, cfg.duals[0]), (4, cfg.duals[1])] {
        for _ in 0..count {
            let mut pick: Option<(f32, usize, usize)> = None;
            for r in 0..4 {
                for slot in if mode == 5 { 0..1 } else { 1..3 } {
                    if tried[r][slot]
                        || (mode == 4 && r == 0 && px.iter().all(|p| p[3] == px[0][3]))
                    {
                        continue;
                    }
                    if pick.is_none_or(|p| estimates[r][slot] < p.0) {
                        pick = Some((estimates[r][slot], r, slot));
                    }
                }
            }
            let Some((_, rotation, slot)) = pick else {
                break;
            };
            let iters = if mode == 4 {
                cfg.mode4_iters
            } else {
                cfg.rough_iters
            };
            if !work.take(2 * (axis_work(n) + rough_work(n, iters))) {
                break;
            }
            tried[rotation][slot] = true;
            let sel = slot.saturating_sub(1);
            let (p, r) = &rotated[rotation];
            let targets = [axis_targets::<0, 3>(p, r), axis_targets::<3, 4>(p, r)];
            let (color, alpha) = dual_fit(p, r, &targets, mode, sel, iters, false);
            let err = color.err + alpha.err;
            if mixed_alpha {
                let ac = usize::from(r.objective.unwrap().alpha);
                let (_, cl, _, al) = dual_layout(mode, sel);
                let (f, lv) = if ac == 3 { (&alpha, al) } else { (&color, cl) };
                let mut alpha_error = 0u32;
                for (i, count) in r.iter() {
                    let w = lv.w[usize::from(f.idx[i])];
                    let d = ((64 - w) * f.d[0][ac] + w * f.d[1][ac] + 32) >> 6;
                    alpha_error += (count * (i32::from(p[i][ac]) - d).pow(2)) as u32;
                }
                let rank = (alpha_error, err);
                if alpha_frontier.is_none_or(|k| rank < k.0) {
                    alpha_frontier = Some((rank, (err, mode, rotation, sel, color, alpha)));
                }
            }
            let k = &mut kept[usize::from(mode == 4)];
            if k.is_none_or(|k| err < k.0) {
                *k = Some((err, mode, rotation, sel, color, alpha));
            }
        }
    }
    let extra = alpha_frontier.map(|(_, k)| k).filter(|candidate| {
        !kept
            .iter()
            .flatten()
            .any(|k| (k.1, k.2, k.3) == (candidate.1, candidate.2, candidate.3))
    });
    for (_, mode, rotation, sel, mut color, mut alpha) in kept.into_iter().flatten().chain(extra) {
        let charge = 2 * improve_work(n, cfg.refine_iters, cfg.refine_rounds);
        if work.take(charge) {
            let (p, r) = &rotated[rotation];
            let passes = refine_dual(p, r, mode, sel, &mut color, &mut alpha, cfg, false);
            work.refund(charge, passes * n);
        }
        let candidate = (
            color.err + alpha.err,
            Cand::Dual {
                mode,
                rotation,
                sel,
                color,
                alpha,
            },
        );
        push_finalist(&mut finalists, candidate);
        if candidate.0 < best.0 {
            best = candidate;
        }
    }
    let alpha_penalty: u32 = px.iter().map(|p| (255 - u32::from(p[3])).pow(2)).sum();
    // RGB-only modes remain eligible when their fixed decoded alpha can pay
    // its full error. No extra partitions or modes are introduced. Modes 3
    // and 1 share one RGB ranking; both rankings come from one table of
    // texel sums, fixed work like the opaque moment ranking.
    let table = objective::SumTable::new(px, reps.objective.unwrap());
    let mut rankings: Option<[[(f32, f32); 64]; 2]> = None;
    for (mode, levels, count) in [
        (7, 4.0, cfg.parts[2]),
        (3, 4.0, cfg.parts[0]),
        (1, 8.0, cfg.parts[1]),
    ] {
        let penalty = if mode == 7 { 0 } else { alpha_penalty };
        if penalty >= best.0 || count == 0 {
            continue;
        }
        let estimates = rankings.get_or_insert_with(|| table.rankings())[usize::from(mode == 7)];
        let mut kept = None;
        let mut limit = best.0.saturating_add(best.0 / 4) - penalty;
        for &(_, partition) in &best_partitions(&estimates, levels, count)[..count] {
            let mask = MASKS2[partition][0];
            let r = [reps.masked(!mask), reps.masked(mask)];
            let k = lens(&[!mask, mask]);
            if !work.take(axis_work(k) + rough_work(k, cfg.rough_iters)) {
                break;
            }
            let s = r.map(|r| {
                let t = if mode == 7 {
                    axis_targets::<0, 4>(px, &r)
                } else {
                    axis_targets::<0, 3>(px, &r)
                };
                rough(px, &r, &t, mode, false, cfg.rough_iters, u32::MAX)
            });
            let err = s[0].err + s[1].err;
            if err < limit {
                limit = err;
                kept = Some((partition, s));
            }
        }
        if let Some((partition, mut s)) = kept {
            let mask = MASKS2[partition][0];
            let charge = refine_work(mode, lens(&[!mask, mask]), cfg, false);
            if work.take(charge) {
                let mut used = 0;
                for (k, mask) in [!mask, mask].into_iter().enumerate() {
                    let r = reps.masked(mask);
                    used += r.len as u32 * refine_subset(px, mode, &r, &mut s[k], false, cfg);
                }
                work.refund(charge, used);
            }
            let candidate = (
                s[0].err + s[1].err + penalty,
                Cand::Two { mode, partition, s },
            );
            push_finalist(&mut finalists, candidate);
            if candidate.0 < best.0 {
                best = candidate;
            }
        }
    }
    let count = cfg.parts3;
    if count > 0 && alpha_penalty < best.0 {
        let estimates = table.rankings3(16, masks3);
        let mut kept: Option<RankedThree> = None;
        for &(_, partition) in &best_partitions(&estimates, 8.0, count)[..count] {
            let k = lens(&masks3(partition));
            if !work.take(axis_work(k) + rough_work(k, cfg.rough_iters)) {
                break;
            }
            let s = masks3(partition).map(|mask| {
                fit::<0, 3>(px, &reps.masked(mask), P0, &L3, cfg.rough_iters, 0, false)
            });
            let err = s.iter().map(|s| s.err).sum::<u32>() + alpha_penalty;
            if kept.is_none_or(|k| err < k.0) {
                kept = Some((err, partition, s));
            }
        }
        if let Some((_, partition, mut s)) = kept {
            let charge = refine_work(0, lens(&masks3(partition)), cfg, false);
            if work.take(charge) {
                let mut used = 0;
                for (k, mask) in masks3(partition).into_iter().enumerate() {
                    let r = reps.masked(mask);
                    used += r.len as u32 * refine_subset(px, 0, &r, &mut s[k], false, cfg);
                }
                work.refund(charge, used);
            }
            let candidate = (
                s.iter().map(|s| s.err).sum::<u32>() + alpha_penalty,
                Cand::Three { partition, s },
            );
            push_finalist(&mut finalists, candidate);
            if candidate.0 < best.0 {
                best = candidate;
            }
        }
    }
    // The opt-in policy reproduces the earlier output on every block
    // (its invisible colour would otherwise enter the peak).
    if cfg.hide_transparent {
        return best;
    }
    // Nonopaque blocks take no second look: measured on the 71-texture
    // set, it raised other channels' peaks and cost PSNR on noisy alpha.
    let peak = best.1.peak(px);
    // Quality only: measured on the development set, Balanced's fewer
    // refined finalists gain no colour peak from it and lose alpha PSNR.
    let no_look = [None; PEAK_LOOK_REFINE];
    peak_choice(px, best, peak, &finalists, &no_look, false, cfg.quality)
}

#[cfg(feature = "simd")]
pub(super) fn encode_seeded(px: &Block, cfg: &Search, seed: Option<Fit>) -> [u8; 16] {
    match solid_rgba(px) {
        Some(block) => block,
        None => search_seeded(px, cfg, seed).1.pack(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_sparse_rotations_recover_exact_endpoints() {
        // One channel needs eight-bit endpoints; putting it in the colour
        // stream incurs error even though the rotation estimates tie.
        for colours in [
            [[16, 28, 33], [16, 68, 65]],
            [[0, 4, 5], [0, 8, 10]],
            [[0, 4, 5], [0, 4, 5]],
            [[16, 0, 33], [16, 0, 33]],
        ] {
            for rotation in 0..3 {
                let px: Block = core::array::from_fn(|i| {
                    let c = colours[usize::from(i % 4 != 0)];
                    [
                        c[rotation],
                        c[(rotation + 1) % 3],
                        c[(rotation + 2) % 3],
                        255,
                    ]
                });
                let block = encode_with(&px, &QUALITY);
                let mut decoded = [0; 64];
                bcdec_rs::bc7(&block, &mut decoded, 16);
                assert_eq!(decoded.as_chunks::<4>().0, px.as_slice());
            }
        }
    }

    #[test]
    fn invisible_rgb_cannot_change_search_or_allocation() {
        for visible_count in [0, 1, 7, 16] {
            let mut px: Block = core::array::from_fn(|i| {
                [
                    (i * 13) as u8,
                    (i * 7) as u8,
                    (i * 17) as u8,
                    if i < visible_count { 1 } else { 0 },
                ]
            });
            for cfg in [BALANCED, QUALITY].map(|c| c.hiding_transparent(true)) {
                let a = search(&px, &cfg);
                for p in &mut px[visible_count..] {
                    p[..3].copy_from_slice(&[251, 92, 177]);
                }
                let b = search(&px, &cfg);
                assert_eq!(a.0, b.0);
                assert_eq!(a.1.pack(), b.1.pack());
            }
        }
    }

    #[test]
    fn second_look_candidates_use_the_wider_error_bound() {
        // One red texel of 20 on black: a black fit peaks at 20, a fit of
        // red 10 everywhere peaks at 10.
        let mut px: Block = [[0, 0, 0, 255]; 16];
        px[0][0] = 20;
        let flat = |r: i32| {
            Cand::Six(Fit {
                err: 0,
                d: [[r, 0, 0, 255]; 2],
                idx: [0; 16],
            })
        };
        let (black, red) = (flat(0), flat(10));
        assert_eq!((black.peak(&px), red.peak(&px)), (20, 10));
        // 1200 is above 1000 + 1000 / PEAK_SLACK but within 1000 / PEAK_LOOK_SLACK.
        let mut finalists: Finalists = [None; 8];
        finalists[1] = Some((1200, red));
        let none = [None; PEAK_LOOK_REFINE];
        let (err, _) = peak_choice(&px, (1000, black), 20, &finalists, &none, false, false);
        assert_eq!(err, 1000);
        // An opaque winner at PEAK_LOOK_FLOOR or above extends the look's
        // bound to its finalists.
        let (err, picked) = peak_choice(&px, (1000, black), 20, &finalists, &none, true, false);
        assert_eq!((err, picked.peak(&px)), (1200, 10));
        let mut look: Look = [None; PEAK_LOOK_REFINE];
        look[0] = Some((1200, red));
        let (err, picked) = peak_choice(&px, (1000, black), 20, &[None; 8], &look, false, false);
        assert_eq!((err, picked.peak(&px)), (1200, 10));
    }

    #[test]
    fn peak_choice_does_not_move_a_peak_from_alpha_into_colour() {
        let px: Block = [[100; 4]; 16];
        let flat = |d: [i32; 4]| {
            Cand::Six(Fit {
                err: 0,
                d: [d; 2],
                idx: [0; 16],
            })
        };
        // Winner: colour peak 8, alpha peak 19.
        let winner = flat([108, 100, 100, 119]);
        let none = [None; PEAK_LOOK_REFINE];
        // Peak 18, but in colour, where the winner's was 8.
        let mut finalists: Finalists = [None; 8];
        finalists[1] = Some((1100, flat([118, 100, 100, 114])));
        let (err, _) = peak_choice(&px, (1000, winner), 19, &finalists, &none, false, true);
        assert_eq!(err, 1000);
        // Without the colour guard (Balanced) it is taken.
        let (err, _) = peak_choice(&px, (1000, winner), 19, &finalists, &none, false, false);
        assert_eq!(err, 1100);
        // Peak 18 in alpha with a colour peak of 12 lowers the block's peak.
        finalists[1] = Some((1100, flat([112, 100, 100, 118])));
        let (err, picked) = peak_choice(&px, (1000, winner), 19, &finalists, &none, false, true);
        assert_eq!((err, picked.peaks(&px)), (1100, [12, 18]));
    }

    #[test]
    fn peak_choice_accepts_a_peak_error_of_255() {
        // An all-black block decodes exactly; one texel of 255 then gives the
        // best candidate and its finalist a peak of 255, where `peak + gain`
        // overflowed a u8.
        let black: Block = [[0, 0, 0, 255]; 16];
        let (_, cand) = search(&black, &BALANCED);
        let mut px = black;
        px[0][0] = 255;
        assert_eq!(cand.peak(&px), 255);
        let mut finalists: Finalists = [None; 8];
        finalists[0] = Some((1000, cand));
        let (err, picked) = peak_choice(
            &px,
            (1000, cand),
            cand.peak(&px),
            &finalists,
            &[None; PEAK_LOOK_REFINE],
            true,
            false,
        );
        assert_eq!(err, 1000);
        assert_eq!(picked.pack(), cand.pack());
    }

    #[test]
    fn weighted_rotations_and_subsets_report_separate_decoded_errors() {
        use objective::{axis_targets, dual_fit, refine_dual, refine_subset, rough, Reps};
        let px: Block = core::array::from_fn(|i| {
            [
                (i * 17) as u8,
                (i * 31) as u8,
                (i * 43) as u8,
                [0, 1, 128, 255][i % 4],
            ]
        });
        for weight in [1, 4] {
            let mut reps = Reps::distinct(&px);
            reps.objective = Some(objective::Objective {
                visible: 0xeeee,
                alpha: 3,
                alpha_weight: weight,
                colour_weight: 1,
            });
            for rotation in 0..4 {
                let map = dual_map(rotation, false);
                let p = px.map(|p| map.map(|c| p[c]));
                let mut r = reps;
                r.objective.as_mut().unwrap().alpha =
                    map.iter().position(|&c| c == 3).unwrap() as u8;
                for (mode, sel) in [(5, 0), (4, 0), (4, 1)] {
                    let t = [axis_targets::<0, 3>(&p, &r), axis_targets::<3, 4>(&p, &r)];
                    let (mut c, mut a) = dual_fit(&p, &r, &t, mode, sel, 1, false);
                    refine_dual(&p, &r, mode, sel, &mut c, &mut a, &QUALITY, false);
                    let d = super::super::super::independent_decoder::decode(&pack_dual(
                        mode, rotation, sel, c, a,
                    ));
                    let mut error = 0;
                    for (i, (a, b)) in px.iter().zip(d).enumerate() {
                        for c in 0..4 {
                            error += reps.objective.unwrap().weight(i, c)
                                * (i32::from(a[c]) - i32::from(b[c])).pow(2);
                        }
                    }
                    assert_eq!(c.err + a.err, error as u32);
                }
            }
            for mode in [1, 3, 7] {
                for mask in [0, 1, 0xaaaa, 0xffff] {
                    let r = reps.masked(mask);
                    let t = if mode == 7 {
                        axis_targets::<0, 4>(&px, &r)
                    } else {
                        axis_targets::<0, 3>(&px, &r)
                    };
                    let mut f = rough(&px, &r, &t, mode, false, 1, u32::MAX);
                    refine_subset(&px, mode, &r, &mut f, false, &QUALITY);
                    assert!(f.d.iter().flatten().all(|&v| (0..=255).contains(&v)));
                    if mask == 0 {
                        assert_eq!(f.err, 0);
                    }
                }
            }
        }
    }

    #[test]
    fn weighted_partition_modes_include_the_fixed_alpha_penalty() {
        use objective::{axis_targets, fit, refine_subset, rough, Reps};
        let px: Block = core::array::from_fn(|i| {
            [
                (i * 17) as u8,
                (i * 29) as u8,
                (i * 43) as u8,
                [0, 1, 254, 255][i % 4],
            ]
        });
        let mut reps = Reps::distinct(&px);
        reps.objective = Some(objective::Objective {
            visible: 0xeeee,
            alpha: 3,
            alpha_weight: 4,
            colour_weight: 4,
        });
        let alpha_penalty: u32 = 4 * px
            .iter()
            .map(|p| (255 - u32::from(p[3])).pow(2))
            .sum::<u32>();
        for mode in [0, 1, 3, 7] {
            for partition in 0..if mode == 0 { 16 } else { 64 } {
                let (err, block) = if mode == 0 {
                    let mut s = masks3(partition)
                        .map(|m| fit::<0, 3>(&px, &reps.masked(m), P0, &L3, 1, 0, false));
                    for (k, m) in masks3(partition).into_iter().enumerate() {
                        refine_subset(&px, mode, &reps.masked(m), &mut s[k], false, &QUALITY);
                    }
                    (
                        s.iter().map(|f| f.err).sum::<u32>() + alpha_penalty,
                        pack3(partition, s),
                    )
                } else {
                    let m = MASKS2[partition][0];
                    let mut s = [!m, m].map(|m| {
                        let r = reps.masked(m);
                        let t = if mode == 7 {
                            axis_targets::<0, 4>(&px, &r)
                        } else {
                            axis_targets::<0, 3>(&px, &r)
                        };
                        rough(&px, &r, &t, mode, false, 1, u32::MAX)
                    });
                    for (k, m) in [!m, m].into_iter().enumerate() {
                        refine_subset(&px, mode, &reps.masked(m), &mut s[k], false, &QUALITY);
                    }
                    (
                        s[0].err + s[1].err + if mode == 7 { 0 } else { alpha_penalty },
                        pack2(mode, partition, s),
                    )
                };
                let d = super::super::super::independent_decoder::decode(&block);
                let mut expected = 0;
                for (i, (a, b)) in px.iter().zip(d).enumerate() {
                    for c in 0..4 {
                        expected += reps.objective.unwrap().weight(i, c)
                            * (i32::from(a[c]) - i32::from(b[c])).pow(2);
                    }
                }
                assert_eq!(err, expected as u32, "mode {mode}, partition {partition}");
            }
        }
    }

    #[test]
    fn block_allowances_cannot_exceed_preset_ceilings() {
        for ceiling in [BALANCED, QUALITY] {
            for units in 1..=4 {
                for cfg in [ceiling.for_block(units), ceiling.for_opaque_block(units)] {
                    assert!(cfg.axis_rounds <= ceiling.axis_rounds);
                    assert!(cfg.refine_iters <= ceiling.refine_iters);
                    assert!(cfg.refine_rounds <= ceiling.refine_rounds);
                    assert!(cfg.rough_iters <= ceiling.rough_iters);
                    assert!(cfg.mode4_iters <= ceiling.mode4_iters);
                    for k in 0..2 {
                        assert!(cfg.duals[k] <= ceiling.duals[k]);
                    }
                    assert!(cfg.parts3 <= ceiling.parts3);
                    for k in 0..3 {
                        assert!(cfg.parts[k] <= ceiling.parts[k]);
                    }
                    // The partition-order array and per-mode ranking buffers
                    // remain sufficient even when every ranking is disjoint.
                    assert!(cfg.parts[0] + cfg.parts[1] <= 32);
                    assert!(cfg.parts.iter().all(|&n| n <= 16));
                }
            }
        }
    }

    fn decoded(px: &Block, block: &[u8; 16]) -> (u64, bool) {
        let d = super::super::super::independent_decoder::decode(block);
        let mut e = 0u64;
        for (a, b) in px.iter().zip(d.iter()) {
            for c in 0..4 {
                let x = i64::from(a[c]) - i64::from(b[c]);
                e += (x * x) as u64;
            }
        }
        (e, d.iter().all(|p| p[3] == 255))
    }

    #[test]
    fn cached_partition_moments_preserve_rankings_and_targets() {
        let mut state = 0x342a_6b91u32;
        for case in 0..64 {
            let mut px = [[0u8; 4]; 16];
            for i in 0..16 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                px[i] = if i > 0 && case % 2 == 0 {
                    px[(state as usize >> 16) % i]
                } else {
                    state.to_le_bytes()
                };
            }
            let reps = Reps::distinct(&px);
            let rgb = Partitions::new::<3>(&px);
            let rgba = Partitions::new::<4>(&px);
            assert_eq!(rgb.est, partition_estimates::<3>(&px));
            assert_eq!(rgba.est, partition_estimates::<4>(&px));
            let counts = [1 + case % 4, 1 + case / 4 % 8];
            assert_eq!(
                best_partitions_31(&rgb.est, counts),
                [
                    best_partitions(&rgb.est, 4.0, counts[0]),
                    best_partitions(&rgb.est, 8.0, counts[1])
                ]
            );
            for p in 0..64 {
                let mask = MASKS2[p][0];
                let r = [reps.masked(!mask), reps.masked(mask)];
                assert_eq!(
                    rgb.targets::<3>(&px, &r, p),
                    r.map(|r| axis_targets::<0, 3>(&px, &r))
                );
                assert_eq!(
                    rgba.targets::<4>(&px, &r, p),
                    r.map(|r| axis_targets::<0, 4>(&px, &r))
                );
            }
        }
    }

    #[test]
    fn omitting_constant_colour_preserves_dual_fits() {
        let mut state = 0x174a_c957u32;
        for case in 0..128 {
            let mut px = [[0u8; 4]; 16];
            let alpha_value = if case % 3 == 0 { 0 } else { 255 };
            for i in 0..16 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                px[i] = if i > 0 && case % 2 == 0 {
                    px[(state as usize >> 16) % i]
                } else {
                    state.to_le_bytes()
                };
                if case % 4 == 1 {
                    px[i][0] = (i * 17) as u8;
                    px[i][1] = ((i * 5 % 16) * 17) as u8;
                    px[i][2] = ((i * 7 % 16) * 17) as u8;
                }
                px[i][3] = alpha_value;
            }
            let reps = Reps::distinct(&px);
            for rotation in 1..4 {
                let map = dual_map(rotation, false);
                let full = px.map(|p| map.map(|c| p[c]));
                let map = dual_map(rotation, true);
                let compact = px.map(|p| map.map(|c| p[c]));
                let ft = [
                    axis_targets::<0, 3>(&full, &reps),
                    scalar_targets(&full, &reps),
                ];
                let ct = [
                    axis_targets::<0, 2>(&compact, &reps),
                    scalar_targets(&compact, &reps),
                ];
                for (mode, sel) in [(5, 0), (4, 0), (4, 1)] {
                    let (mut fc, mut fa) = dual_fit(&full, &reps, &ft, mode, sel, 2, false);
                    let (mut cc, mut ca) = dual_fit(&compact, &reps, &ct, mode, sel, 2, true);
                    for refined in [false, true] {
                        if refined {
                            refine_dual(&full, &reps, mode, sel, &mut fc, &mut fa, &QUALITY, false);
                            refine_dual(
                                &compact, &reps, mode, sel, &mut cc, &mut ca, &QUALITY, true,
                            );
                        }
                        let restored = restore_color(cc, rotation, true, alpha_value);
                        assert_eq!(
                            (fc.err, fc.d, fc.idx),
                            (restored.err, restored.d, restored.idx)
                        );
                        assert_eq!((fa.err, fa.d, fa.idx), (ca.err, ca.d, ca.idx));
                    }
                }
            }
        }
    }

    #[test]
    fn cached_rotation_moments_preserve_axis_targets() {
        let mut state = 0x394b_2c17u32;
        for case in 0..512 {
            let mut px = [[0u8; 4]; 16];
            for i in 0..16 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                px[i] = if i > 0 && case % 3 == 0 {
                    px[(state as usize >> 16) % i]
                } else {
                    state.to_le_bytes()
                };
                if case % 2 == 0 {
                    px[i][3] = 255;
                }
            }
            let reps = Reps::distinct(&px);
            let stats = dual_estimates(&px, &reps);
            for rotation in 0..4 {
                let mut map = [0, 1, 2, 3];
                let mut rotated = px;
                if rotation > 0 {
                    map.swap(rotation - 1, 3);
                    for p in &mut rotated {
                        p.swap(rotation - 1, 3);
                    }
                }
                let mean = map.map(|c| stats.mean[c]);
                let cov = map.map(|c| map.map(|e| stats.cov[c][e]));
                assert_eq!(
                    axis_targets::<0, 3>(&rotated, &reps),
                    axis_endpoints::<0, 3>(&rotated, &reps, &mean, &cov)
                );
                assert_eq!(
                    axis_targets::<3, 4>(&rotated, &reps),
                    axis_endpoints::<3, 4>(&rotated, &reps, &mean, &cov)
                );
                assert_eq!(
                    axis_targets::<3, 4>(&rotated, &reps),
                    scalar_targets(&rotated, &reps)
                );
            }
        }
    }

    #[test]
    fn extra_axis_moves_preserve_score_parity_and_opaque_alpha() {
        for base in [0, 64, 192] {
            for span in [17, 24, 32, 48, 64] {
                for salt in 0..8 {
                    let px: Block = core::array::from_fn(|i| {
                        let x = i as i32 * span / 15;
                        [
                            (base + x).min(255) as u8,
                            (base + x / 2 + (i as i32 + salt) % 3).min(255) as u8,
                            (base + x * 3 / 4).min(255) as u8,
                            255,
                        ]
                    });
                    let reps = Reps::distinct(&px);
                    let mut f = fit::<0, 3>(&px, &reps, P6, &L4, 3, 3, true);
                    let before = f.err;
                    axis_search::<0, 3>(&px, &reps, &mut f, &L4, 4, 4);
                    assert!(f.err <= before);
                    assert!(f.d.iter().all(|e| e[..3].iter().all(|v| v & 1 == 1)));
                    let (error, alpha_kept) = decoded(&px, &pack6(f));
                    assert_eq!(error, u64::from(f.err));
                    assert!(alpha_kept);
                }
            }
        }
    }

    // Every searched candidate reports its decoded SSE, three-subset fits
    // pack legal blocks for every partition, and opaque blocks keep alpha.
    #[test]
    fn searched_candidates_report_their_decoded_error() {
        let mut state = 0x9e37_79b9u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for case in 0..1500 {
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
                    *p = palette[(r as usize + i) % 3].to_le_bytes();
                }
                if case % 4 != 1 {
                    p[3] = 255;
                }
                if case % 11 == 0 {
                    p[3] = 0;
                }
            }
            if solid_rgba(&px).is_some() {
                continue;
            }
            let opaque = px.iter().all(|p| p[3] == 255);
            for (cfg, hide) in [BALANCED, QUALITY]
                .into_iter()
                .flat_map(|c| [false, true].map(|h| (c.hiding_transparent(h), h)))
            {
                let (err, cand) = search(&px, &cfg);
                let block = cand.pack();
                let d = super::super::super::independent_decoder::decode(&block);
                assert_eq!(cand.decode(), d, "case {case}");
                let sse: u64 = px
                    .iter()
                    .zip(d)
                    .map(|(a, b)| {
                        (0..4)
                            .filter(|&c| !hide || c == 3 || a[3] > 0)
                            .map(|c| (i64::from(a[c]) - i64::from(b[c])).pow(2) as u64)
                            .sum::<u64>()
                    })
                    .sum();
                let (_, alpha_kept) = decoded(&px, &block);
                assert_eq!(u64::from(err), sse, "case {case}");
                assert!(!opaque || alpha_kept, "case {case}");
            }
            {
                let partition = case % 16;
                let all = Reps::distinct(&px);
                let mut s = fit_partition3(&px, &all, partition, 1, 0);
                for (k, mask) in masks3(partition).into_iter().enumerate() {
                    refine_subset(&px, 0, &all.masked(mask), &mut s[k], true, &QUALITY);
                }
                let (sse, _) = decoded(&px, &pack3(partition, s));
                let alpha_err: u64 = px.iter().map(|p| (255 - u64::from(p[3])).pow(2)).sum();
                assert_eq!(
                    u64::from(s[0].err + s[1].err + s[2].err) + alpha_err,
                    sse,
                    "case {case}"
                );
            }
        }
    }
}
