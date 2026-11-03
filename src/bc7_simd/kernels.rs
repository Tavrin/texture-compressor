// Expanded inside each ISA module. Arithmetic is slot-wise, with explicit
// multiply then add and the scalar order for every floating-point sum.
macro_rules! kernels {
    ($feature:literal, $n:literal) => {
        #[target_feature(enable = $feature)]
        pub(super) fn select<const C0: usize, const C1: usize>(
            px: &Block,
            reps: &Reps,
            d: &Pair,
            lv: &Levels,
            idx: &mut [u8; 16],
            limit: u32,
            objective: Option<objective::Objective>,
        ) -> u32 {
            debug_assert!(matches!(lv.w.len(), 4 | 8 | 16));
            match (lv.w.len(), objective) {
                (4, None) => select_window::<C0, C1, 4>(px, reps, d, idx, limit),
                (8, None) => select_window::<C0, C1, 8>(px, reps, d, idx, limit),
                (_, None) => select_window::<C0, C1, 16>(px, reps, d, idx, limit),
                (4, Some(o)) => select_full::<C0, C1, 4>(px, reps, d, idx, limit, o),
                (8, Some(o)) => select_full::<C0, C1, 8>(px, reps, d, idx, limit, o),
                (_, Some(o)) => select_full::<C0, C1, 16>(px, reps, d, idx, limit, o),
            }
        }

        // One element per representative ($n representatives per vector).
        // Pixels, palette entries and thresholds are byte tables indexed
        // per element; only the ordered error accumulation and its cutoff stay
        // scalar, so partial selector prefixes match the scalar selector.
        #[target_feature(enable = $feature)]
        fn select_window<const C0: usize, const C1: usize, const N: usize>(
            px: &Block,
            reps: &Reps,
            d: &Pair,
            idx: &mut [u8; 16],
            limit: u32,
        ) -> u32 {
            let (weights, thresholds) = level_tables(N);
            let planes = planes(px);
            let mut palette = [planes[0]; 4];
            let mut dir = [0; 4];
            let mut len2 = 0;
            for c in C0..C1 {
                dir[c] = d[1][c] - d[0][c];
                len2 += dir[c] * dir[c];
                palette[c] = palette_table::<N>(weights, d[0][c], d[1][c]);
            }
            let scale = fsplat(if len2 == 0 { 0.0 } else { 64.0 / len2 as f32 });
            let ids = bytes(&reps.ids);
            let thresholds = bytes(thresholds);
            let iota = load(&core::array::from_fn(|l| l as i32));
            let last = splat(N as i32 - 1);
            let mut err = 0u32;
            let mut base = 0;
            while base < reps.len {
                let r = lookup(ids, add(splat(base as i32), iota));
                let p: [_; 4] = core::array::from_fn(|c| {
                    if (C0..C1).contains(&c) {
                        lookup(planes[c], r)
                    } else {
                        splat(0)
                    }
                });
                // The scalar projection: integer dot product, then the same
                // f32 multiply and add, truncation and clamp.
                let mut t = splat(0);
                for c in C0..C1 {
                    t = add(t, mul(sub(p[c], splat(d[0][c])), splat(dir[c])));
                }
                let k = to_int(fadd(fmul(to_float(t), scale), fsplat(0.5)));
                let k = vmin(vmax(k, splat(0)), splat(64));
                // Nearest entry: a guess g within one of it (checked at
                // compile time), corrected by the two midpoint thresholds
                // around g. Compare masks are -1 where true.
                let g = vmin(vmax(shr6(add(mul(k, last), splat(32))), splat(1)), last);
                let twice = add(k, k);
                let below = sub(g, splat(1));
                let j = sub(
                    sub(below, gt(twice, lookup(thresholds, below))),
                    gt(twice, lookup(thresholds, g)),
                );
                let start = vmax(sub(j, splat(1)), splat(0));
                let end = vmin(add(j, splat(2)), splat(N as i32));
                let mut best = splat(0);
                let mut best_index = start;
                for offset in 0..3 {
                    // Entries past the window may wrap in the table lookup;
                    // `end` excludes them before any comparison counts.
                    let choice = add(start, splat(offset));
                    let mut e = splat(0);
                    for c in C0..C1 {
                        let diff = sub(p[c], lookup(palette[c], choice));
                        e = add(e, mul(diff, diff));
                    }
                    if offset == 0 {
                        best = e;
                    } else {
                        let better = and(gt(end, choice), gt(best, e));
                        best = blend(better, e, best);
                        best_index = blend(better, choice, best_index);
                    }
                }
                let errors = store(best);
                let chosen = store(best_index);
                for l in 0..(reps.len - base).min($n) {
                    let rep = base + l;
                    idx[usize::from(reps.ids[rep])] = chosen[l] as u8;
                    err += (errors[l] * i32::from(reps.counts[rep])) as u32;
                    if err >= limit {
                        return err;
                    }
                }
                base += $n;
            }
            if reps.expand {
                for i in 0..16 {
                    idx[i] = idx[usize::from(reps.of[i])];
                }
            }
            err
        }

        // Weighted selectors scan the full palette in ascending order for
        // $n representatives at once. Colour channels share one weight per
        // texel, so Σ w·x² = colour·Σ x² + alpha·x_alpha² exactly.
        #[target_feature(enable = $feature)]
        fn select_full<const C0: usize, const C1: usize, const N: usize>(
            px: &Block,
            reps: &Reps,
            d: &Pair,
            idx: &mut [u8; 16],
            limit: u32,
            o: objective::Objective,
        ) -> u32 {
            let (weights, _) = level_tables(N);
            let planes = planes(px);
            let mut palette = [[0; 16]; 4];
            for c in C0..C1 {
                for j in 0..N {
                    let w = weights[j];
                    palette[c][j] = ((64 - w) * d[0][c] + w * d[1][c] + 32) >> 6;
                }
            }
            let ids = bytes(&reps.ids);
            let (low, high) = (bytes(&VISIBLE_LOW), bytes(&VISIBLE_HIGH));
            let visible_low = splat(i32::from(o.visible & 0xff));
            let visible_high = splat(i32::from(o.visible >> 8));
            let alpha = usize::from(o.alpha);
            let alpha_weight = splat(i32::from(o.alpha_weight));
            let iota = load(&core::array::from_fn(|l| l as i32));
            let mut err = 0u32;
            let mut base = 0;
            while base < reps.len {
                let r = lookup(ids, add(splat(base as i32), iota));
                let p: [_; 4] = core::array::from_fn(|c| {
                    if (C0..C1).contains(&c) {
                        lookup(planes[c], r)
                    } else {
                        splat(0)
                    }
                });
                let seen = or(
                    and(lookup(low, r), visible_low),
                    and(lookup(high, r), visible_high),
                );
                let colour_weight = mul(vmin(seen, splat(1)), splat(i32::from(o.colour_weight)));
                let mut best = splat(i32::MAX);
                let mut best_index = splat(0);
                for j in 0..N {
                    let mut colour = splat(0);
                    let mut opacity = splat(0);
                    for c in C0..C1 {
                        let diff = sub(p[c], splat(palette[c][j]));
                        if c == alpha {
                            opacity = mul(diff, diff);
                        } else {
                            colour = add(colour, mul(diff, diff));
                        }
                    }
                    let e = add(mul(colour, colour_weight), mul(opacity, alpha_weight));
                    let better = gt(best, e);
                    best = blend(better, e, best);
                    best_index = blend(better, splat(j as i32), best_index);
                }
                let errors = store(best);
                let chosen = store(best_index);
                for l in 0..(reps.len - base).min($n) {
                    let rep = base + l;
                    idx[usize::from(reps.ids[rep])] = chosen[l] as u8;
                    err += (errors[l] * i32::from(reps.counts[rep])) as u32;
                    if err >= limit {
                        return err;
                    }
                }
                base += $n;
            }
            if reps.expand {
                for i in 0..16 {
                    idx[i] = idx[usize::from(reps.of[i])];
                }
            }
            err
        }

        // The 16-entry palette of one channel as a byte table; entries at
        // and after N are never eligible.
        #[target_feature(enable = $feature)]
        fn palette_table<const N: usize>(weights: &[i32; 16], d0: i32, d1: i32) -> Table {
            let (chunks, _) = weights.as_chunks::<$n>();
            let entries: [_; 16 / $n] = core::array::from_fn(|q| {
                if q * $n < N {
                    let w = load(&chunks[q]);
                    shr6(add(
                        add(mul(sub(splat(64), w), splat(d0)), mul(w, splat(d1))),
                        splat(32),
                    ))
                } else {
                    splat(0)
                }
            });
            pack(entries)
        }

        #[target_feature(enable = $feature)]
        pub(super) fn polish<const C0: usize, const C1: usize>(
            px: &Block,
            reps: &Reps,
            f: &mut Fit,
            prec: Prec,
            lv: &Levels,
            objective: Option<objective::Objective>,
        ) -> bool {
            if objective.is_some() || reps.expand {
                polish_inner::<C0, C1, true>(px, reps, f, prec, lv, objective)
            } else {
                polish_inner::<C0, C1, false>(px, reps, f, prec, lv, objective)
            }
        }

        #[target_feature(enable = $feature)]
        fn polish_inner<const C0: usize, const C1: usize, const WEIGHTED: bool>(
            px: &Block,
            reps: &Reps,
            f: &mut Fit,
            prec: Prec,
            lv: &Levels,
            objective: Option<objective::Objective>,
        ) -> bool {
            let n = prec.bits as u32;
            let step = if prec.pbit == PBit::None { 1 } else { 2 };
            let max = (1 << n) - 1;
            let mut d = f.d;
            let mut fixed = 0;
            for c in C0..C1 {
                let (x0, x1) = (d[0][c] >> (8 - n), d[1][c] >> (8 - n));
                let codes = [
                    (x0, x1),
                    (x0 - step, x1),
                    (x0 + step, x1),
                    (x0, x1 - step),
                    (x0, x1 + step),
                ];
                let values = codes.map(|(a, b)| {
                    (
                        expand(a.clamp(0, max) as u32, n) as i32,
                        expand(b.clamp(0, max) as u32, n) as i32,
                    )
                });
                let mut errors = [0; 5];
                for base in (0..5).step_by($n) {
                    let a = load(&core::array::from_fn(|l| values[(base + l).min(4)].0));
                    let b = load(&core::array::from_fn(|l| values[(base + l).min(4)].1));
                    let mut e = splat(0);
                    for (i, count) in reps.iter() {
                        let w = lv.w[usize::from(f.idx[i])];
                        let decoded =
                            shr6(add(add(mul(splat(64 - w), a), mul(splat(w), b)), splat(32)));
                        let diff = sub(splat(i32::from(px[i][c])), decoded);
                        let square = mul(diff, diff);
                        let term = if WEIGHTED {
                            let weight = count * objective.map_or(1, |o| o.weight(i, c));
                            mul(square, splat(weight))
                        } else {
                            square
                        };
                        e = add(e, term);
                    }
                    let v = store(e);
                    let end = (base + $n).min(5);
                    errors[base..end].copy_from_slice(&v[..end - base]);
                }
                let mut best = 0;
                for j in 1..5 {
                    let (a, b) = codes[j];
                    if (0..=max).contains(&a) && (0..=max).contains(&b) && errors[j] < errors[best]
                    {
                        best = j;
                    }
                }
                fixed += errors[best] as u32;
                d[0][c] = values[best].0;
                d[1][c] = values[best].1;
            }
            if fixed >= f.err {
                return false;
            }
            let mut idx = f.idx;
            let err = select::<C0, C1>(px, reps, &d, lv, &mut idx, u32::MAX, objective);
            if err <= fixed {
                *f = Fit { err, d, idx };
            } else {
                f.err = fixed;
                f.d = d;
            }
            true
        }

        // 4/8 blocks, texel/channel/slot SoA; no cross-slot reduction.
        #[target_feature(enable = $feature)]
        pub(super) fn axis(blocks: &[Block; 8]) -> [Pair; 8] {
            let pixels: [[_; 3]; 16] = core::array::from_fn(|i| {
                core::array::from_fn(|c| {
                    load(&core::array::from_fn(|l| i32::from(blocks[l][i][c])))
                })
            });
            let mut s = [splat(0); 3];
            let mut prod = [[splat(0); 3]; 3];
            for pixel in &pixels {
                for c in 0..3 {
                    s[c] = add(s[c], pixel[c]);
                    for e in c..3 {
                        prod[c][e] = add(prod[c][e], mul(pixel[c], pixel[e]));
                    }
                }
            }
            let mut cov = [[fsplat(0.0); 3]; 3];
            let mut mean = [fsplat(0.0); 3];
            for c in 0..3 {
                mean[c] = fdiv(to_float(s[c]), fsplat(16.0));
                for e in c..3 {
                    let v = to_float(sub(mul(splat(16), prod[c][e]), mul(s[c], s[e])));
                    cov[c][e] = v;
                    cov[e][c] = v;
                }
            }
            let diag = [fstore(cov[0][0]), fstore(cov[1][1]), fstore(cov[2][2])];
            let top: [usize; $n] = core::array::from_fn(|l| {
                let mut top = 0;
                for c in 0..3 {
                    if diag[c][l] > diag[top][l] {
                        top = c;
                    }
                }
                top
            });
            let mut v = core::array::from_fn::<_, 3, _>(|c| {
                let columns = cov[c].map(|v| fstore(v));
                fload(&core::array::from_fn(|l| columns[top[l]][l]))
            });
            for _ in 0..3 {
                let mut next = [fsplat(0.0); 3];
                for c in 0..3 {
                    for e in 0..3 {
                        next[c] = fadd(next[c], fmul(cov[c][e], v[e]));
                    }
                }
                let values = next.map(|v| fstore(v));
                let largest: [f32; $n] = core::array::from_fn(|l| {
                    let mut m = 0.0f32;
                    for c in 0..3 {
                        m = m.max(values[c][l].abs());
                    }
                    // Zero-covariance lanes are ignored below; other lanes
                    // have a nonzero largest on every power iteration.
                    if m <= 0.0 {
                        1.0
                    } else {
                        m
                    }
                });
                for c in 0..3 {
                    v[c] = fdiv(next[c], fload(&largest));
                }
            }
            let mut len2 = fsplat(0.0);
            for c in 0..3 {
                len2 = fadd(len2, fmul(v[c], v[c]));
            }
            let mut lo = [f32::MAX; $n];
            let mut hi = [f32::MIN; $n];
            for pixel in pixels {
                let mut dot = fsplat(0.0);
                for c in 0..3 {
                    dot = fadd(dot, fmul(fsub(to_float(pixel[c]), mean[c]), v[c]));
                }
                let dot = fstore(dot);
                for l in 0..$n {
                    lo[l] = lo[l].min(dot[l]);
                    hi[l] = hi[l].max(dot[l]);
                }
            }
            let mut out = [[[0; 4]; 2]; 8];
            for c in 0..3 {
                let u = fdiv(v[c], len2);
                let a = fstore(fadd(fadd(mean[c], fmul(fload(&lo), u)), fsplat(0.5)));
                let b = fstore(fadd(fadd(mean[c], fmul(fload(&hi), u)), fsplat(0.5)));
                let m = fstore(mean[c]);
                for l in 0..$n {
                    if diag[top[l]][l] <= 0.0 {
                        out[l][0][c] = (m[l] + 0.5) as i32;
                        out[l][1][c] = out[l][0][c];
                    } else {
                        out[l][0][c] = (a[l] as i32).clamp(0, 255);
                        out[l][1][c] = (b[l] as i32).clamp(0, 255);
                    }
                }
            }
            out
        }

        // Initial mode-6 palette/error evaluation across blocks. Invalid
        // candidate offsets and inactive image blocks never affect live fits.
        #[target_feature(enable = $feature)]
        fn select_batch(blocks: &[Block; 8], endpoints: &[Pair; 8]) -> [Fit; 8] {
            let mut out = core::array::from_fn(|l| Fit {
                err: 0,
                d: endpoints[l],
                idx: [0; 16],
            });
            let d: [[_; 3]; 2] = core::array::from_fn(|k| {
                core::array::from_fn(|c| load(&core::array::from_fn(|l| endpoints[l][k][c])))
            });
            let dir = core::array::from_fn::<_, 3, _>(|c| sub(d[1][c], d[0][c]));
            let mut len2 = splat(0);
            for c in 0..3 {
                len2 = add(len2, mul(dir[c], dir[c]));
            }
            let lengths = store(len2);
            let scale = fload(&lengths.map(|n| if n == 0 { 0.0 } else { 64.0 / n as f32 }));
            for i in 0..16 {
                let pixel: [_; 3] = core::array::from_fn(|c| {
                    load(&core::array::from_fn(|l| i32::from(blocks[l][i][c])))
                });
                let mut t = splat(0);
                for c in 0..3 {
                    t = add(t, mul(sub(pixel[c], d[0][c]), dir[c]));
                }
                let projections = fstore(fadd(fmul(to_float(t), scale), fsplat(0.5)));
                let j = projections.map(|v| usize::from(L4.near[(v as i32).clamp(0, 64) as usize]));
                let start = j.map(|j| j.saturating_sub(1));
                let end = j.map(|j| (j + 2).min(16));
                let mut best = [i32::MAX; $n];
                let mut index = [0; $n];
                for offset in 0..3 {
                    let choices: [usize; $n] =
                        core::array::from_fn(|l| (start[l] + offset).min(15));
                    let w = load(&choices.map(|j| W4[j]));
                    let mut error = splat(0);
                    for c in 0..3 {
                        let decoded = shr6(add(
                            add(mul(sub(splat(64), w), d[0][c]), mul(w, d[1][c])),
                            splat(32),
                        ));
                        let diff = sub(pixel[c], decoded);
                        error = add(error, mul(diff, diff));
                    }
                    let errors = store(error);
                    for l in 0..$n {
                        if start[l] + offset < end[l] && errors[l] < best[l] {
                            best[l] = errors[l];
                            index[l] = choices[l] as u8;
                        }
                    }
                }
                for l in 0..$n {
                    out[l].err += best[l] as u32;
                    out[l].idx[i] = index[l];
                }
            }
            out
        }

        #[target_feature(enable = $feature)]
        pub(super) fn seed(blocks: &[Block; 8]) -> [Fit; 8] {
            let targets = axis(blocks);
            let d = targets.map(|t| quantize::<0, 3>(&t, P6, true));
            let first = select_batch(blocks, &d);
            let mut next = d;
            for l in 0..$n {
                let reps = Reps::distinct(&blocks[l]);
                if let Some(t) = least_squares::<0, 3>(&blocks[l], &reps, &first[l].idx, &L4) {
                    next[l] = quantize::<0, 3>(&t, P6, true);
                }
            }
            let second = select_batch(blocks, &next);
            core::array::from_fn(|l| {
                if second[l].err < first[l].err {
                    second[l]
                } else {
                    first[l]
                }
            })
        }

        #[target_feature(enable = $feature)]
        pub(super) fn estimates<const CH: usize>(moments: &[Moments; 8]) -> [(f32, f32); 8] {
            // Only the count, sums and products of CH channels are read.
            let used = 1 + CH + CH * (CH + 1) / 2;
            let m: [_; 15] = core::array::from_fn(|k| {
                if k < used {
                    load(&core::array::from_fn(|l| moments[l].m[k]))
                } else {
                    splat(0)
                }
            });
            let mut cov = [[fsplat(0.0); 4]; 4];
            let mut trace = fsplat(0.0);
            let mut k = 1 + CH;
            for c in 0..CH {
                for e in c..CH {
                    let v = to_float(sub(mul(m[0], m[k]), mul(m[1 + c], m[1 + e])));
                    cov[c][e] = v;
                    cov[e][c] = v;
                    k += 1;
                }
                trace = fadd(trace, cov[c][c]);
            }
            let diag = core::array::from_fn::<_, 4, _>(|c| fstore(cov[c][c]));
            let top: [usize; $n] = core::array::from_fn(|l| {
                let mut t = 0;
                for c in 0..CH {
                    if diag[c][l] > diag[t][l] {
                        t = c;
                    }
                }
                t
            });
            let w: [_; 4] = core::array::from_fn(|c| {
                let cols = cov[c].map(|v| fstore(v));
                fload(&core::array::from_fn(|l| cols[top[l]][l]))
            });
            let mut ww = fsplat(0.0);
            let mut wcw = fsplat(0.0);
            for c in 0..CH {
                let mut cw = fsplat(0.0);
                for e in 0..CH {
                    cw = fadd(cw, fmul(cov[c][e], w[e]));
                }
                ww = fadd(ww, fmul(w[c], w[c]));
                wcw = fadd(wcw, fmul(w[c], cw));
            }
            let quot = fstore(fdiv(wcw, ww));
            let traces = fstore(trace);
            let lambda = fload(&core::array::from_fn(|l| quot[l].min(traces[l])));
            let nf = to_float(m[0]);
            let res = fstore(fdiv(fsub(trace, lambda), nf));
            let var = fstore(fdiv(lambda, nf));
            let mut out = [(0.0, 0.0); 8];
            for l in 0..$n {
                if traces[l] > 0.0 {
                    out[l] = (res[l], var[l]);
                }
            }
            out
        }
    };
}
