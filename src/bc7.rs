//! Experimental scalar BC7 encoding. Format: Microsoft BC7 mode reference.
use crate::Effort;

#[path = "bc7_fast.rs"]
mod fast;
#[path = "bc7_tables.rs"]
pub(super) mod tables;
#[cfg(test)]
use tables::{ANCHOR2, ANCHOR3A, ANCHOR3B};
use tables::{PART2, PART3};

#[cfg(test)]
const W: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

#[cfg(test)]
fn quantize(e: [[i32; 4]; 2], p: [i32; 2]) -> [[i32; 4]; 2] {
    let mut q = [[0; 4]; 2];
    for k in 0..2 {
        for c in 0..4 {
            q[k][c] = (((e[k][c] - p[k] + 1) / 2).clamp(0, 127) * 2) + p[k];
        }
    }
    q
}
#[cfg(test)]
fn score(pixels: &[[u8; 4]; 16], e: [[i32; 4]; 2]) -> (u64, [usize; 16]) {
    let mut indices = [0; 16];
    let mut total = 0;
    for (i, pixel) in pixels.iter().enumerate() {
        let mut best = (u32::MAX, 0);
        for (idx, &w) in W.iter().enumerate() {
            let mut err = 0;
            for c in 0..4 {
                let value = ((64 - w) * e[0][c] + w * e[1][c] + 32) >> 6;
                let d = i32::from(pixel[c]) - value;
                err += (d * d) as u32;
            }
            best = best.min((err, idx));
        }
        total += u64::from(best.0);
        indices[i] = best.1;
    }
    (total, indices)
}
#[cfg(test)]
fn fit(pixels: &[[u8; 4]; 16], indices: [usize; 16], old: [[i32; 4]; 2]) -> [[i32; 4]; 2] {
    let mut aa = 0i64;
    let mut ab = 0i64;
    let mut bb = 0i64;
    let mut ax = [0i64; 4];
    let mut bx = [0i64; 4];
    for (pixel, idx) in pixels.iter().zip(indices) {
        let b = i64::from(W[idx]);
        let a = 64 - b;
        aa += a * a;
        ab += a * b;
        bb += b * b;
        for c in 0..4 {
            ax[c] += a * i64::from(pixel[c]) * 64;
            bx[c] += b * i64::from(pixel[c]) * 64;
        }
    }
    let det = aa * bb - ab * ab;
    if det == 0 {
        return old;
    }
    let mut out = old;
    for c in 0..4 {
        out[0][c] = ((ax[c] * bb - bx[c] * ab + det / 2) / det).clamp(0, 255) as i32;
        out[1][c] = ((bx[c] * aa - ax[c] * ab + det / 2) / det).clamp(0, 255) as i32;
    }
    out
}
fn put(bits: &mut u128, offset: &mut u32, value: u32, width: u32) {
    *bits |= u128::from(value) << *offset;
    *offset += width;
}
// Improve the decoded SSE with selectors fixed, preserving each endpoint's
// shared p-bit. Re-selecting indices afterward cannot increase the error.
#[cfg(test)]
fn refine_mode6(
    pixels: &[[u8; 4]; 16],
    endpoints: [[i32; 4]; 2],
    indices: [usize; 16],
    old_error: u64,
) -> (u64, [[i32; 4]; 2], [usize; 16]) {
    let mut e = endpoints;
    let opaque = pixels.iter().all(|p| p[3] == 255);
    for k in 0..2 {
        for c in 0..if opaque { 3 } else { 4 } {
            let original = e[k][c];
            let fixed_error = |value: i32, endpoint: &[[i32; 4]; 2]| -> u64 {
                pixels
                    .iter()
                    .zip(indices)
                    .map(|(p, idx)| {
                        let w = W[idx];
                        let x = if k == 0 { value } else { endpoint[0][c] };
                        let y = if k == 1 { value } else { endpoint[1][c] };
                        let v = ((64 - w) * x + w * y + 32) >> 6;
                        let d = i32::from(p[c]) - v;
                        (d * d) as u64
                    })
                    .sum()
            };
            let mut best = (fixed_error(original, &e), original);
            for v in [original - 2, original + 2] {
                if (0..=255).contains(&v) {
                    let error = fixed_error(v, &e);
                    if error < best.0 {
                        best = (error, v);
                    }
                }
            }
            e[k][c] = best.1;
        }
    }
    let (error, idx) = score(pixels, e);
    if error < old_error {
        (error, e, idx)
    } else {
        (old_error, endpoints, indices)
    }
}

#[cfg(test)]
fn mode6(pixels: &[[u8; 4]; 16], effort: Effort) -> ([u8; 16], u64) {
    let mut lo = [255; 4];
    let mut hi = [0; 4];
    for pixel in pixels {
        for c in 0..4 {
            lo[c] = lo[c].min(i32::from(pixel[c]));
            hi[c] = hi[c].max(i32::from(pixel[c]));
        }
    }
    let opaque = lo[3] == 255;
    let mut e = [lo, hi];
    let mut best = (u64::MAX, [[0; 4]; 2], [0usize; 16]);
    let rounds = match effort {
        Effort::Fast => 1,
        Effort::Balanced => 3,
        Effort::Quality => 5,
    };
    for round in 0..rounds {
        let mut local = (u64::MAX, [[0; 4]; 2], [0usize; 16]);
        for p0 in 0..=1 {
            for p1 in 0..=1 {
                if opaque && (p0 != 1 || p1 != 1) {
                    continue;
                }
                let q = quantize(e, [p0, p1]);
                let (err, idx) = score(pixels, q);
                local = local.min((err, q, idx));
            }
        }
        best = best.min(local);
        if round + 1 < rounds {
            e = fit(pixels, local.2, local.1);
            if opaque {
                e[0][3] = 255;
                e[1][3] = 255;
            }
        }
    }
    // Spend this bounded refinement on low-residual blocks; each candidate is
    // scored against the decoded palette, and the seed is retained on a tie.
    let best = if best.0 > 0 && best.0 <= 64 {
        refine_mode6(pixels, best.1, best.2, best.0)
    } else {
        best
    };
    let (error, mut endpoints, mut indices) = best;
    if indices[0] >= 8 {
        endpoints.swap(0, 1);
        for i in &mut indices {
            *i = 15 - *i;
        }
    }
    let mut bits = 0u128;
    let mut offset = 0;
    put(&mut bits, &mut offset, 64, 7);
    for c in 0..4 {
        for ep in &endpoints {
            put(&mut bits, &mut offset, (ep[c] >> 1) as u32, 7);
        }
    }
    for ep in &endpoints {
        put(&mut bits, &mut offset, (ep[0] & 1) as u32, 1);
    }
    for (i, &idx) in indices.iter().enumerate() {
        put(
            &mut bits,
            &mut offset,
            idx as u32,
            if i == 0 { 3 } else { 4 },
        );
    }
    debug_assert_eq!(offset, 128);
    (bits.to_le_bytes(), error)
}

// Mode 5 provides separate color and scalar selectors. Rotations select which
// source channel is represented by the scalar stream.
#[cfg(test)]
fn mode5(pixels: &[[u8; 4]; 16], rotation: usize) -> ([u8; 16], u64) {
    let mut src = *pixels;
    if rotation != 0 {
        for p in &mut src {
            p.swap(rotation - 1, 3);
        }
    }
    let mut endpoints = [[0i32; 4]; 2];
    for c in 0..4 {
        let lo = src.iter().map(|p| p[c]).min().unwrap();
        let hi = src.iter().map(|p| p[c]).max().unwrap();
        endpoints[0][c] = i32::from(lo);
        endpoints[1][c] = i32::from(hi);
    }
    let mut q = [[0i32; 4]; 2];
    for k in 0..2 {
        for c in 0..3 {
            let v = (endpoints[k][c] * 127 + 127) / 255;
            q[k][c] = (v << 1) | (v >> 6);
        }
        q[k][3] = endpoints[k][3];
    }
    let weights = [0, 21, 43, 64];
    let mut indexes = [[0usize; 16]; 2];
    let mut error = 0u64;
    for (i, pixel) in src.iter().enumerate() {
        for (stream, selectors) in indexes.iter_mut().enumerate() {
            let mut best = (u32::MAX, 0);
            for (index, &w) in weights.iter().enumerate() {
                let mut e = 0;
                for c in 0..4 {
                    if (c == 3) != (stream == 1) {
                        continue;
                    }
                    let v = ((64 - w) * q[0][c] + w * q[1][c] + 32) >> 6;
                    let d = i32::from(pixel[c]) - v;
                    e += (d * d) as u32;
                }
                best = best.min((e, index));
            }
            error += u64::from(best.0);
            selectors[i] = best.1;
        }
    }
    for (stream, selectors) in indexes.iter_mut().enumerate() {
        if selectors[0] >= 2 {
            let (a, b) = q.split_at_mut(1);
            for (c, (left, right)) in a[0].iter_mut().zip(b[0].iter_mut()).enumerate() {
                if (c == 3) == (stream == 1) {
                    core::mem::swap(left, right);
                }
            }
            for index in selectors {
                *index = 3 - *index;
            }
        }
    }
    let mut bits = 0u128;
    let mut offset = 0;
    put(&mut bits, &mut offset, 32, 6);
    put(&mut bits, &mut offset, rotation as u32, 2);
    for c in 0..4 {
        for ep in &q {
            put(
                &mut bits,
                &mut offset,
                if c == 3 { ep[c] } else { ep[c] >> 1 } as u32,
                if c == 3 { 8 } else { 7 },
            );
        }
    }
    for stream in indexes {
        for (i, index) in stream.into_iter().enumerate() {
            put(
                &mut bits,
                &mut offset,
                index as u32,
                if i == 0 { 1 } else { 2 },
            );
        }
    }
    debug_assert_eq!(offset, 128);
    (bits.to_le_bytes(), error)
}

// Candidate partitioned modes. Format data are the Khronos P2/A2 tables.
#[cfg(test)]
fn partition_rank(pixels: &[[u8; 4]; 16], pattern: u32, channels: usize, subsets: usize) -> u64 {
    // Error of independent constant subsets, less the variance in the most
    // variable channel. This cheap ranking only selects search candidates.
    let mut sum = [[0i64; 4]; 3];
    let mut square = [[0i64; 4]; 3];
    let mut count = [0i64; 3];
    for (i, pixel) in pixels.iter().enumerate() {
        let subset = ((pattern >> (2 * i)) & 3) as usize;
        count[subset] += 1;
        for c in 0..channels {
            let x = i64::from(pixel[c]);
            sum[subset][c] += x;
            square[subset][c] += x * x;
        }
    }
    let mut score = 0;
    for s in 0..subsets {
        let mut total = 0;
        let mut largest = 0;
        for c in 0..channels {
            let variance = square[s][c] * count[s] - sum[s][c] * sum[s][c];
            total += variance;
            largest = largest.max(variance);
        }
        score += (total - largest) / count[s];
    }
    score as u64
}

#[cfg(test)]
fn subset_fit(
    pixels: &[[u8; 4]; 16],
    pattern: u32,
    subset: usize,
    mode: usize,
    rounds: usize,
) -> (u64, [[i32; 4]; 2], [usize; 16]) {
    let channels = if mode == 7 { 4 } else { 3 };
    let width = [4, 6, 5, 7, 0, 0, 0, 5][mode];
    let has_p = mode != 2;
    let weights: &[i32] = if mode == 0 || mode == 1 {
        &[0, 9, 18, 27, 37, 46, 55, 64]
    } else {
        &[0, 21, 43, 64]
    };
    let mut low = [255u8; 4];
    let mut high = [0u8; 4];
    for (i, p) in pixels.iter().enumerate() {
        if ((pattern >> (2 * i)) & 3) as usize != subset {
            continue;
        }
        for c in 0..channels {
            low[c] = low[c].min(p[c]);
            high[c] = high[c].max(p[c]);
        }
    }
    let opaque = mode == 7 && low[3] == 255;
    let axis = (0..channels).max_by_key(|&c| high[c] - low[c]).unwrap();
    let mut sorted = (0..16).filter(|&i| ((pattern >> (2 * i)) & 3) as usize == subset);
    let first = sorted.next().unwrap();
    let mut min_i = first;
    let mut max_i = first;
    for i in sorted {
        if pixels[i][axis] < pixels[min_i][axis] {
            min_i = i;
        }
        if pixels[i][axis] > pixels[max_i][axis] {
            max_i = i;
        }
    }
    let mut ep = [[0i32; 4]; 2];
    for c in 0..channels {
        ep[0][c] = i32::from(pixels[min_i][c]);
        ep[1][c] = i32::from(pixels[max_i][c]);
    }
    let mut best = (u64::MAX, [[0; 4]; 2], [0usize; 16]);
    for round in 0..rounds {
        let mut local = (u64::MAX, [[0; 4]; 2], [0usize; 16]);
        for p in 0..if mode == 2 {
            1
        } else if mode == 1 {
            2
        } else {
            4
        } {
            if opaque && p != 3 {
                continue;
            }
            let mut q = [[255i32; 4]; 2];
            for k in 0..2 {
                let pbit = if mode == 1 { p } else { (p >> k) & 1 };
                let max = (1 << width) - 1;
                for c in 0..channels {
                    let raw = if has_p {
                        (((ep[k][c] * ((max << 1) + 1) + 127) / 255 - pbit + 1) / 2).clamp(0, max)
                    } else {
                        (ep[k][c] * max + 127) / 255
                    };
                    let v = if has_p { (raw << 1) | pbit } else { raw };
                    let n = width + i32::from(has_p);
                    q[k][c] = (v << (8 - n)) | (v >> (2 * n - 8));
                }
            }
            let mut indices = [0usize; 16];
            let mut error = 0u64;
            for (i, pixel) in pixels.iter().enumerate() {
                if ((pattern >> (2 * i)) & 3) as usize != subset {
                    continue;
                }
                let mut choice = (u32::MAX, 0);
                for (j, &w) in weights.iter().enumerate() {
                    let mut e = 0;
                    for c in 0..channels {
                        let v = ((64 - w) * q[0][c] + w * q[1][c] + 32) >> 6;
                        let d = i32::from(pixel[c]) - v;
                        e += (d * d) as u32;
                    }
                    choice = choice.min((e, j));
                }
                error += u64::from(choice.0);
                indices[i] = choice.1;
            }
            local = local.min((error, q, indices));
        }
        best = best.min(local);
        if round + 1 == rounds {
            break;
        }
        let mut aa = 0i64;
        let mut ab = 0i64;
        let mut bb = 0i64;
        let mut ax = [0i64; 4];
        let mut bx = [0i64; 4];
        for (i, pixel) in pixels.iter().enumerate() {
            if ((pattern >> (2 * i)) & 3) as usize != subset {
                continue;
            }
            let b = i64::from(weights[local.2[i]]);
            let a = 64 - b;
            aa += a * a;
            ab += a * b;
            bb += b * b;
            for c in 0..channels {
                ax[c] += a * i64::from(pixel[c]) * 64;
                bx[c] += b * i64::from(pixel[c]) * 64;
            }
        }
        let det = aa * bb - ab * ab;
        if det == 0 {
            break;
        }
        for c in 0..channels {
            ep[0][c] = ((ax[c] * bb - bx[c] * ab + det / 2) / det).clamp(0, 255) as i32;
            ep[1][c] = ((bx[c] * aa - ax[c] * ab + det / 2) / det).clamp(0, 255) as i32;
        }
        if opaque {
            ep[0][3] = 255;
            ep[1][3] = 255;
        }
    }
    best
}

// Conservative continuous-line SSE bound. Gershgorin bounds the largest
// covariance eigenvalue from above; final per-channel interpolation rounding
// can move an m-by-c vector by at most sqrt(m*c)/2 <= 4 in Euclidean norm.
#[cfg(test)]
fn partition_lower_bound(
    pixels: &[[u8; 4]; 16],
    pattern: u32,
    subsets: usize,
    channels: usize,
) -> u64 {
    let mut count = [0i64; 3];
    let mut sums = [[0i64; 4]; 3];
    let mut products = [[[0i64; 4]; 4]; 3];
    for (i, pixel) in pixels.iter().enumerate() {
        let s = ((pattern >> (2 * i)) & 3) as usize;
        count[s] += 1;
        for c in 0..channels {
            let v = i64::from(pixel[c]);
            sums[s][c] += v;
            for d in 0..channels {
                products[s][c][d] += v * i64::from(pixel[d]);
            }
        }
    }
    let mut bound = 0;
    for s in 0..subsets {
        let mut trace = 0i64;
        let mut eigen_upper = 0i64;
        for c in 0..channels {
            let mut row = 0;
            for d in 0..channels {
                let covariance = count[s] * products[s][c][d] - sums[s][c] * sums[s][d];
                row += covariance.abs();
                if c == d {
                    trace += covariance;
                }
            }
            eigen_upper = eigen_upper.max(row);
        }
        let squared = ((trace - eigen_upper).max(0) / count[s]) as u64;
        let distance = squared.isqrt().saturating_sub(4);
        bound += distance * distance;
    }
    bound
}

#[cfg(test)]
fn partition_mode_ranked(
    pixels: &[[u8; 4]; 16],
    mode: usize,
    effort: Effort,
    ranks: &[(u64, usize); 64],
) -> ([u8; 16], u64) {
    partition_mode_ranked_with_bound(pixels, mode, effort, ranks, true)
}

#[cfg(test)]
fn partition_mode_ranked_with_bound(
    pixels: &[[u8; 4]; 16],
    mode: usize,
    effort: Effort,
    ranks: &[(u64, usize); 64],
    use_bound: bool,
) -> ([u8; 16], u64) {
    let channels = if mode == 7 { 4 } else { 3 };
    let width = [4, 6, 5, 7, 0, 0, 0, 5][mode];
    let ib = if mode == 0 || mode == 1 { 3 } else { 2 };
    let weights_count = 1usize << ib;
    let subsets = if mode == 0 || mode == 2 { 3 } else { 2 };
    let partitions = if mode == 0 { 16 } else { 64 };
    let (limit, rounds) = match effort {
        Effort::Fast => (2, 1),
        Effort::Balanced => (6, 2),
        Effort::Quality => (16, 4),
    };
    let mut best = ([0u8; 16], u64::MAX);
    for &(_, partition) in &ranks[..limit.min(partitions)] {
        let pattern = if subsets == 3 {
            PART3[partition]
        } else {
            PART2[partition]
        };
        if use_bound
            && best.1 != u64::MAX
            && partition_lower_bound(pixels, pattern, subsets, channels) >= best.1
        {
            continue;
        }
        let mut result = [
            subset_fit(pixels, pattern, 0, mode, rounds),
            subset_fit(pixels, pattern, 1, mode, rounds),
            (0, [[0; 4]; 2], [0; 16]),
        ];
        if subsets == 3 {
            result[2] = subset_fit(pixels, pattern, 2, mode, rounds);
        }
        let error = result.iter().map(|r| r.0).sum();
        if error >= best.1 {
            continue;
        }
        let anchors = if subsets == 3 {
            [0, ANCHOR3A[partition], ANCHOR3B[partition]]
        } else {
            [0, ANCHOR2[partition], 16]
        };
        for s in 0..subsets {
            if result[s].2[anchors[s]] >= weights_count / 2 {
                result[s].1.swap(0, 1);
                for (i, index) in result[s].2.iter_mut().enumerate() {
                    if ((pattern >> (2 * i)) & 3) as usize == s {
                        *index = weights_count - 1 - *index;
                    }
                }
            }
        }
        let mut bits = 0u128;
        let mut offset = 0;
        put(&mut bits, &mut offset, 1 << mode, (mode + 1) as u32);
        put(
            &mut bits,
            &mut offset,
            partition as u32,
            if mode == 0 { 4 } else { 6 },
        );
        for c in 0..channels {
            for subset in result.iter().take(subsets) {
                for ep in &subset.1 {
                    put(
                        &mut bits,
                        &mut offset,
                        (ep[c] >> (8 - width)) as u32,
                        width as u32,
                    );
                }
            }
        }
        if mode != 2 {
            for subset in result.iter().take(subsets) {
                if mode == 1 {
                    put(
                        &mut bits,
                        &mut offset,
                        ((subset.1[0][0] >> (7 - width)) & 1) as u32,
                        1,
                    );
                } else {
                    for ep in &subset.1 {
                        put(
                            &mut bits,
                            &mut offset,
                            ((ep[0] >> (7 - width)) & 1) as u32,
                            1,
                        );
                    }
                }
            }
        }
        for i in 0..16 {
            let s = ((pattern >> (2 * i)) & 3) as usize;
            put(
                &mut bits,
                &mut offset,
                result[s].2[i] as u32,
                (ib - usize::from(anchors.contains(&i))) as u32,
            );
        }
        debug_assert_eq!(offset, 128);
        best = (bits.to_le_bytes(), error);
    }
    best
}

#[cfg(test)]
fn partition_mode(pixels: &[[u8; 4]; 16], mode: usize, effort: Effort) -> ([u8; 16], u64) {
    let channels = if mode == 7 { 4 } else { 3 };
    let subsets = if mode == 0 || mode == 2 { 3 } else { 2 };
    let partitions = if mode == 0 { 16 } else { 64 };
    let mut ranks = [(u64::MAX, 0usize); 64];
    for (p, r) in ranks.iter_mut().enumerate().take(partitions) {
        *r = (
            partition_rank(
                pixels,
                if subsets == 3 { PART3[p] } else { PART2[p] },
                channels,
                subsets,
            ),
            p,
        );
    }
    ranks.sort_unstable();
    partition_mode_ranked(pixels, mode, effort, &ranks)
}

#[cfg(test)]
fn mode4(pixels: &[[u8; 4]; 16], rotation: usize, selection: usize) -> ([u8; 16], u64) {
    let mut src = *pixels;
    if rotation != 0 {
        for pixel in &mut src {
            pixel.swap(rotation - 1, 3);
        }
    }
    let mut ep = [[0i32; 4]; 2];
    for c in 0..4 {
        ep[0][c] = i32::from(src.iter().map(|p| p[c]).min().unwrap());
        ep[1][c] = i32::from(src.iter().map(|p| p[c]).max().unwrap());
    }
    // Use the pixel extremes along the widest vector channel, preserving
    // opposing channel gradients instead of a diagonal bounding box.
    let axis = (0..3).max_by_key(|&c| ep[1][c] - ep[0][c]).unwrap();
    let lo = src.iter().min_by_key(|p| p[axis]).unwrap();
    let hi = src.iter().max_by_key(|p| p[axis]).unwrap();
    for c in 0..3 {
        ep[0][c] = i32::from(lo[c]);
        ep[1][c] = i32::from(hi[c]);
    }
    let mut q = [[0i32; 4]; 2];
    for k in 0..2 {
        for c in 0..4 {
            let width = if c == 3 { 6 } else { 5 };
            let v = (ep[k][c] * ((1 << width) - 1) + 127) / 255;
            q[k][c] = (v << (8 - width)) | (v >> (2 * width - 8));
        }
    }
    let weights2 = [0, 21, 43, 64];
    let weights3 = [0, 9, 18, 27, 37, 46, 55, 64];
    let mut indices = [[0usize; 16]; 2];
    let mut error = 0u64;
    // Stream zero is the primary 2-bit stream; selector chooses its channels.
    for (stream, selectors) in indices.iter_mut().enumerate() {
        let weights: &[i32] = if stream == 0 { &weights2 } else { &weights3 };
        let scalar = (stream == 1) != (selection == 1);
        for (i, pixel) in src.iter().enumerate() {
            let mut best = (u32::MAX, 0);
            for (j, &w) in weights.iter().enumerate() {
                let mut e = 0;
                for c in 0..4 {
                    if (c == 3) != scalar {
                        continue;
                    }
                    let v = ((64 - w) * q[0][c] + w * q[1][c] + 32) >> 6;
                    let d = i32::from(pixel[c]) - v;
                    e += (d * d) as u32;
                }
                best = best.min((e, j));
            }
            error += u64::from(best.0);
            selectors[i] = best.1;
        }
        if selectors[0] >= weights.len() / 2 {
            let (a, b) = q.split_at_mut(1);
            for c in 0..4 {
                if (c == 3) == scalar {
                    core::mem::swap(&mut a[0][c], &mut b[0][c]);
                }
            }
            for index in selectors {
                *index = weights.len() - 1 - *index;
            }
        }
    }
    let mut bits = 0u128;
    let mut offset = 0;
    put(&mut bits, &mut offset, 16, 5);
    put(&mut bits, &mut offset, rotation as u32, 2);
    put(&mut bits, &mut offset, selection as u32, 1);
    for c in 0..4 {
        for endpoint in &q {
            let width = if c == 3 { 6 } else { 5 };
            put(
                &mut bits,
                &mut offset,
                (endpoint[c] >> (8 - width)) as u32,
                width as u32,
            );
        }
    }
    for (stream, selectors) in indices.iter().enumerate() {
        for (i, &index) in selectors.iter().enumerate() {
            put(
                &mut bits,
                &mut offset,
                index as u32,
                (if stream == 0 { 2 } else { 3 }) - u32::from(i == 0),
            );
        }
    }
    debug_assert_eq!(offset, 128);
    (bits.to_le_bytes(), error)
}

// Exact constant RGBA path. For the mode-6 midpoint, same-parity endpoints
// bracketing a byte of opposite parity round back to it at weight 30.
fn solid_rgba(pixels: &[[u8; 4]; 16]) -> Option<[u8; 16]> {
    let p = pixels[0];
    if pixels[1..].iter().any(|q| *q != p) {
        return None;
    }
    let has_zero = p.contains(&0);
    let has_max = p.contains(&255);
    if !(has_zero && has_max) {
        let parity = i32::from(!has_zero);
        let mut e = [[0i32; 4]; 2];
        for c in 0..4 {
            let v = i32::from(p[c]);
            e[0][c] = if v & 1 != parity { v - 1 } else { v };
            e[1][c] = if v & 1 != parity { v + 1 } else { v };
        }
        let mut bits = 0u128;
        let mut offset = 0;
        put(&mut bits, &mut offset, 64, 7);
        for c in 0..4 {
            for endpoint in &e {
                put(&mut bits, &mut offset, (endpoint[c] >> 1) as u32, 7);
            }
        }
        put(&mut bits, &mut offset, parity as u32, 1);
        put(&mut bits, &mut offset, parity as u32, 1);
        for i in 0..16 {
            put(&mut bits, &mut offset, 7, if i == 0 { 3 } else { 4 });
        }
        debug_assert_eq!(offset, 128);
        return Some(bits.to_le_bytes());
    }
    let mut q = [0u32; 3];
    for c in 0..3 {
        q[c] = u32::from(p[c]) >> 1;
        if ((q[c] << 1) | (q[c] >> 6)) != u32::from(p[c]) {
            return None;
        }
    }
    let mut bits = 0u128;
    let mut offset = 0;
    put(&mut bits, &mut offset, 32, 6);
    put(&mut bits, &mut offset, 0, 2);
    for &v in &q {
        put(&mut bits, &mut offset, v, 7);
        put(&mut bits, &mut offset, v, 7);
    }
    put(&mut bits, &mut offset, u32::from(p[3]), 8);
    put(&mut bits, &mut offset, u32::from(p[3]), 8);
    for _ in 0..2 {
        for i in 0..16 {
            put(&mut bits, &mut offset, 0, if i == 0 { 1 } else { 2 });
        }
    }
    debug_assert_eq!(offset, 128);
    Some(bits.to_le_bytes())
}

// A bounded endpoint-neighborhood search for opaque grayscale blocks. Scores
// the actual 4-bit BC7 palette; no texture identity participates in decisions.
#[cfg(test)]
fn gray_mode6(pixels: &[[u8; 4]; 16], effort: Effort) -> ([u8; 16], u64) {
    debug_assert!(pixels
        .iter()
        .all(|p| p[0] == p[1] && p[1] == p[2] && p[3] == 255));
    let baseline = mode6(pixels, effort);
    if baseline.1 == 0 {
        return baseline;
    }
    let bits = u128::from_le_bytes(baseline.0);
    let mut a = (((bits >> 7) & 127) as i32) * 2 + 1;
    let mut b = (((bits >> 14) & 127) as i32) * 2 + 1;
    if a > b {
        core::mem::swap(&mut a, &mut b);
    }
    let radius = match effort {
        Effort::Fast => 2,
        Effort::Balanced => 3,
        Effort::Quality => 5,
    };
    let mut best = (baseline.1, a, b, [0usize; 16]);
    let mut found = false;
    for da in -radius..=radius {
        for db in -radius..=radius {
            let x = (a + 2 * da).clamp(1, 255);
            let y = (b + 2 * db).clamp(1, 255);
            if x > y {
                continue;
            }
            let mut palette = [0i32; 16];
            for (j, &w) in W.iter().enumerate() {
                palette[j] = ((64 - w) * x + w * y + 32) >> 6;
            }
            let mut indices = [0usize; 16];
            let mut error = 0u64;
            for (i, p) in pixels.iter().enumerate() {
                let v = i32::from(p[0]);
                let k = palette.partition_point(|&z| z < v);
                let mut local = (u32::MAX, 0usize);
                if k < 16 {
                    let d = v - palette[k];
                    local = local.min(((d * d) as u32, k));
                }
                if k > 0 {
                    let n = k - 1;
                    let target = palette[n];
                    let first = palette.partition_point(|&z| z < target);
                    let d = v - target;
                    local = local.min(((d * d) as u32, first));
                }
                error += 3 * u64::from(local.0);
                indices[i] = local.1;
                if error >= best.0 {
                    break;
                }
            }
            if error < best.0 {
                best = (error, x, y, indices);
                found = true;
            }
        }
    }
    if !found {
        return baseline;
    }
    let (error, mut x, mut y, mut indices) = best;
    if indices[0] >= 8 {
        core::mem::swap(&mut x, &mut y);
        for idx in &mut indices {
            *idx = 15 - *idx;
        }
    }
    let mut bits = 0u128;
    let mut offset = 0;
    put(&mut bits, &mut offset, 64, 7);
    for _ in 0..3 {
        put(&mut bits, &mut offset, (x >> 1) as u32, 7);
        put(&mut bits, &mut offset, (y >> 1) as u32, 7);
    }
    put(&mut bits, &mut offset, 127, 7);
    put(&mut bits, &mut offset, 127, 7);
    put(&mut bits, &mut offset, 1, 1);
    put(&mut bits, &mut offset, 1, 1);
    for (i, &idx) in indices.iter().enumerate() {
        put(
            &mut bits,
            &mut offset,
            idx as u32,
            if i == 0 { 3 } else { 4 },
        );
    }
    debug_assert_eq!(offset, 128);
    (bits.to_le_bytes(), error)
}

// Exact equivalent of partition_rank, reusing sums for each 4-pixel mask.
// A 16-pixel subset has sum <= 4080 and square sum <= 1,040,400;
// count * square and sum * sum are at most 16,646,400 per channel,
// and their four-channel sum is at most 66,585,600.
#[cfg(test)]
#[derive(Clone, Copy, Default)]
struct Moments {
    count: u32,
    sum: [u32; 4],
    square: [u32; 4],
}

const fn partition_masks(patterns: [u32; 64]) -> [[u16; 2]; 64] {
    let mut out = [[0u16; 2]; 64];
    let mut p = 0;
    while p < 64 {
        let mut i = 0;
        while i < 16 {
            let s = ((patterns[p] >> (i * 2)) & 3) as usize;
            if s > 0 {
                out[p][s - 1] |= 1 << i;
            }
            i += 1;
        }
        p += 1;
    }
    out
}
const MASKS2: [[u16; 2]; 64] = partition_masks(PART2);
const MASKS3: [[u16; 2]; 64] = partition_masks(PART3);

#[cfg(test)]
fn moment_lut(pixels: &[[u8; 4]; 16]) -> ([[Moments; 16]; 4], Moments) {
    let mut lut = [[Moments::default(); 16]; 4];
    for row in 0..4 {
        for mask in 1usize..16 {
            let i = mask.trailing_zeros() as usize;
            let mut m = lut[row][mask & (mask - 1)];
            m.count += 1;
            for (c, &value) in pixels[row * 4 + i].iter().enumerate() {
                let v = u32::from(value);
                m.sum[c] += v;
                m.square[c] += v * v;
            }
            lut[row][mask] = m;
        }
    }
    let mut total = Moments::default();
    for row in &lut {
        total.count += row[15].count;
        for c in 0..4 {
            total.sum[c] += row[15].sum[c];
            total.square[c] += row[15].square[c];
        }
    }
    (lut, total)
}
#[cfg(test)]
fn mask_moments(lut: &[[Moments; 16]; 4], mask: u16) -> Moments {
    let mut m = Moments::default();
    for (row, table) in lut.iter().enumerate() {
        let part = table[((mask >> (4 * row)) & 15) as usize];
        m.count += part.count;
        for c in 0..4 {
            m.sum[c] += part.sum[c];
            m.square[c] += part.square[c];
        }
    }
    m
}
#[cfg(test)]
fn rank_moments(
    lut: &[[Moments; 16]; 4],
    total: Moments,
    masks: [u16; 2],
    subsets: usize,
    channels: usize,
) -> u64 {
    let mut parts = [Moments::default(); 3];
    parts[0] = total;
    for s in 1..subsets {
        parts[s] = mask_moments(lut, masks[s - 1]);
        parts[0].count -= parts[s].count;
        for c in 0..4 {
            parts[0].sum[c] -= parts[s].sum[c];
            parts[0].square[c] -= parts[s].square[c];
        }
    }
    let mut score = 0;
    for p in parts.iter().take(subsets) {
        let mut sum = 0;
        let mut largest = 0;
        for c in 0..channels {
            let v = p.square[c] * p.count - p.sum[c] * p.sum[c];
            sum += v;
            largest = largest.max(v);
        }
        score += (sum - largest) / p.count;
    }
    score as u64
}

pub(crate) fn encode(pixels: &[[u8; 4]; 16], effort: Effort) -> [u8; 16] {
    match effort {
        Effort::Fast => fast::encode(pixels),
        Effort::Balanced => fast::encode_with(pixels, &fast::BALANCED),
        Effort::Quality => fast::encode_with(pixels, &fast::QUALITY),
    }
}

#[cfg(feature = "simd")]
pub(crate) fn batch_width() -> usize {
    fast::batch_width()
}
#[cfg(feature = "simd")]
pub(crate) fn encode_batch(blocks: &[[[u8; 4]; 16]; 8], out: &mut [u8], effort: Effort) {
    fast::encode_batch(blocks, out, effort);
}

#[cfg(feature = "simd")]
pub(crate) fn hdr_distances(src: &[[[f64; 8]; 16]; 3]) -> [[[f64; 16]; 16]; 8] {
    fast::simd::hdr_distances(src)
}

#[cfg(test)]
#[path = "../tests/bc7_decoder.rs"]
mod independent_decoder;

#[cfg(test)]
mod partition_tests {
    use super::*;
    #[test]
    fn partitioned_modes_score_their_decoded_blocks() {
        let mut state = 0x71b4862fu32;
        for sample in 0..100 {
            let mut pixels = [[0u8; 4]; 16];
            for pixel in &mut pixels {
                for component in pixel {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    *component = (state >> 24) as u8;
                }
            }
            for mode in [0, 1, 2, 3, 7] {
                if mode != 7 {
                    for pixel in &mut pixels {
                        pixel[3] = 255;
                    }
                } else if sample % 2 == 0 {
                    for pixel in &mut pixels {
                        pixel[3] = (state >> 16) as u8;
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    }
                }
                let (block, score) = partition_mode(&pixels, mode, Effort::Quality);
                assert_eq!(block[0].trailing_zeros() as usize, mode);
                let decoded = independent_decoder::decode(&block);
                let actual: u64 = pixels
                    .iter()
                    .zip(decoded)
                    .map(|(a, b)| {
                        (0..4)
                            .map(|c| {
                                let d = i32::from(a[c]) - i32::from(b[c]);
                                (d * d) as u64
                            })
                            .sum::<u64>()
                    })
                    .sum();
                assert_eq!(score, actual, "mode {mode}, sample {sample}");
            }
        }
    }
    #[test]
    fn exact_constant_rgba() {
        for v in 0..=255u8 {
            for p in [[v, v, v, v], [v, v, v, 255], [v, v, v, 0], [0, 0, 0, v]] {
                let block = solid_rgba(&[p; 16]).unwrap();
                assert_eq!(independent_decoder::decode(&block), [p; 16]);
                for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
                    assert_eq!(
                        independent_decoder::decode(&encode(&[p; 16], effort)),
                        [p; 16]
                    );
                }
            }
        }
        let mut state = 1234567u32;
        for _ in 0..10_000 {
            let mut p = [0u8; 4];
            for component in &mut p {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *component = (state >> 24) as u8;
            }
            if let Some(block) = solid_rgba(&[p; 16]) {
                assert_eq!(independent_decoder::decode(&block), [p; 16]);
            }
        }
    }

    #[test]
    fn gray_refinement_never_regresses_mode6() {
        let mut state = 1234567u32;
        for sample in 0..1000usize {
            let mut pixels = [[0u8; 4]; 16];
            for p in &mut pixels {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let v = if sample % 2 == 0 {
                    (state >> 24) as u8
                } else {
                    (sample % 245 + (state >> 29) as usize) as u8
                };
                *p = [v, v, v, 255];
            }
            let (block, error) = gray_mode6(&pixels, Effort::Balanced);
            let baseline = mode6(&pixels, Effort::Balanced);
            let decoded = independent_decoder::decode(&block);
            let actual: u64 = pixels
                .iter()
                .zip(decoded)
                .map(|(a, b)| {
                    (0..4)
                        .map(|c| {
                            let d = i32::from(a[c]) - i32::from(b[c]);
                            (d * d) as u64
                        })
                        .sum::<u64>()
                })
                .sum();
            assert_eq!(error, actual);
            assert!(error <= baseline.1);
        }
    }

    #[test]
    fn refinement_preserves_pbits_and_never_increases_score() {
        let mut state = 0x51a4_231fu32;
        for sample in 0..1000 {
            let mut pixels = [[0u8; 4]; 16];
            for pixel in &mut pixels {
                for value in pixel {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    *value = (state >> 24) as u8;
                }
            }
            if sample % 2 == 0 {
                for pixel in &mut pixels {
                    pixel[3] = 255;
                }
            }
            let mut endpoints = [[0i32; 4]; 2];
            for (k, endpoint) in endpoints.iter_mut().enumerate() {
                for (c, value) in endpoint.iter_mut().enumerate() {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    *value = if sample % 2 == 0 && c == 3 {
                        255
                    } else {
                        i32::from((state >> 25) as u8) * 2
                            + if sample % 2 == 0 { 1 } else { k as i32 }
                    };
                }
            }
            let (old, indices) = score(&pixels, endpoints);
            let (new, q, new_indices) = refine_mode6(&pixels, endpoints, indices, old);
            assert!(new <= old);
            assert_eq!((new, new_indices), score(&pixels, q));
            for k in 0..2 {
                for c in 0..4 {
                    assert_eq!(q[k][c] & 1, endpoints[k][c] & 1);
                }
            }
            if sample % 2 == 0 {
                assert_eq!(q[0][3], 255);
                assert_eq!(q[1][3], 255);
            }
        }
    }

    #[test]
    fn opaque_blocks_keep_alpha_exact() {
        let mut state = 0x546b_79c1u32;
        for sample in 0..128usize {
            let mut pixels = [[0u8; 4]; 16];
            for (i, pixel) in pixels.iter_mut().enumerate() {
                for (c, value) in pixel[..3].iter_mut().enumerate() {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let base = (sample * (c + 1) * 7) % 80;
                    let step = (sample % 7 + c + 1) * i;
                    *value = (base + step + (state >> 29) as usize).min(255) as u8;
                }
                pixel[3] = 255;
            }
            for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
                for block in [
                    mode6(&pixels, effort).0,
                    partition_mode(&pixels, 7, effort).0,
                    encode(&pixels, effort),
                ] {
                    let decoded = independent_decoder::decode(&block);
                    assert!(
                        decoded.iter().all(|p| p[3] == 255),
                        "sample {sample}, mode {}",
                        block[0].trailing_zeros()
                    );
                }
            }
        }
    }

    #[test]
    fn existing_modes_score_their_decoded_blocks() {
        let mut state = 0x1653_278bu32;
        for sample in 0..100 {
            let mut pixels = [[0u8; 4]; 16];
            for pixel in &mut pixels {
                for component in pixel {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    *component = (state >> 24) as u8;
                }
            }
            let mut candidates = [([0u8; 16], 0u64); 5];
            candidates[0] = mode6(&pixels, Effort::Quality);
            for (rotation, candidate) in candidates[1..].iter_mut().enumerate() {
                *candidate = mode5(&pixels, rotation);
            }
            for (block, score) in candidates {
                let decoded = independent_decoder::decode(&block);
                let actual: u64 = pixels
                    .iter()
                    .zip(decoded)
                    .map(|(a, b)| {
                        (0..4)
                            .map(|c| {
                                let d = i32::from(a[c]) - i32::from(b[c]);
                                (d * d) as u64
                            })
                            .sum::<u64>()
                    })
                    .sum();
                assert_eq!(
                    score,
                    actual,
                    "sample {sample}, mode {}",
                    block[0].trailing_zeros()
                );
            }
        }
    }

    #[test]
    fn mode4_scores_each_rotation_and_index_selection() {
        let mut state = 0x12345678u32;
        for sample in 0..100 {
            let mut pixels = [[0u8; 4]; 16];
            for pixel in &mut pixels {
                for component in pixel {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    *component = (state >> 24) as u8;
                }
            }
            for rotation in 0..4 {
                for selection in 0..2 {
                    let (block, score) = mode4(&pixels, rotation, selection);
                    let decoded = independent_decoder::decode(&block);
                    let actual: u64 = pixels
                        .iter()
                        .zip(decoded)
                        .map(|(a, b)| {
                            (0..4)
                                .map(|c| {
                                    let d = i32::from(a[c]) - i32::from(b[c]);
                                    (d * d) as u64
                                })
                                .sum::<u64>()
                        })
                        .sum();
                    assert_eq!(
                        score, actual,
                        "sample {sample}, rotation {rotation}, selection {selection}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod rank_reuse_tests {
    use super::*;
    #[test]
    fn reused_ranks_preserve_partition_mode_bytes() {
        let mut state = 0x937a_2134u32;
        for _ in 0..100 {
            let mut pixels = [[0u8; 4]; 16];
            for p in &mut pixels {
                for c in p {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    *c = (state >> 24) as u8;
                }
            }
            let (lut, total) = moment_lut(&pixels);
            for mode in [0, 1, 2, 3, 7] {
                let subsets = if mode == 0 || mode == 2 { 3 } else { 2 };
                let channels = if mode == 7 { 4 } else { 3 };
                let mut ranks = [(u64::MAX, 0usize); 64];
                for (p, r) in ranks
                    .iter_mut()
                    .enumerate()
                    .take(if mode == 0 { 16 } else { 64 })
                {
                    *r = (
                        rank_moments(
                            &lut,
                            total,
                            if subsets == 3 { MASKS3[p] } else { MASKS2[p] },
                            subsets,
                            channels,
                        ),
                        p,
                    );
                }
                ranks.sort_unstable();
                for effort in [Effort::Fast, Effort::Balanced, Effort::Quality] {
                    assert_eq!(
                        partition_mode(&pixels, mode, effort),
                        partition_mode_ranked(&pixels, mode, effort, &ranks),
                        "mode {mode}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod rank_moments_tests {
    use super::*;
    #[test]
    fn moment_rank_matches_original_every_partition() {
        let mut state = 0x7a5b_9231u32;
        for sample in 0..1000 {
            let mut pixels = [[0u8; 4]; 16];
            for p in &mut pixels {
                for c in p {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    *c = (state >> 24) as u8;
                }
            }
            let (lut, total) = moment_lut(&pixels);
            for p in 0..64 {
                for channels in [3, 4] {
                    assert_eq!(
                        rank_moments(&lut, total, MASKS2[p], 2, channels),
                        partition_rank(&pixels, PART2[p], channels, 2),
                        "sample {sample} P2 {p}"
                    );
                    assert_eq!(
                        rank_moments(&lut, total, MASKS3[p], 3, channels),
                        partition_rank(&pixels, PART3[p], channels, 3),
                        "sample {sample} P3 {p}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod lower_bound_tests {
    use super::*;
    #[test]
    fn lower_bound_does_not_exceed_actual_decoded_error() {
        let mut state = 0x8123_4823u32;
        for sample in 0..1000 {
            let mut pixels = [[0u8; 4]; 16];
            for p in &mut pixels {
                for v in p {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    *v = (state >> 24) as u8;
                }
            }
            for mode in [0, 1, 2, 3, 7] {
                let mut pixels = pixels;
                if mode != 7 {
                    for p in &mut pixels {
                        p[3] = 255;
                    }
                }
                let (block, score) = partition_mode(&pixels, mode, Effort::Fast);
                let bits = u128::from_le_bytes(block);
                let partition = ((bits >> (mode + 1)) & if mode == 0 { 15 } else { 63 }) as usize;
                let subsets = if mode == 0 || mode == 2 { 3 } else { 2 };
                let channels = if mode == 7 { 4 } else { 3 };
                let pattern = if subsets == 3 {
                    PART3[partition]
                } else {
                    PART2[partition]
                };
                let bound = partition_lower_bound(&pixels, pattern, subsets, channels);
                assert!(
                    bound <= score,
                    "sample {sample}, mode {mode}, bound {bound}, score {score}"
                );
            }
        }
    }
}

#[cfg(test)]
mod rank_u32_edge_tests {
    use super::*;
    #[test]
    fn rank_u32_extreme_blocks_match_original() {
        for kind in 0..5 {
            let mut pixels = [[0u8; 4]; 16];
            for (i, p) in pixels.iter_mut().enumerate() {
                for (c, v) in p.iter_mut().enumerate() {
                    *v = match kind {
                        0 => 0,
                        1 => 255,
                        2 => {
                            if i % 2 == 0 {
                                0
                            } else {
                                255
                            }
                        }
                        3 => {
                            if (i + c) % 2 == 0 {
                                255
                            } else {
                                0
                            }
                        }
                        _ => {
                            if i < 15 {
                                255
                            } else {
                                0
                            }
                        }
                    };
                }
            }
            let (lut, total) = moment_lut(&pixels);
            for part in 0..64 {
                for channels in [3, 4] {
                    assert_eq!(
                        rank_moments(&lut, total, MASKS2[part], 2, channels),
                        partition_rank(&pixels, PART2[part], channels, 2)
                    );
                    assert_eq!(
                        rank_moments(&lut, total, MASKS3[part], 3, channels),
                        partition_rank(&pixels, PART3[part], channels, 3)
                    );
                }
            }
        }
    }
}
