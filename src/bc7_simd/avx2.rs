use super::*;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[target_feature(enable = "avx2")]
fn load(a: &[i32; 8]) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr().cast()) }
}
#[target_feature(enable = "avx2")]
fn store(v: __m256i) -> [i32; 8] {
    let mut a = [0; 8];
    unsafe {
        _mm256_storeu_si256(a.as_mut_ptr().cast(), v);
    }
    a
}
#[target_feature(enable = "avx2")]
fn fload(a: &[f32; 8]) -> __m256 {
    unsafe { _mm256_loadu_ps(a.as_ptr()) }
}
#[target_feature(enable = "avx2")]
fn fstore(v: __m256) -> [f32; 8] {
    let mut a = [0.0; 8];
    unsafe {
        _mm256_storeu_ps(a.as_mut_ptr(), v);
    }
    a
}
#[target_feature(enable = "avx2")]
fn add(a: __m256i, b: __m256i) -> __m256i {
    _mm256_add_epi32(a, b)
}
#[target_feature(enable = "avx2")]
fn sub(a: __m256i, b: __m256i) -> __m256i {
    _mm256_sub_epi32(a, b)
}
#[target_feature(enable = "avx2")]
fn mul(a: __m256i, b: __m256i) -> __m256i {
    _mm256_mullo_epi32(a, b)
}
#[target_feature(enable = "avx2")]
fn fadd(a: __m256, b: __m256) -> __m256 {
    _mm256_add_ps(a, b)
}
#[target_feature(enable = "avx2")]
fn fsub(a: __m256, b: __m256) -> __m256 {
    _mm256_sub_ps(a, b)
}
#[target_feature(enable = "avx2")]
fn fmul(a: __m256, b: __m256) -> __m256 {
    _mm256_mul_ps(a, b)
}
#[target_feature(enable = "avx2")]
fn fdiv(a: __m256, b: __m256) -> __m256 {
    _mm256_div_ps(a, b)
}
#[target_feature(enable = "avx2")]
fn splat(a: i32) -> __m256i {
    _mm256_set1_epi32(a)
}
#[target_feature(enable = "avx2")]
fn fsplat(a: f32) -> __m256 {
    _mm256_set1_ps(a)
}
#[target_feature(enable = "avx2")]
fn shr6(a: __m256i) -> __m256i {
    _mm256_srai_epi32::<6>(a)
}
#[target_feature(enable = "avx2")]
fn to_float(a: __m256i) -> __m256 {
    _mm256_cvtepi32_ps(a)
}
#[target_feature(enable = "avx2")]
fn to_int(a: __m256) -> __m256i {
    _mm256_cvttps_epi32(a)
}
#[target_feature(enable = "avx2")]
fn gt(a: __m256i, b: __m256i) -> __m256i {
    _mm256_cmpgt_epi32(a, b)
}
#[target_feature(enable = "avx2")]
fn and(a: __m256i, b: __m256i) -> __m256i {
    _mm256_and_si256(a, b)
}
#[target_feature(enable = "avx2")]
fn or(a: __m256i, b: __m256i) -> __m256i {
    _mm256_or_si256(a, b)
}
// Elements of `a` where `mask` is set, else elements of `b`.
#[target_feature(enable = "avx2")]
fn blend(mask: __m256i, a: __m256i, b: __m256i) -> __m256i {
    _mm256_blendv_epi8(b, a, mask)
}
#[target_feature(enable = "avx2")]
fn vmin(a: __m256i, b: __m256i) -> __m256i {
    _mm256_min_epi32(a, b)
}
#[target_feature(enable = "avx2")]
fn vmax(a: __m256i, b: __m256i) -> __m256i {
    _mm256_max_epi32(a, b)
}
// A 16-byte lookup table, repeated in both 128-bit halves.
type Table = __m256i;
#[target_feature(enable = "avx2")]
fn bytes(a: &[u8; 16]) -> __m256i {
    _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(a.as_ptr().cast()) })
}
#[target_feature(enable = "avx2")]
fn quad(a: &[[u8; 4]; 4]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}
// Byte t[i] for each element index i (0..16; larger indices read some entry).
#[target_feature(enable = "avx2")]
fn lookup(t: __m256i, i: __m256i) -> __m256i {
    _mm256_shuffle_epi8(
        t,
        _mm256_or_si256(i, _mm256_set1_epi32(0x8080_8000u32 as i32)),
    )
}
// Channel planes: table c holds channel c of texels 0..16.
#[target_feature(enable = "avx2")]
fn planes(px: &Block) -> [__m256i; 4] {
    let (quads, _) = px.as_chunks::<4>();
    let order = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    let s: [_; 4] = core::array::from_fn(|q| _mm_shuffle_epi8(quad(&quads[q]), order));
    let t0 = _mm_unpacklo_epi32(s[0], s[1]);
    let t1 = _mm_unpacklo_epi32(s[2], s[3]);
    let t2 = _mm_unpackhi_epi32(s[0], s[1]);
    let t3 = _mm_unpackhi_epi32(s[2], s[3]);
    [
        _mm_unpacklo_epi64(t0, t1),
        _mm_unpackhi_epi64(t0, t1),
        _mm_unpacklo_epi64(t2, t3),
        _mm_unpackhi_epi64(t2, t3),
    ]
    .map(|p| _mm256_broadcastsi128_si256(p))
}
// Sixteen element values in 0..=255 as a byte table, in element order.
#[target_feature(enable = "avx2")]
fn pack(e: [__m256i; 2]) -> __m256i {
    let low = _mm_packus_epi32(
        _mm256_castsi256_si128(e[0]),
        _mm256_extracti128_si256::<1>(e[0]),
    );
    let high = _mm_packus_epi32(
        _mm256_castsi256_si128(e[1]),
        _mm256_extracti128_si256::<1>(e[1]),
    );
    _mm256_broadcastsi128_si256(_mm_packus_epi16(low, high))
}
kernels!("avx2", 8);

// Each vector spans independent HDR blocks. No horizontal reductions or FMA:
// the RGB accumulation order is exactly the scalar f64 order.
#[target_feature(enable = "avx2")]
pub(super) fn hdr_distances(src: &[[[f64; 8]; 16]; 3]) -> [[[f64; 16]; 16]; 8] {
    let mut out = [[[0.0; 16]; 16]; 8];
    for a in 0..16 {
        for b in a..16 {
            for base in (0..8).step_by(4) {
                let mut d = _mm256_setzero_pd();
                for channel in src {
                    // base+4 <= 8, and unaligned loads are permitted.
                    let x = unsafe { _mm256_loadu_pd(channel[a][base..].as_ptr()) };
                    let y = unsafe { _mm256_loadu_pd(channel[b][base..].as_ptr()) };
                    let v = _mm256_sub_pd(x, y);
                    d = _mm256_add_pd(d, _mm256_mul_pd(v, v));
                }
                let mut values = [0.0; 4];
                unsafe {
                    _mm256_storeu_pd(values.as_mut_ptr(), d);
                }
                for l in 0..4 {
                    out[base + l][a][b] = values[l];
                }
            }
        }
    }
    out
}
