use super::*;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[target_feature(enable = "sse4.1")]
fn load(a: &[i32; 4]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}
#[target_feature(enable = "sse4.1")]
fn store(v: __m128i) -> [i32; 4] {
    let mut a = [0; 4];
    unsafe {
        _mm_storeu_si128(a.as_mut_ptr().cast(), v);
    }
    a
}
#[target_feature(enable = "sse4.1")]
fn fload(a: &[f32; 4]) -> __m128 {
    unsafe { _mm_loadu_ps(a.as_ptr()) }
}
#[target_feature(enable = "sse4.1")]
fn fstore(v: __m128) -> [f32; 4] {
    let mut a = [0.0; 4];
    unsafe {
        _mm_storeu_ps(a.as_mut_ptr(), v);
    }
    a
}
#[target_feature(enable = "sse4.1")]
fn add(a: __m128i, b: __m128i) -> __m128i {
    _mm_add_epi32(a, b)
}
#[target_feature(enable = "sse4.1")]
fn sub(a: __m128i, b: __m128i) -> __m128i {
    _mm_sub_epi32(a, b)
}
#[target_feature(enable = "sse4.1")]
fn mul(a: __m128i, b: __m128i) -> __m128i {
    _mm_mullo_epi32(a, b)
}
#[target_feature(enable = "sse4.1")]
fn fadd(a: __m128, b: __m128) -> __m128 {
    _mm_add_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
fn fsub(a: __m128, b: __m128) -> __m128 {
    _mm_sub_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
fn fmul(a: __m128, b: __m128) -> __m128 {
    _mm_mul_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
fn fdiv(a: __m128, b: __m128) -> __m128 {
    _mm_div_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
fn splat(a: i32) -> __m128i {
    _mm_set1_epi32(a)
}
#[target_feature(enable = "sse4.1")]
fn fsplat(a: f32) -> __m128 {
    _mm_set1_ps(a)
}
#[target_feature(enable = "sse4.1")]
fn shr6(a: __m128i) -> __m128i {
    _mm_srai_epi32::<6>(a)
}
#[target_feature(enable = "sse4.1")]
fn to_float(a: __m128i) -> __m128 {
    _mm_cvtepi32_ps(a)
}
#[target_feature(enable = "sse4.1")]
fn to_int(a: __m128) -> __m128i {
    _mm_cvttps_epi32(a)
}
#[target_feature(enable = "sse4.1")]
fn gt(a: __m128i, b: __m128i) -> __m128i {
    _mm_cmpgt_epi32(a, b)
}
#[target_feature(enable = "sse4.1")]
fn and(a: __m128i, b: __m128i) -> __m128i {
    _mm_and_si128(a, b)
}
#[target_feature(enable = "sse4.1")]
fn or(a: __m128i, b: __m128i) -> __m128i {
    _mm_or_si128(a, b)
}
// Elements of `a` where `mask` is set, else elements of `b`.
#[target_feature(enable = "sse4.1")]
fn blend(mask: __m128i, a: __m128i, b: __m128i) -> __m128i {
    _mm_blendv_epi8(b, a, mask)
}
#[target_feature(enable = "sse4.1")]
fn vmin(a: __m128i, b: __m128i) -> __m128i {
    _mm_min_epi32(a, b)
}
#[target_feature(enable = "sse4.1")]
fn vmax(a: __m128i, b: __m128i) -> __m128i {
    _mm_max_epi32(a, b)
}
// A 16-byte lookup table.
type Table = __m128i;
#[target_feature(enable = "sse4.1")]
fn bytes(a: &[u8; 16]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}
#[target_feature(enable = "sse4.1")]
fn quad(a: &[[u8; 4]; 4]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}
// Byte t[i] for each element index i (0..16; larger indices read some entry).
#[target_feature(enable = "sse4.1")]
fn lookup(t: __m128i, i: __m128i) -> __m128i {
    _mm_shuffle_epi8(t, _mm_or_si128(i, _mm_set1_epi32(0x8080_8000u32 as i32)))
}
// Channel planes: table c holds channel c of texels 0..16.
#[target_feature(enable = "sse4.1")]
fn planes(px: &Block) -> [__m128i; 4] {
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
}
// Sixteen element values in 0..=255 as a byte table, in element order.
#[target_feature(enable = "sse4.1")]
fn pack(e: [__m128i; 4]) -> __m128i {
    _mm_packus_epi16(_mm_packus_epi32(e[0], e[1]), _mm_packus_epi32(e[2], e[3]))
}
kernels!("sse4.1", 4);

// Each vector spans independent HDR blocks. No horizontal reductions or FMA:
// the RGB accumulation order is exactly the scalar f64 order.
#[target_feature(enable = "sse4.1")]
pub(super) fn hdr_distances(src: &[[[f64; 8]; 16]; 3]) -> [[[f64; 16]; 16]; 8] {
    let mut out = [[[0.0; 16]; 16]; 8];
    for a in 0..16 {
        for b in a..16 {
            for base in (0..8).step_by(2) {
                let mut d = _mm_setzero_pd();
                for channel in src {
                    // base+2 <= 8, and unaligned loads are permitted.
                    let x = unsafe { _mm_loadu_pd(channel[a][base..].as_ptr()) };
                    let y = unsafe { _mm_loadu_pd(channel[b][base..].as_ptr()) };
                    let v = _mm_sub_pd(x, y);
                    d = _mm_add_pd(d, _mm_mul_pd(v, v));
                }
                let mut values = [0.0; 2];
                unsafe {
                    _mm_storeu_pd(values.as_mut_ptr(), d);
                }
                for l in 0..2 {
                    out[base + l][a][b] = values[l];
                }
            }
        }
    }
    out
}
