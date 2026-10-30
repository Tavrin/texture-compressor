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
kernels!("sse4.1", 4);
