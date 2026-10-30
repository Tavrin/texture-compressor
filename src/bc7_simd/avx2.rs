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
kernels!("avx2", 8);
