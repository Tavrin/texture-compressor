use super::*;
use core::arch::wasm32::*;
#[target_feature(enable = "simd128")]
fn load(a: &[i32; 4]) -> v128 {
    unsafe { v128_load(a.as_ptr().cast()) }
}
#[target_feature(enable = "simd128")]
fn store(v: v128) -> [i32; 4] {
    let mut a = [0; 4];
    unsafe {
        v128_store(a.as_mut_ptr().cast(), v);
    }
    a
}
#[target_feature(enable = "simd128")]
fn fload(a: &[f32; 4]) -> v128 {
    unsafe { v128_load(a.as_ptr().cast()) }
}
#[target_feature(enable = "simd128")]
fn fstore(v: v128) -> [f32; 4] {
    let mut a = [0.0; 4];
    unsafe {
        v128_store(a.as_mut_ptr().cast(), v);
    }
    a
}
#[target_feature(enable = "simd128")]
fn add(a: v128, b: v128) -> v128 {
    i32x4_add(a, b)
}
#[target_feature(enable = "simd128")]
fn sub(a: v128, b: v128) -> v128 {
    i32x4_sub(a, b)
}
#[target_feature(enable = "simd128")]
fn mul(a: v128, b: v128) -> v128 {
    i32x4_mul(a, b)
}
#[target_feature(enable = "simd128")]
fn fadd(a: v128, b: v128) -> v128 {
    f32x4_add(a, b)
}
#[target_feature(enable = "simd128")]
fn fsub(a: v128, b: v128) -> v128 {
    f32x4_sub(a, b)
}
#[target_feature(enable = "simd128")]
fn fmul(a: v128, b: v128) -> v128 {
    f32x4_mul(a, b)
}
#[target_feature(enable = "simd128")]
fn fdiv(a: v128, b: v128) -> v128 {
    f32x4_div(a, b)
}
#[target_feature(enable = "simd128")]
fn splat(a: i32) -> v128 {
    i32x4_splat(a)
}
#[target_feature(enable = "simd128")]
fn fsplat(a: f32) -> v128 {
    f32x4_splat(a)
}
#[target_feature(enable = "simd128")]
fn shr6(a: v128) -> v128 {
    i32x4_shr(a, 6)
}
#[target_feature(enable = "simd128")]
fn to_float(a: v128) -> v128 {
    f32x4_convert_i32x4(a)
}
kernels!("simd128", 4);
