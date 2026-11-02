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
#[target_feature(enable = "simd128")]
fn to_int(a: v128) -> v128 {
    i32x4_trunc_sat_f32x4(a)
}
#[target_feature(enable = "simd128")]
fn gt(a: v128, b: v128) -> v128 {
    i32x4_gt(a, b)
}
#[target_feature(enable = "simd128")]
fn and(a: v128, b: v128) -> v128 {
    v128_and(a, b)
}
#[target_feature(enable = "simd128")]
fn or(a: v128, b: v128) -> v128 {
    v128_or(a, b)
}
// Elements of `a` where `mask` is set, else elements of `b`.
#[target_feature(enable = "simd128")]
fn blend(mask: v128, a: v128, b: v128) -> v128 {
    v128_bitselect(a, b, mask)
}
#[target_feature(enable = "simd128")]
fn vmin(a: v128, b: v128) -> v128 {
    i32x4_min(a, b)
}
#[target_feature(enable = "simd128")]
fn vmax(a: v128, b: v128) -> v128 {
    i32x4_max(a, b)
}
// A 16-byte lookup table.
type Table = v128;
#[target_feature(enable = "simd128")]
fn bytes(a: &[u8; 16]) -> v128 {
    unsafe { v128_load(a.as_ptr().cast()) }
}
#[target_feature(enable = "simd128")]
fn quad(a: &[[u8; 4]; 4]) -> v128 {
    unsafe { v128_load(a.as_ptr().cast()) }
}
// Byte t[i] for each element index i (0..16; larger indices read zero).
#[target_feature(enable = "simd128")]
fn lookup(t: v128, i: v128) -> v128 {
    i8x16_swizzle(t, v128_or(i, i32x4_splat(0x8080_8000u32 as i32)))
}
// Channel planes: table c holds channel c of texels 0..16.
#[target_feature(enable = "simd128")]
fn planes(px: &Block) -> [v128; 4] {
    let (quads, _) = px.as_chunks::<4>();
    let s: [_; 4] = core::array::from_fn(|q| {
        let v = quad(&quads[q]);
        i8x16_shuffle::<0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15>(v, v)
    });
    let t0 = i32x4_shuffle::<0, 4, 1, 5>(s[0], s[1]);
    let t1 = i32x4_shuffle::<0, 4, 1, 5>(s[2], s[3]);
    let t2 = i32x4_shuffle::<2, 6, 3, 7>(s[0], s[1]);
    let t3 = i32x4_shuffle::<2, 6, 3, 7>(s[2], s[3]);
    [
        i64x2_shuffle::<0, 2>(t0, t1),
        i64x2_shuffle::<1, 3>(t0, t1),
        i64x2_shuffle::<0, 2>(t2, t3),
        i64x2_shuffle::<1, 3>(t2, t3),
    ]
}
// Sixteen element values in 0..=255 as a byte table, in element order.
#[target_feature(enable = "simd128")]
fn pack(e: [v128; 4]) -> v128 {
    u8x16_narrow_i16x8(
        i16x8_narrow_i32x4(e[0], e[1]),
        i16x8_narrow_i32x4(e[2], e[3]),
    )
}
kernels!("simd128", 4);
