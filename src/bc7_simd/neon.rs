use super::*;
use core::arch::aarch64::*;
#[target_feature(enable = "neon")]
fn load(a: &[i32; 4]) -> int32x4_t {
    unsafe { vld1q_s32(a.as_ptr()) }
}
#[target_feature(enable = "neon")]
fn store(v: int32x4_t) -> [i32; 4] {
    let mut a = [0; 4];
    unsafe {
        vst1q_s32(a.as_mut_ptr(), v);
    }
    a
}
#[target_feature(enable = "neon")]
fn fload(a: &[f32; 4]) -> float32x4_t {
    unsafe { vld1q_f32(a.as_ptr()) }
}
#[target_feature(enable = "neon")]
fn fstore(v: float32x4_t) -> [f32; 4] {
    let mut a = [0.0; 4];
    unsafe {
        vst1q_f32(a.as_mut_ptr(), v);
    }
    a
}
#[target_feature(enable = "neon")]
fn add(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vaddq_s32(a, b)
}
#[target_feature(enable = "neon")]
fn sub(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vsubq_s32(a, b)
}
#[target_feature(enable = "neon")]
fn mul(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vmulq_s32(a, b)
}
#[target_feature(enable = "neon")]
fn fadd(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vaddq_f32(a, b)
}
#[target_feature(enable = "neon")]
fn fsub(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vsubq_f32(a, b)
}
#[target_feature(enable = "neon")]
fn fmul(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vmulq_f32(a, b)
}
#[target_feature(enable = "neon")]
fn fdiv(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vdivq_f32(a, b)
}
#[target_feature(enable = "neon")]
fn splat(a: i32) -> int32x4_t {
    vdupq_n_s32(a)
}
#[target_feature(enable = "neon")]
fn fsplat(a: f32) -> float32x4_t {
    vdupq_n_f32(a)
}
#[target_feature(enable = "neon")]
fn shr6(a: int32x4_t) -> int32x4_t {
    vshrq_n_s32::<6>(a)
}
#[target_feature(enable = "neon")]
fn to_float(a: int32x4_t) -> float32x4_t {
    vcvtq_f32_s32(a)
}
// Truncates toward zero, as `as i32` does for in-range values.
#[target_feature(enable = "neon")]
fn to_int(a: float32x4_t) -> int32x4_t {
    vcvtq_s32_f32(a)
}
#[target_feature(enable = "neon")]
fn gt(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vreinterpretq_s32_u32(vcgtq_s32(a, b))
}
#[target_feature(enable = "neon")]
fn and(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vandq_s32(a, b)
}
#[target_feature(enable = "neon")]
fn or(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vorrq_s32(a, b)
}
// Elements of `a` where `mask` is set, else elements of `b`.
#[target_feature(enable = "neon")]
fn blend(mask: int32x4_t, a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vbslq_s32(vreinterpretq_u32_s32(mask), a, b)
}
#[target_feature(enable = "neon")]
fn vmin(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vminq_s32(a, b)
}
#[target_feature(enable = "neon")]
fn vmax(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vmaxq_s32(a, b)
}
// A 16-byte lookup table.
type Table = uint8x16_t;
#[target_feature(enable = "neon")]
fn bytes(a: &[u8; 16]) -> uint8x16_t {
    unsafe { vld1q_u8(a.as_ptr()) }
}
// Byte t[i] for each element index i (0..16; larger indices read zero).
#[target_feature(enable = "neon")]
fn lookup(t: uint8x16_t, i: int32x4_t) -> int32x4_t {
    let i = vorrq_s32(i, vdupq_n_s32(0x8080_8000u32 as i32));
    vreinterpretq_s32_u8(vqtbl1q_u8(t, vreinterpretq_u8_s32(i)))
}
// Channel planes: table c holds channel c of texels 0..16 (a 4-way
// de-interleaving load of exactly the block's 64 bytes).
#[target_feature(enable = "neon")]
fn planes(px: &Block) -> [uint8x16_t; 4] {
    let v = unsafe { vld4q_u8(px.as_ptr().cast()) };
    [v.0, v.1, v.2, v.3]
}
// Sixteen element values in 0..=255 as a byte table, in element order.
#[target_feature(enable = "neon")]
fn pack(e: [int32x4_t; 4]) -> uint8x16_t {
    let low = vcombine_u16(vqmovun_s32(e[0]), vqmovun_s32(e[1]));
    let high = vcombine_u16(vqmovun_s32(e[2]), vqmovun_s32(e[3]));
    vcombine_u8(vqmovn_u16(low), vqmovn_u16(high))
}
kernels!("neon", 4);
