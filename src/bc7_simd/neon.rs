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
kernels!("neon", 4);
