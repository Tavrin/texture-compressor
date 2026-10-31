use std::{env, fs::File, io::Write, path::PathBuf};

// For each decoded channel, selector, ordering relation, and source byte,
// choose the same (squared error, endpoint 0, endpoint 1) minimum as the
// original solid search. Relations are x < y, x == y, and x > y.
fn main() {
    // intel_tex_2's bundled Linux archive references the C++ unwind runtime.
    // Keep this qualification-only linkage confined to example binaries.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-examples=-lstdc++");
    }
    // Exact decoded-half objective values, generated with the same software
    // logarithm as the source objective. Both targets embed identical bits.
    let mut hdr =
        File::create(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("hdr_log.rs")).unwrap();
    writeln!(hdr, "static HALF_LOG: [f64; 31744] = [").unwrap();
    for h in 0..31744u32 {
        let e = h >> 10;
        let m = h & 1023;
        let x = if e == 0 {
            m as f32 / 16777216.0
        } else {
            f32::from_bits(((e + 112) << 23) | (m << 13))
        };
        writeln!(
            hdr,
            "f64::from_bits({}),",
            libm::log2(1.0 + 16.0 * f64::from(x)).to_bits()
        )
        .unwrap();
    }
    writeln!(hdr, "]; ").unwrap();
    println!("cargo:rerun-if-changed=build.rs");
    let mut out =
        File::create(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("solid.rs")).unwrap();
    writeln!(out, "static SOLID: SolidTable = [").unwrap();
    for three in [false, true] {
        writeln!(out, "[").unwrap();
        for selector in 0..4 {
            writeln!(out, "[").unwrap();
            for green in [false, true] {
                let limit = if green { 63 } else { 31 };
                let mut entries = [[(i32::MAX, 0, 0); 3]; 256];
                for x in 0..=limit {
                    for y in 0..=limit {
                        let rel = if x < y {
                            0
                        } else if x == y {
                            1
                        } else {
                            2
                        };
                        let (mul, add) = if green { (259, 33) } else { (527, 23) };
                        let decoded = match selector {
                            0 => (x * mul + add) >> 6,
                            1 => (y * mul + add) >> 6,
                            2 if three => {
                                if green {
                                    ((x + y) * 4145 + 1019) >> 11
                                } else {
                                    ((x + y) * 1053 + 125) >> 8
                                }
                            }
                            2 => {
                                if green {
                                    ((2 * x + y) * 2763 + 1039) >> 11
                                } else {
                                    ((2 * x + y) * 351 + 61) >> 7
                                }
                            }
                            _ => {
                                if green {
                                    ((x + 2 * y) * 2763 + 1039) >> 11
                                } else {
                                    ((x + 2 * y) * 351 + 61) >> 7
                                }
                            }
                        };
                        for (v, entry) in entries.iter_mut().enumerate() {
                            let d = decoded - v as i32;
                            entry[rel] = entry[rel].min((d * d, x, y));
                        }
                    }
                }
                writeln!(out, "{:?},", entries).unwrap();
            }
            writeln!(out, "],").unwrap();
        }
        writeln!(out, "],").unwrap();
    }
    writeln!(out, "]; ").unwrap();
}
