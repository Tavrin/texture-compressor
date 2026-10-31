//! Arbitrary 16-byte blocks through the BC6H decoder, both signs.
#![no_main]
use libfuzzer_sys::fuzz_target;
use texture_compressor::{decode_bc6h, Format};

fuzz_target!(|data: &[u8]| {
    let Ok(block) = <&[u8; 16]>::try_from(data.get(..16).unwrap_or(&[])) else {
        return;
    };
    for format in [Format::Bc6hUf16, Format::Bc6hSf16] {
        let a = decode_bc6h(block, format).unwrap();
        assert_eq!(a, decode_bc6h(block, format).unwrap());
        if format == Format::Bc6hUf16 {
            // Unsigned output never has the binary16 sign bit set.
            assert!(a.iter().flatten().all(|&v| v & 0x8000 == 0));
        }
    }
});
