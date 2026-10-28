# texture-compressor

Encode decoded pixels into GPU block-compressed data (BC1, BC3, BC4, BC5 and experimental BC7). The crate does not decode image files, generate mipmaps, or write containers. BC7 has not yet met its full quality and performance qualification targets.

```rust
use texture_compressor::{encode, Effort, Format, ImageView, Settings};

let pixels = [255u8; 4 * 4 * 4];
let image = ImageView::rgba8(&pixels, 4, 4, 16).unwrap();
let blocks = encode(image, &Settings::new(Format::Bc1, Effort::Fast)).unwrap();
assert_eq!(blocks.len(), 8);
```

R8 and RG8 input views are also supported. The optional `parallel` feature enables parallel encoding.
