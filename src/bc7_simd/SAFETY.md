# BC7 SIMD safety audit

Only this private module permits unsafe code. The rest of the crate denies
unsafe code; disabling `simd` forbids it for the complete library. No vector
or raw pointer escapes the module. No new dependency is required.

Dispatch is immutable after detection, cached in an AtomicU8 with relaxed
ordering (there is no dependent memory to publish). x86 with `std` checks
`is_x86_feature_detected!`; without `std` it uses only compile-time features.
AArch64 requires compile-time NEON, Wasm requires compile-time simd128.
Unsupported targets fall back to scalar. The optional `simd-diagnostics`
feature accepts a process environment override once; it rejects unsupported
ISAs, and can only select scalar or a supported backend. Workers share the
same selection. No global mutable override can race a running kernel.

The three unsafe expressions in the dispatch macro call target-feature
functions only after the exact feature has been established. Wasm simd128
calls are safe under its required compile-time target feature. Each of the
four backend files has four unsafe memory expressions: integer and float
load/store helpers. Their arguments/results are fully initialized fixed
arrays of exactly 4 or 8 elements. Unaligned loads/stores permit the arrays'
natural alignment. Stores use exclusive local arrays, loads use shared
borrows, and pointers are neither offset nor retained. Arithmetic intrinsics
use values only inside functions with the corresponding target feature.
There is no unchecked indexing, transmute, union, raw-pointer arithmetic,
uninitialized memory, FFI, allocator manipulation or unsafe synchronization.
Inventory: 19 source unsafe blocks, all in this directory (16 array memory
blocks, 3 checked dispatch branches). The kernel macro itself has none.

Search control, candidate ordering and packing remain scalar. Mode-6 seeds (moments/PCA, palette/error and one LS refinement)
batch 4/8 blocks in texel/channel/block SoA; short image batches stage zeroed
inactive blocks and scatter only live outputs. Per-block alpha policy chooses
whether a seed is consumed. Seed selector evaluation visits all 16 texels;
integer SSE equals multiplicity-weighted representatives exactly. The LS
step uses the original distinct representatives and scalar i64/f64 arithmetic.
Its full-error evaluation is equivalent to scalar cutoffs: a cutoff can only
reject a candidate, and every improving candidate completes all selectors. Palette errors vectorize independent candidate
entries, then visit valid selectors in ascending order with strict `<` ties.
Cutoffs accumulate representatives in the original order, writing exactly
the same partial selector prefix. Polish vectorizes the five endpoint
choices and preserves candidate order, validity checks and fixed-error
fallback. Partition estimates batch independent subsets in SoA.

Integer products and accumulators retain i32 semantics: pixel products are
at most 65025, moment sums at most 1040400, scaled covariance expressions
at most 16646400, and weighted channel SSE at most 4161600 (weight <=4,
16 pixels). SSE totals fit u32. Floating operations retain each scalar
multiply/add/divide order with no fused, reassociated or reciprocal-estimate
intrinsics. Covariance-zero vectors may compute discarded IEEE NaNs; no
invalid float is used for a live nondegenerate seed or estimate. LS remains
scalar i64/f64. No relaxed Wasm SIMD is used.

Finite differential tests establish the explicitly recorded identity matrix,
not a proof over every possible block. AArch64 execution needs a native or
emulated runner; cross-target linting alone does not establish parity there.

The selector and polish kernels specialize observation weights at entry.
Unweighted selection retains the same projected palette window and strict
ascending ties. Unweighted polish is selected only without an objective and
without duplicate representatives (`expand == false`); every multiplicity is
then one. All other cases retain the original weight multiplication. This
removes multiplications by one without changing bounds, traversal or sums.
The array helpers and feature-checked unsafe dispatch are unchanged.

Palette storage specializes to N=4/8/16, with N+lane_width initialized
entries per channel. The latest projected-window base is N-2 and the last
load ends at N-2+lane_width, within that padding. Weighted full-palette
loads obey the same bound. Extra slots are never eligible for selection.
On std x86, AVX2 dispatch uses SSE4.1 with four elements for the unweighted window
only after an independent runtime SSE4.1 check. Weighted palettes, seeds,
polish and estimates retain their selected backend. No new unsafe block
or unchecked memory operation is added.

HDR pair distances use f64 vectors across eight independent blocks (two per
SSE4.1 vector, four per AVX2 vector). Runtime dispatch establishes the ISA.
Each unaligned load covers a fixed array at offsets whose width fits eight;
stores target equally sized local arrays. RGB operations preserve scalar order
without fused multiply-add or horizontal reduction. Other targets use scalar.
