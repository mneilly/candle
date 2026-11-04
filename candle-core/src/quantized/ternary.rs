//! The ternary weight formats of ggml: `Q2_0` (upstream llama.cpp) and the `PQ2_0` and `PTQ1_0`
//! formats of PrismML's llama.cpp fork, used by the Ternary Bonsai models.
//!
//! All three hold weights `d * q` with one fp16 scale `d` per block:
//!
//! - `Q2_0`: 64 two-bit codes per block, `00, 01, 10, 11` for `q = -1, 0, 1, 2`, four per byte
//!   from the low bits (2.25 bits per weight).
//! - `PQ2_0`: the same codes, 128 per block (2.125 bits per weight).
//! - `PTQ1_0`: 128 trits `q = -1, 0, 1` per block, five per byte in `qs` and four per byte in `qh`,
//!   most significant trit first, interleaved across the bytes of a stage (1.75 bits per weight).
//!
//! Activations are quantized to Q8_0, in 32-wide sub-blocks as llama.cpp does, grouped to match
//! the weight block size.
//!
//! - 💻 [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp/tree/prism):
//!   `ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`

use super::k_quants::{BlockQ8_0, GgmlType, QK8_0};
use super::GgmlDType;
use half::f16;

pub const QK2_0: usize = 64;
pub const QK_PQ2_0: usize = 128;
pub const QK_PTQ1_0: usize = 128;

/// `N` consecutive Q8_0 blocks: the activation blocks of the ternary types, whose weight blocks
/// span several Q8_0 blocks. Only used as a `VecDotType`.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0x<const N: usize>(pub(crate) [BlockQ8_0; N]);

impl<const N: usize> GgmlType for BlockQ8_0x<N> {
    const DTYPE: GgmlDType = GgmlDType::Q8_0;
    const BLCK_SIZE: usize = QK8_0 * N;
    type VecDotType = Self;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (x, ys) in xs.iter().zip(ys.chunks_exact_mut(Self::BLCK_SIZE)) {
            BlockQ8_0::to_float(&x.0, ys)
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (y, xs) in ys.iter_mut().zip(xs.chunks_exact(Self::BLCK_SIZE)) {
            BlockQ8_0::from_float(xs, &mut y.0)
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self]) -> f32 {
        let nb = n / Self::BLCK_SIZE;
        xs[..nb]
            .iter()
            .zip(&ys[..nb])
            .map(|(x, y)| BlockQ8_0::vec_dot(Self::BLCK_SIZE, &x.0, &y.0))
            .sum()
    }
}

/// The four codes of a byte of two-bit codes, as `q = code - 1`.
const CODES2: [[i8; 4]; 256] = {
    let mut t = [[0i8; 4]; 256];
    let mut b = 0;
    while b < 256 {
        let mut j = 0;
        while j < 4 {
            t[b][j] = ((b >> (2 * j)) & 3) as i8 - 1;
            j += 1
        }
        b += 1
    }
    t
};

/// Trit `n` of a packed byte, most significant first, as `q = trit - 1`. A byte holds
/// `ceil(v * 256 / 243)` for the base-3 value `v` of its trits; multiplying by `3^n` (mod 256)
/// moves trit `n` to the top, where `(x * 3) >> 8` reads it.
#[inline(always)]
fn trit(b: u8, n: usize) -> i8 {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    ((b.wrapping_mul(POW3[n]) as u16 * 3) >> 8) as i8 - 1
}

/// `sum_k d_w * d_k * (w_k . a_k)` over the 32-wide sub-blocks `a_k` covered by one block of
/// weights `w`, decoded to `i8`.
fn dot_q8(d: f16, w: &[i8], ys: &[BlockQ8_0]) -> f32 {
    let mut sum = 0f32;
    for (w, y) in w.chunks_exact(QK8_0).zip(ys) {
        let s: i32 = w
            .iter()
            .zip(&y.qs)
            .map(|(&w, &a)| w as i32 * a as i32)
            .sum();
        sum += y.d.to_f32() * s as f32;
    }
    d.to_f32() * sum
}

/// `amax(xs)` as the scale and `round(x / scale)` clamped to `lo..=hi`.
fn quantize_block<const K: usize>(xs: &[f32], lo: i32, hi: i32) -> (f16, [i32; K]) {
    let amax = xs.iter().fold(0f32, |m, x| m.max(x.abs()));
    let id = if amax > 0. { 1. / amax } else { 0. };
    let mut q = [0i32; K];
    for (q, &x) in q.iter_mut().zip(xs) {
        *q = ((x * id).round() as i32).clamp(lo, hi)
    }
    (f16::from_f32(amax), q)
}

macro_rules! two_bit_block {
    ($name:ident, $qk:ident, $dtype:ident) => {
        #[derive(Debug, Clone, PartialEq)]
        #[repr(C)]
        pub struct $name {
            pub(crate) d: f16,
            pub(crate) qs: [u8; $qk / 4],
        }
        const _: () = assert!(std::mem::size_of::<$name>() == 2 + $qk / 4);

        impl $name {
            fn decode(&self) -> [i8; $qk] {
                let mut q = [0i8; $qk];
                for (q, &b) in q.chunks_exact_mut(4).zip(&self.qs) {
                    q.copy_from_slice(&CODES2[b as usize])
                }
                q
            }
        }

        impl GgmlType for $name {
            const DTYPE: GgmlDType = GgmlDType::$dtype;
            const BLCK_SIZE: usize = $qk;
            type VecDotType = BlockQ8_0x<{ $qk / QK8_0 }>;

            fn to_float(xs: &[Self], ys: &mut [f32]) {
                for (x, ys) in xs.iter().zip(ys.chunks_exact_mut($qk)) {
                    let d = x.d.to_f32();
                    for (y, &q) in ys.iter_mut().zip(x.decode().iter()) {
                        *y = q as f32 * d
                    }
                }
            }

            /// `round(x / amax)` clamped to `-1..=2`, as llama.cpp's reference quantizer.
            fn from_float(xs: &[f32], ys: &mut [Self]) {
                for (y, xs) in ys.iter_mut().zip(xs.chunks_exact($qk)) {
                    let (d, q) = quantize_block::<$qk>(xs, -1, 2);
                    y.d = d;
                    for (b, q) in y.qs.iter_mut().zip(q.chunks_exact(4)) {
                        *b = q
                            .iter()
                            .enumerate()
                            .fold(0u8, |b, (j, &q)| b | (((q + 1) as u8) << (2 * j)))
                    }
                }
            }

            fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
                Self::vec_dot_unopt(n, xs, ys)
            }

            fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
                let nb = n / $qk;
                xs[..nb]
                    .iter()
                    .zip(&ys[..nb])
                    .map(|(x, y)| dot_q8(x.d, &x.decode(), &y.0))
                    .sum()
            }
        }
    };
}

two_bit_block!(BlockQ2_0, QK2_0, Q2_0);
two_bit_block!(BlockPQ2_0, QK_PQ2_0, PQ2_0);

/// 128 trits: 24 `qs` bytes of five trits and 2 `qh` bytes of four, then the scale.
#[derive(Debug, Clone, PartialEq)]
#[repr(C)]
pub struct BlockPTQ1_0 {
    pub(crate) qs: [u8; 24],
    pub(crate) qh: [u8; 2],
    pub(crate) d: f16,
}
const _: () = assert!(std::mem::size_of::<BlockPTQ1_0>() == 28);

/// The `(first byte, bytes)` stages of `qs`: a stage of `c` bytes holds `5 c` weights, trit `n` of
/// its byte `m` being weight `n c + m` of the stage. llama.cpp tries stages of 32, 16 and 8 bytes
/// in turn while they fit, which for 24 bytes is one of 16 and one of 8.
const PTQ1_0_STAGES: [(usize, usize); 2] = [(0, 16), (16, 8)];

/// `ceil(v * 256 / 243)` for the base-3 value `v` of `trits` in `-1..=1`, most significant first,
/// padded with zero trits to five.
fn pack_trits(trits: &[i32]) -> u8 {
    let mut v = 0u16;
    for n in 0..5 {
        v = v * 3 + trits.get(n).map_or(0, |&t| (t + 1) as u16)
    }
    ((v * 256).div_ceil(243)) as u8
}

impl BlockPTQ1_0 {
    /// Trit `n` of each byte of a stage is a run of consecutive weights, so the loops over the
    /// bytes vectorize.
    fn decode(&self) -> [i8; QK_PTQ1_0] {
        fn trits(bytes: &[u8], n: usize, out: &mut [i8]) {
            for (o, &b) in out.iter_mut().zip(bytes) {
                *o = trit(b, n)
            }
        }
        let mut q = [0i8; QK_PTQ1_0];
        let mut o = 0;
        for (start, c) in PTQ1_0_STAGES {
            for n in 0..5 {
                trits(
                    &self.qs[start..start + c],
                    n,
                    &mut q[o + n * c..o + (n + 1) * c],
                )
            }
            o += 5 * c
        }
        for n in 0..4 {
            trits(&self.qh, n, &mut q[o + n * 2..o + (n + 1) * 2])
        }
        q
    }
}

impl GgmlType for BlockPTQ1_0 {
    const DTYPE: GgmlDType = GgmlDType::PTQ1_0;
    const BLCK_SIZE: usize = QK_PTQ1_0;
    type VecDotType = BlockQ8_0x<{ QK_PTQ1_0 / QK8_0 }>;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (x, ys) in xs.iter().zip(ys.chunks_exact_mut(QK_PTQ1_0)) {
            let d = x.d.to_f32();
            for (y, &q) in ys.iter_mut().zip(x.decode().iter()) {
                *y = q as f32 * d
            }
        }
    }

    /// `round(x / amax)` clamped to `-1..=1`, as llama.cpp's reference quantizer.
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (y, xs) in ys.iter_mut().zip(xs.chunks_exact(QK_PTQ1_0)) {
            let (d, q) = quantize_block::<QK_PTQ1_0>(xs, -1, 1);
            y.d = d;
            let mut o = 0;
            for (start, c) in PTQ1_0_STAGES {
                for m in 0..c {
                    let trits: Vec<i32> = (0..5).map(|n| q[o + n * c + m]).collect();
                    y.qs[start + m] = pack_trits(&trits)
                }
                o += 5 * c
            }
            for h in 0..2 {
                let trits: Vec<i32> = (0..4).map(|n| q[o + n * 2 + h]).collect();
                y.qh[h] = pack_trits(&trits)
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        let nb = n / QK_PTQ1_0;
        xs[..nb]
            .iter()
            .zip(&ys[..nb])
            .map(|(x, y)| dot_q8(x.d, &x.decode(), &y.0))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ternary weights times a scale, `n` of them, from a simple generator.
    fn ternary(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) % 3) as f32 - 1.
            })
            .map(|t| t * scale)
            .collect()
    }

    fn roundtrip<T: GgmlType>(xs: &[f32]) -> Vec<f32> {
        let mut q = vec![T::zeros(); xs.len() / T::BLCK_SIZE];
        T::from_float(xs, &mut q);
        let mut ys = vec![0f32; xs.len()];
        T::to_float(&q, &mut ys);
        ys
    }

    #[test]
    fn ternary_weights_roundtrip_exactly() {
        // A power-of-two scale is exact in fp16, and every block holds a +-1 so amax is the scale.
        let xs = ternary(512, 7, 0.125);
        for block in xs.chunks(64) {
            assert!(block.iter().any(|x| x.abs() == 0.125));
        }
        assert_eq!(roundtrip::<BlockQ2_0>(&xs), xs);
        assert_eq!(roundtrip::<BlockPQ2_0>(&xs), xs);
        assert_eq!(roundtrip::<BlockPTQ1_0>(&xs), xs);
    }

    #[test]
    fn two_bit_codes_hold_plus_two() {
        // Q2_0 can store 2 * d; PTQ1_0 clamps it to d.
        let mut xs = vec![0f32; 128];
        xs[0] = 1.;
        xs[1] = 0.5;
        xs[2] = -0.5;
        xs[3] = 0.2;
        let q2 = roundtrip::<BlockPQ2_0>(&xs);
        assert_eq!(&q2[..4], &[1., 1., -1., 0.]);
        // The block scale is amax, so +2 needs a value past amax: only by construction.
        let block = BlockPQ2_0 {
            d: f16::from_f32(0.5),
            qs: {
                let mut qs = [0x55u8; 32];
                qs[0] = 0b11_10_01_00;
                qs
            },
        };
        let mut ys = vec![0f32; 128];
        BlockPQ2_0::to_float(&[block], &mut ys);
        assert_eq!(&ys[..5], &[-0.5, 0., 0.5, 1., 0.]);
        assert!(ys[4..].iter().all(|&y| y == 0.));
    }

    #[test]
    fn ptq1_0_trit_layout() {
        // Weight 0 is the top trit of qs[0], weight 16 the next trit of qs[0], weight 80 the top
        // trit of qs[16], weight 120 the top trit of qh[0] and weight 121 that of qh[1].
        for (i, expected_byte) in [(0, 0), (16, 0), (1, 1), (80, 16), (85, 21), (88, 16)] {
            let mut xs = vec![0f32; 128];
            xs[i] = 1.;
            let mut q = [BlockPTQ1_0::zeros()];
            BlockPTQ1_0::from_float(&xs, &mut q);
            let zero = pack_trits(&[0; 5]);
            for (b, &v) in q[0].qs.iter().enumerate() {
                assert_eq!(v != zero, b == expected_byte, "weight {i}, byte {b}");
            }
        }
        for (i, h) in [(120, 0), (121, 1), (126, 0), (127, 1)] {
            let mut xs = vec![0f32; 128];
            xs[i] = -1.;
            let mut q = [BlockPTQ1_0::zeros()];
            BlockPTQ1_0::from_float(&xs, &mut q);
            let zero = pack_trits(&[0; 4]);
            assert_ne!(q[0].qh[h], zero, "weight {i}");
            assert_eq!(q[0].qh[1 - h], zero, "weight {i}");
        }
        // All-zero trits are 121 (base 3: 11111) and decode back to zeros.
        assert_eq!(pack_trits(&[0; 5]), (121u16 * 256).div_ceil(243) as u8);
        let unpack = |b: u8| (0..5).map(|n| trit(b, n)).collect::<Vec<_>>();
        assert_eq!(unpack(pack_trits(&[0; 5])), [0; 5]);
        assert_eq!(unpack(pack_trits(&[1, -1, 0, 1, -1])), [1, -1, 0, 1, -1]);
        // Every packed value decodes back to its trits.
        for v in 0..243u16 {
            let t: Vec<i32> = (0..5)
                .rev()
                .map(|n| (v / 3u16.pow(n)) as i32 % 3 - 1)
                .collect();
            let got: Vec<i32> = unpack(pack_trits(&t)).iter().map(|&q| q as i32).collect();
            assert_eq!(got, t, "{v}");
        }
    }

    fn check_vec_dot<T: GgmlType>(k: usize) {
        let w = ternary(k, 3, 0.25);
        let a: Vec<f32> = (0..k)
            .map(|i| ((i * 37 % 101) as f32 - 50.) / 25.)
            .collect();
        let mut qw = vec![T::zeros(); k / T::BLCK_SIZE];
        T::from_float(&w, &mut qw);
        let mut qa = vec![T::VecDotType::zeros(); k / T::BLCK_SIZE];
        T::VecDotType::from_float(&a, &mut qa);
        let exact: f32 = w.iter().zip(&a).map(|(w, a)| w * a).sum();
        let got = T::vec_dot(k, &qw, &qa);
        // Only the activations are rounded, to 8 bits per 32-wide block.
        let bound = a.iter().map(|a| a.abs()).sum::<f32>() * 0.25 / 127.;
        assert!(
            (got - exact).abs() <= bound,
            "{:?}: {got} vs {exact}",
            T::DTYPE
        );
        assert_eq!(got, T::vec_dot_unopt(k, &qw, &qa));
    }

    #[test]
    fn vec_dots_match_the_dequantized_dot() {
        check_vec_dot::<BlockQ2_0>(512);
        check_vec_dot::<BlockPQ2_0>(512);
        check_vec_dot::<BlockPTQ1_0>(512);
    }

    #[test]
    fn q8_0x_matches_q8_0() {
        let a: Vec<f32> = (0..256).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut x4 = vec![BlockQ8_0x::<4>::zeros(); 2];
        BlockQ8_0x::<4>::from_float(&a, &mut x4);
        let mut x1 = vec![BlockQ8_0::zeros(); 8];
        BlockQ8_0::from_float(&a, &mut x1);
        let flat: Vec<BlockQ8_0> = x4.iter().flat_map(|b| b.0.clone()).collect();
        assert_eq!(flat, x1);
        let mut ys = vec![0f32; 256];
        BlockQ8_0x::<4>::to_float(&x4, &mut ys);
        let mut ys1 = vec![0f32; 256];
        BlockQ8_0::to_float(&x1, &mut ys1);
        assert_eq!(ys, ys1);
        // Summed per group of four blocks, so equal up to rounding.
        let (d4, d1) = (
            BlockQ8_0x::<4>::vec_dot(256, &x4, &x4),
            BlockQ8_0::vec_dot(256, &x1, &x1),
        );
        assert!((d4 - d1).abs() <= 1e-6 * d1.abs(), "{d4} vs {d1}");
    }
}
