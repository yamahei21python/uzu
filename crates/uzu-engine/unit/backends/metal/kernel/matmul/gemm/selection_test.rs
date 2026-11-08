use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use super::*;
use crate::{
    backends::{
        common::{Context, kernel::matmul::MatmulKernel},
        metal::{context::MetalContext, kernel::matmul::MatmulMetalKernel},
    },
    tests::matmul::{a8_format, matmul_shape as shape, quantized_matmul_shape},
};

fn quant(mut shape: MatmulShape) -> MatmulShape {
    shape.b_prologue = GemmBPrologueKind::ScaleSymmetricDequant;
    shape.b_bits = Some(4);
    shape.b_group_size = Some(64);
    shape.signed_codes = true;
    shape
}

fn w8_zero_point(mut shape: MatmulShape) -> MatmulShape {
    shape.b_prologue = GemmBPrologueKind::ScaleZeroPointDequant;
    shape.b_bits = Some(8);
    shape.b_group_size = Some(32);
    shape.signed_codes = true;
    shape
}

#[uzu_test]
fn zero_point_int8_weights_select_split_k() {
    // int8 zero-point weights must split like their symmetric counterparts: the
    // split-k scratch accumulates in f32, so splitting is numerically safe again.
    for (k, expected_split_k) in [(256u32, 2u32), (512, 4), (768, 6), (1024, 8)] {
        let plan = problem(w8_zero_point(shape(16, 128, k)), DataType::BF16)
            .select_plan_for_engine(GemmEngine::Simdgroup)
            .expect("simdgroup plan");
        assert_eq!(plan.split_k, expected_split_k, "k={k}");
    }
    let m33 = problem(w8_zero_point(shape(33, 128, 1024)), DataType::BF16)
        .select_plan_for_engine(GemmEngine::Simdgroup)
        .expect("simdgroup plan");
    assert_eq!(m33.split_k, 8, "m=33");
    // W4 zero-point keeps the same selection rule.
    let mut w4 = w8_zero_point(shape(16, 128, 1024));
    w4.b_prologue = GemmBPrologueKind::ScaleZeroPointDequant;
    w4.b_bits = Some(4);
    let w4_plan = problem(w4, DataType::BF16).select_plan_for_engine(GemmEngine::Simdgroup).expect("simdgroup plan");
    assert_eq!(w4_plan.split_k, 8, "W4 zero-point");
    // The wide prefill shape still splits on the MXU.
    let wide = problem(w8_zero_point(shape(512, 4096, 4096)), DataType::BF16).select_plan();
    assert_eq!(wide.engine, GemmEngine::Mxu);
    assert_eq!(wide.split_k, 4, "m=512 n=4096 k=4096");
    // The simdgroup engine splits the same batch size once N is narrow enough.
    let narrow = problem(w8_zero_point(shape(512, 512, 4096)), DataType::BF16)
        .select_plan_for_engine(GemmEngine::Simdgroup)
        .expect("simdgroup plan");
    assert_eq!(narrow.split_k, 8, "m=512 n=512 k=4096");
}

fn problem(
    shape: MatmulShape,
    output_data_type: DataType,
) -> GemmProblem {
    GemmProblem::new(shape, DataType::BF16, output_data_type, true)
}

#[uzu_test]
fn illegal_layouts_fall_back() {
    use GemmEngine::*;

    let mut invalid_layout = quant(shape(64, 4096, 4096));
    invalid_layout.b_transpose = false;
    assert_eq!(problem(invalid_layout, DataType::BF16).select_plan().engine, Simdgroup);
    invalid_layout.a = a8_format();
    assert_eq!(problem(invalid_layout, DataType::BF16).select_plan().engine, Mxu);
    assert_eq!(problem(quant(shape(64, 4096, 4095)), DataType::BF16).select_plan().engine, Simdgroup);
}

#[uzu_test]
fn forced_engine_errors_are_preserved() {
    let huge = shape(u32::MAX, u32::MAX, u32::MAX);
    assert_eq!(
        GemmProblem::new(huge, DataType::BF16, DataType::BF16, false).select_plan_for_engine(GemmEngine::Mxu),
        Err(GemmPlanError::MxuUnavailable)
    );

    let mut invalid_layout = quant(huge);
    invalid_layout.b_transpose = false;
    assert_eq!(
        problem(invalid_layout, DataType::BF16).select_plan_for_engine(GemmEngine::Mxu),
        Err(GemmPlanError::UnsupportedLayout("quantized weights require transposed contiguous B"))
    );
}

#[uzu_test]
fn int8_activations_require_mxu_tiled_weights() {
    let context = block_on(MetalContext::new()).expect("Metal context");
    if !context.supports_mxu {
        return;
    }
    let kernel =
        MatmulMetalKernel::new(&context, DataType::BF16, DataType::BF16, DataType::BF16).expect("matmul kernel");
    for (n, k, bits, expected) in [
        (96, 256, 4, true),
        (72, 256, 4, false),
        (64, 40960, 4, true),
        (64, 41088, 4, false),
        (64, 20480, 8, true),
        (64, 20608, 8, false),
    ] {
        let shape = quantized_matmul_shape(16, n, k, bits, 64);
        assert_eq!(kernel.select_activation_quantization(&shape, &context).is_some(), expected, "n={n} k={k} W{bits}");
    }
}
