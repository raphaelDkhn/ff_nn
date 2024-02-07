use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_fc4_weight() -> Tensor<FP16x16> {
    let mut shape = array![1, 32];

    let mut data = array![FP16x16 { mag: 2443, sign: true }, FP16x16 { mag: 6157, sign: true }, FP16x16 { mag: 737, sign: false }, FP16x16 { mag: 6020, sign: false }, FP16x16 { mag: 7607, sign: false }, FP16x16 { mag: 975, sign: false }, FP16x16 { mag: 2603, sign: false }, FP16x16 { mag: 2600, sign: false }, FP16x16 { mag: 2116, sign: false }, FP16x16 { mag: 2539, sign: true }, FP16x16 { mag: 5757, sign: false }, FP16x16 { mag: 796, sign: false }, FP16x16 { mag: 5968, sign: false }, FP16x16 { mag: 2564, sign: false }, FP16x16 { mag: 1069, sign: true }, FP16x16 { mag: 4993, sign: true }, FP16x16 { mag: 695, sign: false }, FP16x16 { mag: 156, sign: false }, FP16x16 { mag: 596, sign: true }, FP16x16 { mag: 3143, sign: true }, FP16x16 { mag: 944, sign: true }, FP16x16 { mag: 8459, sign: false }, FP16x16 { mag: 2934, sign: true }, FP16x16 { mag: 5920, sign: true }, FP16x16 { mag: 160, sign: false }, FP16x16 { mag: 5601, sign: true }, FP16x16 { mag: 8961, sign: true }, FP16x16 { mag: 749, sign: true }, FP16x16 { mag: 2431, sign: true }, FP16x16 { mag: 3145, sign: false }, FP16x16 { mag: 5518, sign: false }, FP16x16 { mag: 3658, sign: false }];

    TensorTrait::new(shape.span(), data.span())
}