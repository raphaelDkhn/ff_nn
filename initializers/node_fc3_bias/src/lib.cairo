use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_fc3_bias() -> Tensor<FP16x16> {
    let mut shape = array![32];

    let mut data = array![
        FP16x16 { mag: 6540, sign: true },
        FP16x16 { mag: 9626, sign: true },
        FP16x16 { mag: 9006, sign: true },
        FP16x16 { mag: 2855, sign: true },
        FP16x16 { mag: 8366, sign: true },
        FP16x16 { mag: 247885, sign: false },
        FP16x16 { mag: 173587, sign: false },
        FP16x16 { mag: 8899, sign: false },
        FP16x16 { mag: 132434, sign: false },
        FP16x16 { mag: 1428, sign: true },
        FP16x16 { mag: 4606, sign: true },
        FP16x16 { mag: 186, sign: true },
        FP16x16 { mag: 1034, sign: false },
        FP16x16 { mag: 422755, sign: false },
        FP16x16 { mag: 5132, sign: true },
        FP16x16 { mag: 11770, sign: true },
        FP16x16 { mag: 1911, sign: true },
        FP16x16 { mag: 2371, sign: false },
        FP16x16 { mag: 7451, sign: true },
        FP16x16 { mag: 4029, sign: true },
        FP16x16 { mag: 10004, sign: true },
        FP16x16 { mag: 8391, sign: true },
        FP16x16 { mag: 6661, sign: true },
        FP16x16 { mag: 8, sign: true },
        FP16x16 { mag: 1053, sign: false },
        FP16x16 { mag: 5, sign: true },
        FP16x16 { mag: 17394, sign: true },
        FP16x16 { mag: 4675, sign: true },
        FP16x16 { mag: 6231, sign: true },
        FP16x16 { mag: 1198, sign: true },
        FP16x16 { mag: 2541, sign: true },
        FP16x16 { mag: 126812, sign: false }
    ];

    TensorTrait::new(shape.span(), data.span())
}
