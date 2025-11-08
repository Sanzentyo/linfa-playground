#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct AccelData {
    pub timestamp: u64,
    pub accel_x: f32,
    pub accel_y: f32,
    pub accel_z: f32,
}

pub const WINDOW_SIZE: usize = 64;

pub fn extract_window_features(window: &[AccelData]) -> [f32; 6] {
    // mean_x, mean_y, mean_z, std_x, std_y, std_z
    let n = window.len() as f32;
    let mut sum_x = 0f32;
    let mut sum_y = 0f32;
    let mut sum_z = 0f32;
    for r in window.iter() {
        sum_x += r.accel_x;
        sum_y += r.accel_y;
        sum_z += r.accel_z;
    }
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let mean_z = sum_z / n;

    let mut var_x = 0f32;
    let mut var_y = 0f32;
    let mut var_z = 0f32;
    for r in window.iter() {
        var_x += (r.accel_x - mean_x).powi(2);
        var_y += (r.accel_y - mean_y).powi(2);
        var_z += (r.accel_z - mean_z).powi(2);
    }
    // population std (divide by n), then sqrt
    let std_x = (var_x / n).sqrt();
    let std_y = (var_y / n).sqrt();
    let std_z = (var_z / n).sqrt();

    [mean_x, mean_y, mean_z, std_x, std_y, std_z]
}

#[derive(bincode::Encode, bincode::Decode)]
pub struct BincodeDecisionTree {
    #[bincode(with_serde)]
    pub tree: linfa_trees::DecisionTree<f32, usize>,
}

use linfa::traits::Predict as _;
use wasm_bindgen::prelude::*;

const MODEL_BIN: &[u8] = include_bytes!("../activity_decision_tree.bincode");
static ACTIVITY_MODEL: std::sync::OnceLock<linfa_trees::DecisionTree<f32, usize>> =
    std::sync::OnceLock::new();

#[wasm_bindgen]
pub fn load_model() {
    ACTIVITY_MODEL.get_or_init(|| {
        let bincode_model = bincode::decode_from_slice::<BincodeDecisionTree, _>(
            MODEL_BIN,
            bincode::config::standard(),
        )
        .unwrap();

        bincode_model.0.tree
    });
}

#[wasm_bindgen]
pub fn predict_activity(features: Vec<f32>) -> i32 {
    // decode the trained model from embedded bytes
    let model = ACTIVITY_MODEL.get_or_init(|| {
        let bincode_model = bincode::decode_from_slice::<BincodeDecisionTree, _>(
            MODEL_BIN,
            bincode::config::standard(),
        )
        .unwrap();

        bincode_model.0.tree
    });
    let feature_array = ndarray::Array2::from_shape_vec((1, 6), features).unwrap();
    model.predict(&feature_array)[0] as i32
}

#[wasm_bindgen]
pub fn predict_activity_from_rawdata(data: Vec<f32>) -> i32 {
    // Use chunks_exact to avoid accidental indexing into a short remainder chunk.
    // If the input length is not a multiple of 3 the remainder is ignored instead of panicking.
    let accel_data: Vec<AccelData> = data
        .chunks_exact(3)
        .map(|chunk| AccelData {
            timestamp: 0,
            accel_x: chunk[0],
            accel_y: chunk[1],
            accel_z: chunk[2],
        })
        .collect();

    if accel_data.is_empty() {
        // No valid triplets -> return -1
        return -1;
    }

    let features = extract_window_features(&accel_data);
    predict_activity(features.to_vec())
}
