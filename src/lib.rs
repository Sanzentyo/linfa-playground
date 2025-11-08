#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct AccelData {
    pub timestamp: u64,
    pub accel_x: f32,
    pub accel_y: f32,
    pub accel_z: f32,
}

pub const WINDOW_SIZE: usize = 64;

pub fn extract_window_features(window: &[AccelData]) -> [f32; 9] {
    // xmin, xmax, xave, ymin, ymax, yave, zmin, zmax, zave
    let n = window.len() as f32;
    let mut sum_x = 0f32;
    let mut sum_y = 0f32;
    let mut sum_z = 0f32;

    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;

    for r in window.iter() {
        sum_x += r.accel_x;
        sum_y += r.accel_y;
        sum_z += r.accel_z;

        if r.accel_x < min_x { min_x = r.accel_x; }
        if r.accel_x > max_x { max_x = r.accel_x; }
        if r.accel_y < min_y { min_y = r.accel_y; }
        if r.accel_y > max_y { max_y = r.accel_y; }
        if r.accel_z < min_z { min_z = r.accel_z; }
        if r.accel_z > max_z { max_z = r.accel_z; }
    }

    let ave_x = sum_x / n;
    let ave_y = sum_y / n;
    let ave_z = sum_z / n;

    [min_x, max_x, ave_x, min_y, max_y, ave_y, min_z, max_z, ave_z]
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
    let feature_array = ndarray::Array2::from_shape_vec((1, 9), features).unwrap();
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
