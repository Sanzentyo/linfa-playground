use anyhow::Result;
use linfa::prelude::*;

const MODEL_BIN: &[u8] = include_bytes!("../../activity_decision_tree.bincode");

fn main() -> Result<()> {
    // load the trained model from file
    let mut model_file = std::io::Cursor::new(MODEL_BIN);

    let bincode_model: linfa_playground::BincodeDecisionTree =
        bincode::decode_from_std_read(&mut model_file, bincode::config::standard())?;
    let model = bincode_model.tree;

    println!("Model loaded successfully.");

    let example_features = (0..linfa_playground::WINDOW_SIZE)
        .map(|i| linfa_playground::AccelData {
            timestamp: i as u64,
            accel_x: 0.0,
            accel_y: 0.0,
            accel_z: 9.8,
        })
        .collect::<Vec<linfa_playground::AccelData>>();

    let feat = linfa_playground::extract_window_features(&example_features);
    let feature_array = ndarray::Array2::from_shape_vec((1, 9), feat.to_vec())?;

    let result = model.predict(&feature_array);
    let predicted_label = result[0];
    println!("Predicted label for the example features: {}", predicted_label);

    Ok(())
}