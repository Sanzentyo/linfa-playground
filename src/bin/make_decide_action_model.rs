use linfa::prelude::*;
use linfa_playground::{AccelData, BincodeDecisionTree};
use linfa_trees::DecisionTree;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// Window sizeはこれを使う
use linfa_playground::WINDOW_SIZE;
use std::io::Write;
use std::path::Path;



fn main() -> anyhow::Result<()> {
    // activity -> label mapping
    let mapping = vec![
        ("sit.csv", 0usize),
        ("walk-with-hand.csv", 1usize),
        ("walking-in-pocket.csv", 2usize),
        ("climb-up.csv", 3usize),
        ("four-legged-walking.csv", 4usize),
    ];

    // Collect windows per label first for balancing
    let mut windows_per_label: Vec<Vec<[f32; 9]>> = vec![Vec::new(); mapping.len()];

    let data_dir = Path::new("data");
    if !data_dir.exists() {
        anyhow::bail!("data directory not found: {}", data_dir.display());
    }
    for (file_name, label) in mapping.iter() {
        let path = data_dir.join(file_name);
        if !path.exists() {
            eprintln!("warning: {} not found, skipping", path.display());
            continue;
        }

        let file = std::fs::File::open(&path)?;
        let mut csv_reader = csv::Reader::from_reader(file);
        let csv_data = csv_reader
            .deserialize::<AccelData>()
            .into_iter()
            .collect::<Result<Vec<AccelData>, csv::Error>>()?;

        if csv_data.is_empty() {
            eprintln!("warning: {} empty, skipping", path.display());
            continue;
        }

        // Skip first and last 1 second based on timestamp
        let min_ts = csv_data.iter().map(|r| r.timestamp).min().unwrap();
        let max_ts = csv_data.iter().map(|r| r.timestamp).max().unwrap();
        let start_cut = min_ts + 1000; // assume millis
        let end_cut = max_ts.saturating_sub(1000);
        let filtered: Vec<AccelData> = csv_data
            .into_iter()
            .filter(|r| r.timestamp > start_cut && r.timestamp < end_cut)
            .collect();

        if filtered.len() < WINDOW_SIZE {
            eprintln!("warning: {} insufficient data after trimming, skipping", path.display());
            continue;
        }

        // non-overlapping windows
        let mut i = 0usize;
        while i + WINDOW_SIZE <= filtered.len() {
            let window = &filtered[i..i + WINDOW_SIZE];
            let feat = linfa_playground::extract_window_features(window);
            windows_per_label[*label].push(feat);
            i += WINDOW_SIZE / 2; // stride == WINDOW_SIZE / 2
        }
    }
    // Balance classes: find minimum window count > 0
    let mut rng_balance = SmallRng::seed_from_u64(7);
    let counts: Vec<usize> = windows_per_label.iter().map(|v| v.len()).collect();
    let non_zero: Vec<usize> = counts.iter().cloned().filter(|&c| c > 0).collect();
    if non_zero.is_empty() {
        anyhow::bail!("no feature windows extracted — check data files");
    }
    let target = *non_zero.iter().min().unwrap();
    if target == 0 {
        anyhow::bail!("no usable data after trimming");
    }
    let mut features: Vec<[f32; 9]> = Vec::new();
    let mut labels: Vec<usize> = Vec::new();
    for (label_idx, feats) in windows_per_label.into_iter().enumerate() {
        if feats.is_empty() { continue; }
        let mut feats_mut = feats;
        use rand::seq::SliceRandom;
        feats_mut.shuffle(&mut rng_balance);
        for f in feats_mut.into_iter().take(target) {
            features.push(f);
            labels.push(label_idx);
        }
    }

    let n_samples = features.len();
    let mut feature_array = Array2::<f32>::zeros((n_samples, 9));
    for (i, f) in features.iter().enumerate() {
        for j in 0..9 {
            feature_array[[i, j]] = f[j];
        }
    }

    let label_array = ndarray::Array1::from(labels.clone());

    let dataset = linfa::Dataset::new(feature_array, label_array);

    let mut rng = SmallRng::seed_from_u64(42);
    let (train, test) = dataset.shuffle(&mut rng).split_with_ratio(0.8);

    // Train a Decision Tree classifier (default params)
    let model = DecisionTree::params().fit(&train)?;

    let pred = model.predict(&test);

    // compute accuracy manually
    let y_true = test.targets();
    let correct = pred
        .iter()
        .zip(y_true.iter())
        .filter(|(p, t)| p == t)
        .count();
    let acc = correct as f32 / y_true.len() as f32;
    println!(
        "Test accuracy: {:.2}% ({} samples)",
        acc * 100.0,
        test.nsamples()
    );

    let mut tikz = std::fs::File::create("activity_decision_tree.tex")?;
    tikz.write_all(model.export_to_tikz().with_legend().to_string().as_bytes())?;

    // parameterの保存
    let model_path = Path::new("activity_decision_tree.bincode");
    let mut model_file = std::fs::File::create(model_path)?;
    let model = BincodeDecisionTree { tree: model };
    bincode::encode_into_std_write(&model, &mut model_file, bincode::config::standard())?;

    Ok(())
}
