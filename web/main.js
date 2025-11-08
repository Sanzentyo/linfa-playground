import init, { predict_activity, predict_activity_from_rawdata, load_model } from "../pkg/linfa_playground.js";

// Utility: generate an array of 6 float32 values in a reasonable range
function randomFeatures() {
    // mean_x, mean_y, mean_z in [-1.5, 1.5], std_x, std_y, std_z in [0, 2]
    const means = Array.from({ length: 3 }, () => (Math.random() * 3 - 1.5));
    const stds = Array.from({ length: 3 }, () => (Math.random() * 2));
    return new Float32Array([...means, ...stds]);
}

function renderResult(root, features, pred, elapsed) {
    const feat = Array.from(features).map(v => v.toFixed(4));
    root.innerHTML = `
    <table>
      <thead>
        <tr><th>feature</th><th>value</th></tr>
      </thead>
      <tbody>
        <tr><td>mean_x</td><td>${feat[0]}</td></tr>
        <tr><td>mean_y</td><td>${feat[1]}</td></tr>
        <tr><td>mean_z</td><td>${feat[2]}</td></tr>
        <tr><td>std_x</td><td>${feat[3]}</td></tr>
        <tr><td>std_y</td><td>${feat[4]}</td></tr>
        <tr><td>std_z</td><td>${feat[5]}</td></tr>
      </tbody>
    </table>
    <p>予測クラス (数値ラベル): <strong>${pred}</strong></p>
    <p>予測にかかった時間: ${elapsed} ms</p>
  `;
}

// ランダムにxyzの値に相当するものの配列を生成し、モデルで予測を行う
// ウィンドウサイズは50サンプル
// 生成したデータも返す
function runPredictionFromRandomRawData() {
  const sampleCount = 50; // 1サンプル = (x,y,z) の3要素
  const rawData = [];
  for (let i = 0; i < sampleCount; i++) {
    rawData.push(Math.random() * 3 - 1.5); // x
    rawData.push(Math.random() * 3 - 1.5); // y
    rawData.push(Math.random() * 3 - 1.5); // z
  }
  // これをモデルに渡して予測を行う (長さは sampleCount*3 で 3の倍数保証)
  const pred = predict_activity_from_rawdata(new Float32Array(rawData));
  return { pred, rawData };
}

async function main() {
    // Initialize the generated wasm JS glue
    await init();

    const loadTime = document.getElementById("load-time");
    const loadBtn = document.getElementById("load-model");

    loadBtn.addEventListener("click", () => {
        const start = performance.now();
        load_model();
        const end = performance.now();
        const duration = end - start;
        loadTime.textContent = `モデルの読み込みにかかった時間: ${duration.toFixed(2)} ms`;
        console.log(`Model loading took ${duration} ms`);
    });

    const out = document.getElementById("out");
    const runBtn = document.getElementById("run");

    runBtn.addEventListener("click", () => {
        const start = performance.now();
        const f = randomFeatures();
        // wasm-bindgen can accept a regular array or TypedArray for Vec<f32>
        const pred = predict_activity(f);
        const end = performance.now();
        const elapsed = end - start;
        console.log(`Prediction took ${elapsed} ms`);
        renderResult(out, f, pred, elapsed);
    });

    const runRawBtn = document.getElementById("run-raw");
    const outRaw = document.getElementById("out-raw");
    runRawBtn.addEventListener("click", () => {
        const start = performance.now();
        const { pred, rawData } = runPredictionFromRandomRawData();
        const end = performance.now();
        const elapsed = end - start;
        console.log(`Raw data prediction took ${elapsed} ms`);
        outRaw.innerHTML = `
      <p>生成した生データ (50サンプル):</p>
      <pre>[${rawData.map(v => v.toFixed(4)).join(", ")}]</pre>
      <p>予測クラス (数値ラベル): <strong>${pred}</strong></p>
      <p>予測にかかった時間: ${elapsed} ms</p>
    `;
    });
    
}

main().catch(err => {
    console.error(err);
    document.getElementById("out").textContent = "初期化に失敗しました: " + err;
});
