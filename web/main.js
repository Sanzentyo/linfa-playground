import init, { predict_activity, predict_activity_from_rawdata, load_model } from "../pkg/linfa_playground.js";

// Utility: generate an array of 9 float32 values (xmin,xmax,xave, ... for x/y/z)
function randomFeatures() {
  const out = [];
  for (let axis = 0; axis < 3; axis++) {
    const center = Math.random() * 3 - 1.5;   // [-1.5, 1.5]
    const halfRange = Math.random() * 2;      // [0, 2]
    const min = center - halfRange;
    const max = center + halfRange;
    const ave = min + Math.random() * (max - min);
    out.push(min, max, ave);
  }
  return new Float32Array(out);
}

function renderResult(root, features, pred, elapsed) {
    const feat = Array.from(features).map(v => v.toFixed(4));
    root.innerHTML = `
    <table>
      <thead>
        <tr><th>feature</th><th>value</th></tr>
      </thead>
      <tbody>
        <tr><td>xmin</td><td>${feat[0]}</td></tr>
        <tr><td>xmax</td><td>${feat[1]}</td></tr>
        <tr><td>xave</td><td>${feat[2]}</td></tr>
        <tr><td>ymin</td><td>${feat[3]}</td></tr>
        <tr><td>ymax</td><td>${feat[4]}</td></tr>
        <tr><td>yave</td><td>${feat[5]}</td></tr>
        <tr><td>zmin</td><td>${feat[6]}</td></tr>
        <tr><td>zmax</td><td>${feat[7]}</td></tr>
        <tr><td>zave</td><td>${feat[8]}</td></tr>
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
