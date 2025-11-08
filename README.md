# linfa-playground WASM demo

# 注意: 以下のドキュメントは半分程度がLLMで生成されています

ブラウザから `wasm-bindgen` で Rust の `predict_activity(features: Vec<f32>) -> u32` を呼び出し、ランダムなダミー特徴量で予測を表示する最小デモです。

## 要件
- Rust と `wasm-pack`
- 任意の静的ファイルサーバ (例: Node.js の `npx serve`)

## ビルド
### モデルの準備
まず、リポジトリの直下に、`activity_decision_tree.bincode` を空ファイルでいいので作成しておきます。

ヘッダが、`timestamp`, `accel_x`, `accel_y`, `accel_z`の順である CSV ファイル (例: `data/activity.csv`) を用意し、そのファイルとidをsrc/bin/make_decide_action_model.rs の 以下の部分と対応させてから、
``` rs
    // activity -> label mapping
    let mapping = vec![
        ("sit.csv", 0usize),
        ("walk-with-hand.csv", 1usize),
        ("walking-in-pocket.csv", 2usize),
        ("climb-up.csv", 3usize),
        ("four-legged-walking.csv", 4usize),
    ];
```
これを、以下のコマンドで実行することで、`activity_decision_tree.bincode` にモデルのバイナリが保存されます。
```
cargo run --bin make_decide_action_model --release
```

生成された `activity_decision_tree.bincode`はconstとしてWASMバイナリに埋め込まれるので、モデルを更新する場合は以下のビルド手順を再度実行してください。

```
# Web ターゲットでビルドして ./pkg を生成
wasm-pack build --target web
```

ビルドに失敗する場合は、`Cargo.toml` の `getrandom` に `features = ["js"]` が入っているか確認してください。

## 実行

```
# リポジトリルートを公開 (web/ と pkg/ の両方にアクセスするため)
# 例: Node.js がある場合
npx --yes serve -l 8080 .

# ブラウザで開く
# http://localhost:8080/web/
```

ページの「ランダムに予測する」ボタンを押すと、6 要素のダミー特徴量
`[mean_x, mean_y, mean_z, std_x, std_y, std_z]` が生成され、`predict_activity` に渡されて数値ラベルの予測結果が表示されます。

## 構成
- `src/lib.rs`: `#[wasm_bindgen] pub fn predict_activity(features: Vec<f32>) -> u32` を公開
- `activity_decision_tree.bincode`: 事前学習済みモデル (バイナリ) を `include_bytes!` で組み込み
- `web/index.html`, `web/main.js`: ブラウザ側の最小 UI / 起動コード
- `pkg/`: `wasm-pack build --target web` で生成されるアーティファクト

## 備考
- JS 側では `Float32Array` を渡していますが、通常の `number[]` でも `Vec<f32>` に変換されます。
- 予測クラスのラベル名が必要であれば、Rust 側にマッピング関数を追加してエクスポートすることも可能です。
