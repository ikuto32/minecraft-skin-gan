# minecraft-skin-gan

64x64 Minecraft skin imagesを使ってDCGANを学習し、FID/KIDで評価するプロジェクトです。

## Setup

### 1) 依存インストール

```bash
uv sync
```

### 2) 学習データを配置

`data/skins` に 64x64 のスキン画像（`.png` / `.jpg` / `.jpeg`）を配置します。

### 3) 学習を実行

```bash
uv run python main.py train \
  --data-dir data/skins \
  --epochs 100 \
  --batch-size 64
```

### 4) 出力物

- 生成サンプル画像: `outputs/epoch_XXXX.png`
- 最新チェックポイント: `checkpoints/latest.pt`
- 定期評価（有効時）: `outputs/eval/latest_metrics.json`, `outputs/eval/metrics_history.csv`

## Training

### 推奨GPU

- **最低ライン**: 8GB VRAM（例: RTX 3060）
- **推奨**: 12GB 以上（例: RTX 4070 / 4080）
- CPUでも実行可能ですが、学習時間は大幅に長くなります。

### 学習コマンド例

```bash
uv run python main.py train \
  --data-dir data/skins \
  --epochs 200 \
  --batch-size 64 \
  --eval-every 10 \
  --eval-sample-count 2048
```

## Inference（サンプル生成）

学習済みモデルの品質確認には、まず学習中に保存される `outputs/epoch_XXXX.png` を確認してください。

追加で定量評価付きの推論サンプルを作る場合は、チェックポイントを使って評価コマンドを実行します。

```bash
uv run python main.py eval \
  --checkpoint checkpoints/latest.pt \
  --real-dir data/skins \
  --output-dir outputs/eval \
  --sample-count 2048 \
  --seed 1234
```

## Evaluation

### 評価コマンド

学習済みチェックポイントから FID/KID を計算:

```bash
uv run python main.py eval \
  --checkpoint checkpoints/latest.pt \
  --real-dir data/skins \
  --output-dir outputs/eval \
  --sample-count 2048 \
  --seed 1234
```

学習中に定期評価を有効化:

```bash
uv run python main.py train \
  --data-dir data/skins \
  --epochs 100 \
  --eval-every 10 \
  --eval-sample-count 2048 \
  --eval-seed 1234
```

### 解釈の目安

- **FID**: 低いほど良い（実画像分布に近い）。
- **KID mean**: 低いほど良い。`kid_std` が小さいほど比較の信頼性が高い。
- メトリクス履歴は `outputs/eval/metrics_history.csv`、可視化は `outputs/eval/metrics.png` に保存。

### 比較時の注意（run 間比較）

- `--sample-count` を必ず一致させる（例: 2048 固定）。
- `--seed` を固定し、生成サンプルセットを run 間で揃える。
- 前処理（リサイズ 64x64、RGBA→RGB の扱い、正規化/復元）を完全一致させる。
- チェックポイント epoch とデータセットを揃えて比較する。
