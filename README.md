# minecraft-skin-gan

64x64 Minecraft skin imagesを使ってDCGANを学習する最小プロジェクトです。

## Setup

```bash
uv sync
```

## Evaluation

### 評価コマンド

学習済みチェックポイントから FID/KID を計算:

```bash
uv run python src/eval.py \
  --checkpoint checkpoints/latest.pt \
  --real-dir data/skins \
  --output-dir outputs/eval \
  --sample-count 2048 \
  --seed 1234
```

学習中に定期評価を有効化:

```bash
uv run python src/train.py \
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
