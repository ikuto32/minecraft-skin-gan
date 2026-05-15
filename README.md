# minecraft-skin-gan

64x64 Minecraft skin imagesを使ってDCGANを学習し、FID/KIDで評価するプロジェクトです。

## Setup

```bash
uv sync
```

## 学習（Hydra / OmegaConf）

学習はHydra構成で実行します。`conf/train.yaml` がデフォルト設定です。

```bash
uv run python main.py train epochs=100 batch_size=64 data_dir=data/skins
```

主なoverride例:

```bash
uv run python main.py train \
  epochs=200 \
  eval_every=10 \
  tracking.backend=wandb \
  tracking.project=minecraft-skin-gan
```

## 追跡（W&B / MLflow）

### W&B

```bash
uv run python main.py train tracking.backend=wandb tracking.project=minecraft-skin-gan
```

### MLflow

```bash
uv run python main.py train \
  tracking.backend=mlflow \
  tracking.mlflow_tracking_uri=http://127.0.0.1:5000 \
  tracking.mlflow_experiment=minecraft-skin-gan
```

- 学習loss / eval(FID, KID)をメトリクスとして記録
- 各epochの生成画像をartifactとして記録

## 自動resume + ベストモデル保存

- `auto_resume=true`（デフォルト）時、`checkpoints/latest.pt` が存在すれば自動再開します。
- 明示的に指定する場合は `resume=checkpoints/latest.pt` を使います。
- 評価実行時、`best_metric`（`fid` or `kid_mean`）が改善したら `checkpoints/best.pt` を保存します。
- 常に `checkpoints/latest.pt` は更新されます。

## Evaluation

```bash
uv run python main.py eval \
  --checkpoint checkpoints/latest.pt \
  --real-dir data/skins \
  --output-dir outputs/eval \
  --sample-count 2048
```
