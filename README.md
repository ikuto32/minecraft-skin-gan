# minecraft-skin-gan

64x64 Minecraft skin imagesを使ってDCGANを学習し、FID/KIDで評価するプロジェクトです。

## Setup

```bash
uv sync
```

## 学習（Hydra / OmegaConf）

学習/評価はHydra構成グループで実行します。`conf/config.yaml` の `defaults` で `mode=train` が選択されます。

```bash
uv run python main.py mode=train epochs=100 batch_size=64 data_dir=data/skins
```

主なoverride例:

```bash
uv run python main.py mode=train \
  epochs=200 \
  eval_every=10 \
  tracking.backend=wandb \
  tracking.project=minecraft-skin-gan
```

### TTUR (Two Time-Scale Update Rule)

TTUR（`https://arxiv.org/abs/1706.08500`）に対応しています。  
`lr_g`（Generator）と `lr_d`（Discriminator）を個別に設定できます。

```bash
uv run python main.py mode=train \
  lr_g=0.0001 \
  lr_d=0.0004
```

`lr_g` / `lr_d` を省略した場合は、従来どおり `lr` の値が両方に適用されます。

### パフォーマンスプロファイル

- `performance_profile=auto`（推奨）: 実行時に `device` / GPU世代（compute capability）/ PyTorch runtime capability を見て、`compile`・`amp_dtype`・`channels_last` を自動選択します。
- `performance_profile=safe`: デバッグ向けの安全設定です。`compile` 無効、AMP 無効、`channels_last` 無効。
- `performance_profile=max`: 可能な範囲で高速化設定を有効化します。

例:

```bash
uv run python main.py mode=train performance_profile=auto
uv run python main.py mode=train performance_profile=safe
```

## 追跡（W&B / MLflow）

### W&B

```bash
uv run python main.py mode=train tracking.backend=wandb tracking.project=minecraft-skin-gan
```

### MLflow

```bash
uv run python main.py mode=train \
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
uv run python main.py mode=eval \
  checkpoint=checkpoints/latest.pt \
  real_dir=data/skins \
  output_dir=outputs/eval \
  sample_count=2048
```


## DataLoader worker設定

- `num_workers` / `eval_num_workers` は `0` を許容します（シングルプロセスDataLoader）。
- `prefetch_factor` は `num_workers > 0` のときだけ有効です。`num_workers=0` の場合は `prefetch_factor=null` にしてください。
- デフォルト例として `conf/mode/eval.yaml` は `num_workers: 0` を使用しています。
