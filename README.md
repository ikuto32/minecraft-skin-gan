# minecraft-skin-gan

64x64 Minecraft skin imagesを使ってDCGANを学習し、FID/KID/Precision/Recallで評価するプロジェクトです。

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

`n_critic` を使うと、1イテレーションで Discriminator を複数回更新してから Generator を1回更新できます（例: `n_critic=5`）。

```bash
uv run python main.py mode=train \
  n_critic=5
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

## 追跡（W&B）

### 異常時にまず見るメトリクス（運用ガイド）

学習が不安定なときは、まず以下の診断キーを確認してください（W&Bで確認可能）。

- `diag/d_real_mean`, `diag/d_fake_mean`, `diag/d_real_var`, `diag/d_fake_var`
  - Dが飽和すると `d_real_mean` が高止まり、`d_fake_mean` が極端に低下しやすい。
- `diag/grad_norm_g`, `diag/grad_norm_d`
  - 急激なスパイク（例: `1e3`超）は勾配爆発の兆候。
- `diag/update_norm_g`, `diag/update_norm_d`, `diag/update_ratio_g_over_d`
  - 更新量比 `|Δθ_G|/|Δθ_D|` が長く極端（過小/過大）なら、学習率バランス崩れを疑う。

ランタイムでは軽量ガードとして、`NaN/Inf`、勾配爆発疑い、D飽和疑いを検出した際に warning を出力します。

### W&B

```bash
uv run python main.py mode=train tracking.backend=wandb tracking.project=minecraft-skin-gan
```

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

必要に応じて高コスト指標をON/OFFできます（`mode=eval`）。

```bash
uv run python main.py mode=eval enable_fid=true enable_kid=false enable_precision_recall=true
```

### 指標の解釈（fidelity vs coverage）

- **FID（低いほど良い）**: 生成分布が実データ分布にどれだけ近いかを測る、総合的なfidelity指標。
- **KID（低いほど良い）**: FIDと同様に分布距離を見るが、サンプル数が少ない場合でも比較的安定しやすい。
- **Precision（高いほど良い）**: 生成画像の「見た目の正確さ・品質」寄り。高いほど、生成サンプルが実データ多様体の近傍にある割合が高い（fidelity重視）。
- **Recall（高いほど良い）**: 生成画像の「カバレッジ・多様性」寄り。高いほど、実データ多様体をどれだけ広く覆えているか（coverage重視）。

使い分けの目安:
- 画質崩れや不自然さを抑えたいとき: **FID/KID + Precision** を重視。
- モード崩壊（似た出力ばかり）を避けたいとき: **Recall** も必ず確認。
- 運用では `best_metric=pr_tradeoff` のように「Recall下限を満たす中でFID最小」を使うと、品質と多様性を同時に管理しやすくなります。

### 再現性を担保した評価（複数seed必須）

- 単一seedのFID/KIDを最終結果として採用しないでください。
- `eval_seeds` に複数seedを指定し、各seedの結果と平均・標準偏差・最良値を確認してください。
- 学習時のデフォルトプリセットでは `eval_seeds: [1234, 2024, 3407, 7777]` を使用します。

```bash
uv run python main.py mode=train \
  eval_every=1 \
  eval_seeds='[1234,2024,3407,7777]'
```

評価出力には以下が追加されます。
- `outputs/eval/seed_metrics_latest.json`（seed別 + summary）
- `outputs/eval/seed_metrics_latest.csv`（seed別FID/KID）



## Loss設定メモ

- `loss.name=wgan_gp` の場合のみ `loss.gp_lambda` を使用し、`loss.gp_lambda >= 0` が必須です。
- `loss.name` が `wgan_gp` 以外のときは `loss.gp_lambda` は計算に使用されません（指定しても無視されます）。

## DataLoader worker設定

- `num_workers` / `eval_num_workers` は `0` を許容します（シングルプロセスDataLoader）。
- `prefetch_factor` は `num_workers > 0` のときだけ有効です。`num_workers=0` の場合は `prefetch_factor=null` にしてください。
- デフォルト例として `conf/mode/eval.yaml` は `num_workers: 0` を使用しています。
