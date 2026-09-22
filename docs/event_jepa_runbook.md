# Event-JEPA 运行手册

Event-JEPA V1 在冻结的 GEP Stage-1 事件 token 上做时间预测式预训练。它不是原始事件流模型；训练前必须先得到形状为 `[N, D]` 的逐帧 `eventToken/*.pt`。

## 1. 环境和自检

```bash
conda env create -f environment.yml
conda activate gep
pytest -q
```

数据目录必须满足：

```text
DATA_ROOT/
├── train_images/<sequence>/images/left/eventToken/<timestamp>.pt
└── test_images/<sequence>/images/left/eventToken/<timestamp>.pt
```

文件名必须是可转换为整数的时间戳。同一序列内按时间戳排序；窗口绝不会跨越 sequence。

## 2. RTX 5070 Ti 16 GB 本地 smoke

只修改 `configs/event_jepa_smoke.yaml` 中的 `data_root` 和 `output_dir`：

```bash
PYTHONPATH=src python src/train_jepa.py \
  --config configs/event_jepa_smoke.yaml
```

断点续训：

```bash
PYTHONPATH=src python src/train_jepa.py \
  --config configs/event_jepa_smoke.yaml \
  --resume runs/event_jepa_smoke/latest.pt
```

smoke 验收：20 个成功 optimizer steps；loss、gradient norm 和 representation standard deviation 均为有限值；EMA 权重发生变化；峰值显存低于 16 GB；生成 `latest.pt` 和 step-20 检查点。

## 3. 4 × A100 80 GB 分布式训练

修改 `configs/event_jepa_a100.yaml` 的数据与输出路径后运行：

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=4 src/train_jepa.py \
  --config configs/event_jepa_a100.yaml
```

## 4. 导出并接入原 GEP 分类器

导出在线上下文 Transformer：

```bash
PYTHONPATH=src python src/export_jepa.py \
  --checkpoint runs/event_jepa_a100/latest.pt \
  --output runs/event_jepa_a100/gep_transformer.pt
```

运行分类前，在 `src/config.py` 的 `CLSConfig` 中统一设置：

- `dataset_name="nima"`
- `modality="event"`
- `transfer="linear"`
- Scratch、GEP、Event-JEPA 三组使用完全相同的 Stage-1 encoder checkpoint
- 每组使用三个随机种子

然后执行：

```bash
EVENT_JEPA_TRANSFORMER_CKPT=runs/event_jepa_a100/gep_transformer.pt \
PYTHONPATH=src python src/cls.py
```

## 5. 公平对比记录表

| 方法 | Stage-1 encoder | GPU 数 | 有效 batch | token 数 | optimizer steps | 墙钟时间 | Seed | Top-1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Scratch | 相同 |  |  |  |  |  | 0/1/2 |  |
| GEP | 相同 |  |  |  |  |  | 0/1/2 |  |
| Event-JEPA | 相同 |  |  |  |  |  | 0/1/2 |  |

除方法本身外，固定数据划分、Stage-1 权重、增强、线性探测器、训练步数与评估协议；同时记录峰值显存、单步耗时、最终 loss、representation standard deviation 和 checkpoint 路径。
