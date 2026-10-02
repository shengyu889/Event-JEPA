# Event-JEPA 运行手册

Event-JEPA V1 在冻结的 GEP Stage-1 事件 token 上做时间预测式预训练。它不是原始事件流模型；训练前必须先得到形状为 `[N, D]` 的逐帧 `eventToken/*.pt`。

## 1. 环境和自检

RTX 5070 Ti 的 compute capability 是 12.0。不要沿用上游的 PyTorch 2.6/CUDA 11.8 轮子；本分支固定 PyTorch 2.7.1/CUDA 12.8，因为 PyTorch 从 2.7 起提供 Blackwell 支持。

```bash
conda env create -f environment.yml
conda activate gep
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name()); print(torch.cuda.get_arch_list())"
pytest -q
```

如果已经创建了 `gep` 环境，不必删环境重装。直接补齐 DSEC 预处理依赖：

```bash
conda activate gep
cd /home/tom/event-jepa/gep-event-jepa
python -m pip install "numpy<2.3" "numba==0.61.2" \
  "opencv-python-headless==4.11.0.86"

python - <<'PY'
import cv2
import numba
import numpy

print("OpenCV:", cv2.__version__)
print("Numba:", numba.__version__)
print("NumPy:", numpy.__version__)
PY
```

这里将 NumPy 限制为 `<2.3`，因为 Numba 0.61.2 的官方 wheel 要求
`numpy>=1.24,<2.3`。`opencv-python-headless` 不需要桌面 GUI，适合训练和
服务器环境；它仍然提供 `cv2.imread`、畸变校正和透视变换等本项目所需接口。

验证输出中必须出现你的 RTX 5070 Ti，并且架构列表包含 `sm_120`。若实验室 A100 驱动不支持 CUDA 12.8，请只在 A100 环境按 PyTorch 官方版本矩阵改用 cu126；不要在 5070 Ti 上退回 cu126。

数据目录必须满足：

```text
DATA_ROOT/
├── train_images/<sequence>/images/left/eventToken/<timestamp>.pt
└── test_images/<sequence>/images/left/eventToken/<timestamp>.pt
```

文件名必须是可转换为整数的时间戳。同一序列内按时间戳排序；窗口绝不会跨越 sequence。

## 2. 先生成一条 DSEC 序列的 Stage-1 event token

第一次只使用 `zurich_city_00_a` 冒烟，不要先下载或处理完整 300 GB 训练集。
解压后的关键路径必须是下面这样：

```text
/home/tom/event-jepa/datasets/DSEC/
├── train_events/zurich_city_00_a/events/left/events.h5
├── train_images/zurich_city_00_a/images/left/rectified/*.png
├── train_images/zurich_city_00_a/images/timestamps.txt
├── train_calibration/zurich_city_00_a/calibration/cam_to_cam.yaml
└── train_semantic_segmentation/train/zurich_city_00_a/11classes_renamed/*.png  # 可选
```

先检查四个必需输入，不要直接开始长时间处理：

```bash
test -f /home/tom/event-jepa/datasets/DSEC/train_events/zurich_city_00_a/events/left/events.h5 && echo "events OK"
test -f /home/tom/event-jepa/datasets/DSEC/train_images/zurich_city_00_a/images/timestamps.txt && echo "timestamps OK"
test -f /home/tom/event-jepa/datasets/DSEC/train_calibration/zurich_city_00_a/calibration/cam_to_cam.yaml && echo "calibration OK"
find /home/tom/event-jepa/datasets/DSEC/train_images/zurich_city_00_a/images/left/rectified -maxdepth 1 -name '*.png' | head
```

确认 Stage-1 Small checkpoint 仍能严格加载。你当前使用的路径是：

```text
/home/tom/event-jepa/checkpoints/gep_stage1_small.pt
```

然后执行完整的“原始事件 → eventImage → eventToken”流程：

```bash
conda activate gep
cd /home/tom/event-jepa/gep-event-jepa

PYTHONPATH=src python src/pre_dse.py \
  --root /home/tom/event-jepa/datasets/DSEC \
  --split train \
  --sequence zurich_city_00_a \
  --event-encoder-ckpt /home/tom/event-jepa/checkpoints/gep_stage1_small.pt \
  --prepare-pairs \
  --event-only \
  --device cuda:0 \
  --workers 1
```

这条命令只需要已经下载的 **Event Encoder Small**。`--event-only` 会跳过
DINOv2 image encoder，不需要第二个权重，也不会创建、删除或覆盖
`imageToken`。

处理完成后检查数量和 token 形状：

```bash
TOKEN_DIR=/home/tom/event-jepa/datasets/DSEC/train_images/zurich_city_00_a/images/left/eventToken
find "$TOKEN_DIR" -maxdepth 1 -name '*.pt' | wc -l

python - <<'PY'
from pathlib import Path
import torch

token_dir = Path("/home/tom/event-jepa/datasets/DSEC/train_images/zurich_city_00_a/images/left/eventToken")
paths = sorted(token_dir.glob("*.pt"), key=lambda path: int(path.stem))
assert paths, f"没有生成 token: {token_dir}"
token = torch.load(paths[0], map_location="cpu", weights_only=True)
print("首个 token:", paths[0])
print("shape:", tuple(token.shape), "dtype:", token.dtype)
assert tuple(token.shape) == (256, 384)
PY
```

如果 pair 已经成功生成、只是 token 阶段中断，重跑时可以去掉
`--prepare-pairs`，避免重新删除并生成 `eventImage/warpped/vis`。当前脚本以
sequence 为重算单位：每次 token 化会重建所选序列的 `eventToken`，不是从某个
`.pt` 文件断点续跑。单序列验收通过后，再逐条增加 `--sequence`，或不传
`--sequence` 处理该 split 下的全部序列。

## 3. RTX 5070 Ti 16 GB 本地 smoke

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

## 4. 4 × A100 80 GB 分布式训练

修改 `configs/event_jepa_a100.yaml` 的数据与输出路径后运行：

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=4 src/train_jepa.py \
  --config configs/event_jepa_a100.yaml
```

## 5. 导出并接入原 GEP 分类器

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

## 6. 公平对比记录表

| 方法 | Stage-1 encoder | GPU 数 | 有效 batch | token 数 | optimizer steps | 墙钟时间 | Seed | Top-1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Scratch | 相同 |  |  |  |  |  | 0/1/2 |  |
| GEP | 相同 |  |  |  |  |  | 0/1/2 |  |
| Event-JEPA | 相同 |  |  |  |  |  | 0/1/2 |  |

除方法本身外，固定数据划分、Stage-1 权重、增强、线性探测器、训练步数与评估协议；同时记录峰值显存、单步耗时、最终 loss、representation standard deviation 和 checkpoint 路径。
