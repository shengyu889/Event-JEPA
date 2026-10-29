# Event-JEPA v8.2 — 冻结底座与零融合一致性审计

目的：不新增任何训练，检查已经完成的两组 **Raw pretrained SEA-RAFT + GEP/JEPA** 负结果是否来自冻结模型/检查点污染或融合接口的错误。此脚本专门用于 v8.2 `--freeze-raft` 训练 checkpoint；不适用于先在 outdoor_day2 微调过的旧 v8.1 三组。

## 0. 安装

把 ZIP 放在服务器 `/mnt/data2/gsy/Event-JEPA` 项目根目录，执行：

```bash
cd /mnt/data2/gsy/Event-JEPA
conda activate eventjepa
unzip -o Event-JEPA_v8_Fusion_Integrity_Audit.zip -d .
export PYTHONPATH="$PWD/dinov2:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/test_audit_v8_fusion_integrity.py
mkdir -p results logs
```

单元测试验证比较方法、零门控恢复基线和 EPE 统计。这些测试仅为 CPU 合成测试，并不替代服务器真实模型验证。

## 1. 审计 Raw + GEP

```bash
CUDA_VISIBLE_DEVICES=5 python -u scripts/audit_v8_fusion_integrity.py \
  --mode gep \
  --checkpoint runs/searaft_v8_2_rawbase_gep_full_seed0/last.pt \
  --root datasets/MVSEC \
  --samples 8 \
  --output results/v8_fusion_integrity_gep.json \
  2>&1 | tee logs/v8_fusion_integrity_gep.log
```

## 2. 审计 Raw + Event-JEPA

```bash
CUDA_VISIBLE_DEVICES=6 python -u scripts/audit_v8_fusion_integrity.py \
  --mode jepa \
  --checkpoint runs/searaft_v8_2_rawbase_jepa_full_seed0/last.pt \
  --root datasets/MVSEC \
  --samples 8 \
  --output results/v8_fusion_integrity_jepa.json \
  2>&1 | tee logs/v8_fusion_integrity_jepa.log
```

每个序列均匀取 8 个样本（相同样本的三种预测）。无需重新训练，未更改 checkpoint；不会向之前 `metrics_final.json` 写入数据。

## 3. 如何解释

- `BASE EXACT MATCH exact_match: true`：完整 SEA-RAFT checkpoint 的**所有参数和 BN 缓存**都与原始预训练完全一致。
- `EVENT ENCODER EXACT MATCH` 的 Stage1/Stage2 也是 `exact_match: true`：冻结事件编码器确实未改变。
- `max_abs_zero_vs_raw <= 1e-5`：**gate 置零时恢复原始 SEA-RAFT 的逐像素预测**；不是只比较平均 EPE。
- `metrics.raw.EPE`、`metrics.zero_gate.EPE` 和 `metrics.fusion.EPE` 共同揭示：是否仅仅是融合增量造成性能下降。
- `trained_gate_values`：记录训练后 gate 是否异常放大。gate 的正负或单独值无法证明模型错误，必须结合预测差和 EPE。

若三项审计通过且 fusion EPE > raw EPE，说明在这一实现和评估协议下，训练出的融合确实使光流恶化；没有证据支持再按同样设计扩大光流实验。若审计失败，首先修复具体工程错误，再决定是否需要重新运行任何实验。

**限制**：此审计检验权重冻结与零门控恒等性，不能证明 GT 与 ScaleEvent 官方协议完全相同，也不能排除训练损失/数据分布导致的泛化问题。

## 4. Stage-2 的论文主线

无须启动新训练，优先整理现有 **GEP Stage1-only vs Event-JEPA Stage1+Stage2** 在分类、DSEC-Semantic 和 MVSEC-Depth 下相同解码头和相同监督预算的配对结果。不同任务先以各自指标比较，不应只合并一个平均排名。记录官方 train/test、冻结/全微调、编码器输出模式与是否真正绕过 Stage2。缺任一确认项则先标注“未验证”，不推断正向结果。
