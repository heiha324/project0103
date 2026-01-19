# SEN12MS-CR 数据集划分规则（参考 SEN12MS 官方 splits）

## 依据
SEN12MS-CR 没有单独发布官方 train/val/test 划分。通常沿用 SEN12MS 官方提供的 splits 规则（按 scene/ROI 划分，避免空间泄漏）。

## 划分规则表

| Split | 规则 | 说明 |
| --- | --- | --- |
| train | 使用 `train_list` 中的 patch | `train_list` 已排除了 hold-out scenes |
| test | 使用 `test_list` 中的 patch | `test_list` 来自 hold-out scenes，约 10% patches |
| val | 官方未提供 | 建议从 `train_list` 中划出一部分作为验证集（可随机或按 ROI/scene 分层） |

## 当前项目现状（便于对照）
- 扩散训练已通过 `split_csv` 使用 train split。
- 每个 epoch 的评估使用 test split（来自同一 CSV 划分）。
- 当前 CSV 未包含 val split（val_ratio=0）。

## 本地已生成的划分文件
- 划分方式：按 `ROIs*_s2_cloudy` 目录分组（按场景/ROI 分组），避免同一 ROI 泄漏到不同 split。
- 随机种子：42
- 比例：train 80% / test 20%（val=0，按组划分，四舍五入）
- 输出文件：`splits/sen12mscr_split.csv`
- 生成脚本：`scripts/split_sen12mscr.py`
