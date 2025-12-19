# EMG 动作模式学习与分割系统

基于 EMG (肌电图) 信号的特征驱动分割与分类系统（无预滤波，随机森林为核心，并保留多模型对比）。

## 核心思路

1. **特征学习** - 使用 `src/features` 中的全量特征（时域/频域/小波/AR）。
2. **小样本分割学习** - 由手工 segment 学习受试者级阈值，再在原始信号上分割。
3. **动作分类** - 对检测到的动作进行 amplitude 和 fatigue 分类，Amplitude 全量，Fatigue 仅 full 幅度。
4. **模型对比** - 保留 RandomForest/XGBoost/SVM/KNN/GBDT 对比，自动输出最佳。

## 功能

- **动作幅度分类 (Amplitude)**: full / half / invalid
- **疲劳程度分类 (Fatigue)**: free / light / medium / heavy（仅 full 幅度预测）
- **智能分割**: 基于 RMS 阈值自适应（按受试者→全局），无需滤波
- **模型对比**: 自动评估多种模型，选择最佳

## 使用方法

```bash
pip install -r requirements.txt
python main.py
```

Python 3.10+ 推荐。

## 输出

```
results/
├── amplitude_confusion_matrix.png   # Amplitude 混淆矩阵
├── fatigue_confusion_matrix.png     # Fatigue 混淆矩阵
├── amplitude_model_comparison.png   # Amplitude 模型对比
├── fatigue_model_comparison.png     # Fatigue 模型对比
├── segmentation_*.png               # 带标签的分割结果
└── segmentation_*_plain.png         # 仅分割边界的结果
```

## 数据格式

`train/` 目录下的 `*_segments/` 文件夹包含分割好的动作片段。

文件名: `[amplitude]_[fatigue]_[subject]_[action]_seg[n].csv`

## 依赖

- Python 3.10+
- scikit-learn, XGBoost, NumPy, Pandas, SciPy, Matplotlib, Seaborn
