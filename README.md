# EMG 动作模式学习与分割系统

基于 EMG (肌电图) 信号的动作模式学习和智能分割系统。

## 核心思路

1. **学习动作模式** - 从 segment 片段学习每种动作的特征模式
2. **模式匹配分割** - 用学到的模式在原始信号中检测动作（不是简单阈值）
3. **动作分类** - 对检测到的动作进行 amplitude 和 fatigue 分类

## 功能

- **动作幅度分类 (Amplitude)**: full / half / invalid
- **疲劳程度分类 (Fatigue)**: free / light / medium / heavy
- **智能分割**: 基于学习的动作模式检测，而非简单阈值
- **模型对比**: 自动评估 7 种模型，选择最佳

## 模型

### 深度学习
- **CNN**: 1D 卷积神经网络
- **CNN+Transformer**: 混合架构
- **ActionPatternDetector**: 动作模式检测器

### 机器学习
- Random Forest
- XGBoost
- SVM
- KNN
- Gradient Boosting

## 使用方法

```bash
pip install -r requirements.txt
python main.py
```

## 输出

```
results/
├── amplitude_confusion_matrix.png   # Amplitude 混淆矩阵
├── fatigue_confusion_matrix.png     # Fatigue 混淆矩阵
├── amplitude_segments.png           # Amplitude segment 示例
├── fatigue_segments.png             # Fatigue segment 示例
├── amplitude_model_comparison.png   # Amplitude 模型对比
├── fatigue_model_comparison.png     # Fatigue 模型对比
└── segmentation_*.png               # 原始信号分割结果
```

## 数据格式

`train/` 目录下的 `*_segments/` 文件夹包含分割好的动作片段。

文件名: `[amplitude]_[fatigue]_[subject]_[action]_seg[n].csv`

## 依赖

- Python 3.8+
- PyTorch >= 2.0
- scikit-learn, XGBoost
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn