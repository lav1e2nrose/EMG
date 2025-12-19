"""
EMG 信号分割与分类主程序（特征驱动、无预滤波）。

目标：
1. 基于手工分割的少量样本学习分割阈值（按受试者优先，其次全局）。
2. 使用特征工程（src/features 中的全量特征）+ 随机森林为核心，并保持多模型对比。
3. 直接在原始信号上分割，不做带通/陷波滤波。
4. 对分割出的动作分别预测动作幅度（full/half/invalid）和疲劳程度（free/light/medium/heavy）。
"""
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# 确保可以直接导入 src 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.features import extract_segment_features
from src.preprocessing import (
    DEFAULT_RMS_WINDOW_MS,
    compute_rms_envelope,
    detect_activity_regions,
    load_emg_data,
)
from src.utils import (
    get_train_files,
    load_segments_metadata,
    parse_filename,
    plot_confusion_matrix,
    plot_signal_with_segments,
    validate_labels,
)

# 全局参数
FS = 2000
RMS_WINDOW_MS = DEFAULT_RMS_WINDOW_MS
PEAK_PERCENTILE_PROFILE = 90
PEAK_PERCENTILE_DETECTION = 95
THRESHOLD_MULTIPLIER_MIN = 0.5
THRESHOLD_MULTIPLIER_MAX = 1.5
SMALL_EPS = 1e-8
TRAIN_DIR = "train"
TEST_DIR = "test"


def build_segment_dataset(segment_metadata):
    """从标注好的 segment 构建特征矩阵和标签。"""
    features = []
    amp_labels = []
    fat_labels = []
    subjects = []
    raw_segments = []
    feature_names = None

    for meta in segment_metadata:
        if not validate_labels(meta):
            continue
        sig = load_emg_data(meta["path"])
        feats = extract_segment_features(sig, fs=FS)
        if feature_names is None:
            feature_names = list(feats.keys())
        features.append([feats[name] for name in feature_names])
        amp_labels.append(meta["amplitude"])
        fat_labels.append(meta["fatigue"])
        subjects.append(meta["subject_id"])
        raw_segments.append(sig)

    return (
        np.array(features),
        amp_labels,
        fat_labels,
        subjects,
        raw_segments,
        feature_names or [],
    )


def evaluate_ml_models(X, y, task_name):
    """使用特征矩阵评估多模型（核心 RandomForest），返回最佳模型与对比结果。"""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, random_state=42, verbosity=0
        ),
        "SVM": SVC(kernel="rbf", probability=False, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="weighted")
        results[name] = {"acc": acc, "f1": f1, "pred": pred, "model": model}
        print(f"[{task_name}] {name}: acc={acc:.4f}, f1={f1:.4f}")

    best = max(results, key=lambda k: results[k]["f1"])
    best_pred_labels = le.inverse_transform(results[best]["pred"])

    return {
        "best": best,
        "results": results,
        "label_encoder": le,
        "scaler": scaler,
        "y_test": y_test,
        "y_test_labels": le.inverse_transform(y_test),
        "best_pred_labels": best_pred_labels,
    }


def plot_model_comparison(results, title, save_path):
    """生成模型对比柱状图。"""
    names = list(results.keys())
    accs = [results[n]["acc"] for n in names]
    f1s = [results[n]["f1"] for n in names]
    x = np.arange(len(names))

    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.2, accs, width=0.4, label="Accuracy")
    plt.bar(x + 0.2, f1s, width=0.4, label="F1")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylim(0, 1.1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"模型对比保存: {save_path}")


def build_segmentation_profiles(segments, subjects):
    """基于已分割片段学习每个受试者的 RMS 峰值特征，用于自适应阈值。"""
    peaks = defaultdict(list)
    window = int(RMS_WINDOW_MS * FS / 1000)
    for sig, subj in zip(segments, subjects):
        env = compute_rms_envelope(sig, window, FS)
        peaks[subj].append(np.percentile(env, PEAK_PERCENTILE_PROFILE))

    subject_profiles = {s: float(np.median(vals)) for s, vals in peaks.items() if len(vals) > 0}
    all_peaks = [p for vals in peaks.values() for p in vals]
    global_peak = float(np.median(all_peaks)) if all_peaks else 1.0
    return subject_profiles, global_peak


def segment_raw_signal(raw_signal, subject_id, subject_profiles, global_peak):
    """在原始信号上进行分割（无滤波），使用受试者优先的阈值调整。"""
    env = compute_rms_envelope(raw_signal, int(RMS_WINDOW_MS * FS / 1000), FS)
    env_peak = np.percentile(env, PEAK_PERCENTILE_DETECTION) + SMALL_EPS
    ref_peak = subject_profiles.get(subject_id, global_peak)
    multiplier = np.clip(
        ref_peak / env_peak, THRESHOLD_MULTIPLIER_MIN, THRESHOLD_MULTIPLIER_MAX
    )

    segments = detect_activity_regions(
        raw_signal,
        fs=FS,
        rms_window_ms=RMS_WINDOW_MS,
        min_segment_ms=300,
        merge_gap_ms=250,
        threshold_multiplier=multiplier,
    )
    return segments


def classify_segments(signal, segments, feature_names, amp_eval, fat_eval):
    """对分割出的片段做动作幅度与疲劳预测。"""
    amp_model = amp_eval["results"][amp_eval["best"]]["model"]
    amp_scaler = amp_eval["scaler"]
    amp_le = amp_eval["label_encoder"]

    fat_model = fat_eval["results"][fat_eval["best"]]["model"] if fat_eval else None
    fat_scaler = fat_eval["scaler"] if fat_eval else None
    fat_le = fat_eval["label_encoder"] if fat_eval else None

    amp_preds = []
    fat_preds = []

    for start, end in segments:
        seg = signal[start:end]
        feats = extract_segment_features(seg, fs=FS)
        vec = np.array([feats[name] for name in feature_names]).reshape(1, -1)

        amp_pred_idx = amp_model.predict(amp_scaler.transform(vec))[0]
        amp_label = amp_le.inverse_transform([amp_pred_idx])[0]
        amp_preds.append(amp_label)

        if fat_model is not None and amp_label == "full":
            fat_pred_idx = fat_model.predict(fat_scaler.transform(vec))[0]
            fat_label = fat_le.inverse_transform([fat_pred_idx])[0]
        else:
            fat_label = "free"
        fat_preds.append(fat_label)

    return amp_preds, fat_preds


def plot_labeled_segments(signal, segments, amp_labels, fat_labels, save_path):
    """绘制带标签的分割结果。"""
    t = np.arange(len(signal)) / FS
    plt.figure(figsize=(15, 6))
    plt.plot(t, signal, lw=0.5, alpha=0.7, label="EMG")

    for (start, end), amp, fat in zip(segments, amp_labels, fat_labels):
        plt.axvspan(start / FS, end / FS, alpha=0.3, color="lightblue")
        plt.text((start + end) / 2 / FS, np.max(signal) * 0.8, f"{amp}/{fat}", ha="center", fontsize=8)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Segmentation with Predictions")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"分割可视化保存: {save_path}")


def main():
    os.makedirs("results", exist_ok=True)

    # 1) 加载训练片段并提取特征
    train_files = get_train_files(TRAIN_DIR)
    segment_metadata = []
    for seg_dir in train_files["segment_dirs"].values():
        segment_metadata.extend(load_segments_metadata(seg_dir))

    if not segment_metadata:
        print("未找到标注的 segment 数据，无法训练。")
        return

    (
        feature_matrix,
        amp_labels,
        fat_labels,
        subjects,
        raw_segments,
        feature_names,
    ) = build_segment_dataset(segment_metadata)

    # 2) 模型对比与训练（Amplitude）
    print("\n=== 动作幅度分类（特征 + 多模型对比） ===")
    amp_eval = evaluate_ml_models(feature_matrix, amp_labels, "Amplitude")
    plot_confusion_matrix(
        amp_eval["y_test_labels"],
        amp_eval["best_pred_labels"],
        list(amp_eval["label_encoder"].classes_),
        title="Amplitude Confusion Matrix",
        save_path="results/amplitude_confusion_matrix.png",
    )
    plot_model_comparison(
        amp_eval["results"],
        "Amplitude Model Comparison",
        "results/amplitude_model_comparison.png",
    )

    # 3) 疲劳分类仅使用 full 幅度
    print("\n=== 疲劳程度分类（仅 full 幅度） ===")
    full_idx = [i for i, a in enumerate(amp_labels) if a == "full"]
    fat_eval = None
    if full_idx:
        fat_feature_matrix = feature_matrix[full_idx]
        fat_subset_labels = [fat_labels[i] for i in full_idx]
        fat_eval = evaluate_ml_models(fat_feature_matrix, fat_subset_labels, "Fatigue")
        plot_confusion_matrix(
            fat_eval["y_test_labels"],
            fat_eval["best_pred_labels"],
            list(fat_eval["label_encoder"].classes_),
            title="Fatigue Confusion Matrix",
            save_path="results/fatigue_confusion_matrix.png",
        )
        plot_model_comparison(
            fat_eval["results"],
            "Fatigue Model Comparison",
            "results/fatigue_model_comparison.png",
        )
    else:
        print("没有 full 幅度的训练样本，跳过疲劳分类。")

    # 4) 基于小样本学习分割阈值
    subject_profiles, global_peak = build_segmentation_profiles(raw_segments, subjects)

    # 5) 对原始文件（train + test）进行分割与预测
    raw_targets = list(train_files["raw_files"])
    if os.path.isdir(TEST_DIR):
        for item in os.listdir(TEST_DIR):
            if item.endswith(".csv"):
                raw_targets.append(os.path.join(TEST_DIR, item))

    for raw_path in raw_targets:
        raw_signal = load_emg_data(raw_path)
        meta = parse_filename(os.path.basename(raw_path)) or {}
        subject_id = meta.get("subject_id")

        segments = segment_raw_signal(raw_signal, subject_id, subject_profiles, global_peak)
        amp_preds, fat_preds = classify_segments(raw_signal, segments, feature_names, amp_eval, fat_eval)

        print(f"\n文件 {os.path.basename(raw_path)} 检测到 {len(segments)} 个动作:")
        for (s, e), a, f in zip(segments, amp_preds, fat_preds):
            print(f"  [{s/FS:.2f}s - {e/FS:.2f}s] amplitude={a}, fatigue={f}")

        save_path = os.path.join(
            "results", f"segmentation_{os.path.basename(raw_path).replace('.csv', '.png')}"
        )
        plot_labeled_segments(raw_signal, segments, amp_preds, fat_preds, save_path)
        # 兼容原有函数输出一份简单标注图
        plot_signal_with_segments(
            raw_signal,
            segments,
            sampling_rate=FS,
            title=f"Segments for {os.path.basename(raw_path)}",
            save_path=save_path.replace(".png", "_plain.png"),
        )

    print("\n=== 完成 ===")
    print("运行环境建议：Python 3.10+，已在无滤波模式下工作。")
    print("主要输出位于 results/ 目录。")


if __name__ == "__main__":
    main()
