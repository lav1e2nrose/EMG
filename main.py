"""
EMG 信号动作模式学习与分割系统

核心思路：
1. 学习 segment 片段的动作模式（每种动作的特征模式）
2. 用滑动窗口在原始信号中匹配动作模式
3. 通过模式相似度检测动作边界，而不是简单的阈值
4. 对检测到的动作进行amplitude和fatigue分类

输出：混淆矩阵、分割信号图、评估报告
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, filtfilt, correlate
from scipy.ndimage import maximum_filter1d
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 数据加载 ====================

def load_signal(filepath):
    """加载信号"""
    data = pd.read_csv(filepath, header=None)
    return data.values.astype(float).flatten()


def parse_filename(filename):
    """解析文件名"""
    basename = os.path.basename(filename).replace('.csv', '')
    parts = basename.split('_')
    if len(parts) >= 4:
        return {'amplitude': parts[0], 'fatigue': parts[1], 'subject': parts[2]}
    return None


def load_all_segments(train_dir):
    """加载所有segment"""
    segments, labels_amp, labels_fat, subjects = [], [], [], []
    
    for item in os.listdir(train_dir):
        item_path = os.path.join(train_dir, item)
        if os.path.isdir(item_path) and item.endswith('_segments'):
            for seg_file in sorted(os.listdir(item_path)):
                if seg_file.endswith('.csv'):
                    seg_path = os.path.join(item_path, seg_file)
                    meta = parse_filename(seg_file)
                    if meta:
                        signal = load_signal(seg_path)
                        segments.append(signal)
                        labels_amp.append(meta['amplitude'])
                        labels_fat.append(meta['fatigue'])
                        subjects.append(meta['subject'])
    
    print(f"加载 {len(segments)} 个 segment")
    print(f"Amplitude: {pd.Series(labels_amp).value_counts().to_dict()}")
    print(f"Fatigue: {pd.Series(labels_fat).value_counts().to_dict()}")
    return segments, labels_amp, labels_fat, subjects


def get_raw_files(train_dir):
    """获取原始信号文件"""
    raw_files = []
    for item in os.listdir(train_dir):
        if item.endswith('.csv') and '_seg' not in item:
            item_path = os.path.join(train_dir, item)
            if os.path.isfile(item_path):
                raw_files.append(item_path)
    return sorted(raw_files)


# ==================== 信号处理 ====================

def bandpass_filter(signal, fs=2000, lowcut=20, highcut=450):
    """带通滤波"""
    nyq = fs / 2
    b, a = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)


def normalize_signal(signal):
    """标准化"""
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def resize_signal(signal, target_length):
    """调整信号长度（插值或截取）"""
    if len(signal) == target_length:
        return signal
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, signal)


# ==================== 特征提取 ====================

def extract_features(signal):
    """提取特征向量"""
    feats = []
    
    # 时域
    feats.append(np.sqrt(np.mean(signal**2)))  # RMS
    feats.append(np.mean(np.abs(signal)))  # MAV
    feats.append(np.var(signal))  # VAR
    feats.append(np.max(signal) - np.min(signal))  # Peak-to-peak
    feats.append(np.sum(np.abs(np.diff(np.sign(signal))) > 0))  # ZC
    feats.append(np.sum(np.abs(np.diff(signal))))  # WL
    
    # 频域
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/2000)
    power = np.abs(fft)**2
    feats.append(np.sum(freqs * power) / (np.sum(power) + 1e-10))  # Mean freq
    
    # 分段特征（将信号分成4段，每段提取RMS）
    n_parts = 4
    part_len = len(signal) // n_parts
    for i in range(n_parts):
        part = signal[i*part_len:(i+1)*part_len]
        feats.append(np.sqrt(np.mean(part**2)))
    
    return np.array(feats)


def extract_all_features(signals):
    """批量提取特征"""
    return np.array([extract_features(s) for s in signals])


# ==================== 深度学习模型 ====================

class CNNClassifier(nn.Module):
    """CNN分类器"""
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, num_classes))
    
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.fc(self.conv(x).squeeze(-1))


class CNNTransformer(nn.Module):
    """CNN + Transformer"""
    def __init__(self, num_classes, d_model=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, d_model, 5, padding=2), nn.BatchNorm1d(d_model), nn.ReLU(), nn.MaxPool1d(4)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, num_classes))
    
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.conv(x).transpose(1, 2)
        return self.fc(self.transformer(x).mean(dim=1))


class ActionPatternDetector(nn.Module):
    """
    动作模式检测器
    学习每种动作的模式，输出：
    1. 是否是有效动作 (action vs background)
    2. 动作类型 (amplitude)
    """
    def __init__(self, num_amp_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc_shared = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        # 动作检测头：是否是有效动作
        self.fc_action = nn.Linear(64, 2)  # [background, action]
        # 动作分类头：amplitude类型
        self.fc_amp = nn.Linear(64, num_amp_classes)
    
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        feat = self.fc_shared(self.conv(x).squeeze(-1))
        return self.fc_action(feat), self.fc_amp(feat)
    
    def get_features(self, x):
        """获取特征向量，用于模式匹配"""
        if x.dim() == 2: x = x.unsqueeze(1)
        return self.fc_shared(self.conv(x).squeeze(-1))


# ==================== 模型训练 ====================

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=1e-3, device='cpu'):
    """训练模型"""
    model = model.to(device)
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    ), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    ), batch_size=16)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_acc, best_state = 0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, pred = torch.max(model(xb), 1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_acc


def train_pattern_detector(detector, segments, labels_amp, epochs=50, device='cpu'):
    """
    训练动作模式检测器
    - 学习动作模式（正样本：真实动作）
    - 学习背景模式（负样本：随机噪声/静止段）
    """
    # 准备正样本（真实动作）
    target_len = 2000  # 统一长度
    X_action = np.array([resize_signal(normalize_signal(s), target_len) for s in segments])
    
    # 生成负样本（背景/噪声）
    n_neg = len(segments)
    X_background = []
    for _ in range(n_neg):
        # 生成低幅度噪声作为背景
        noise = np.random.randn(target_len) * 0.1
        X_background.append(noise)
    X_background = np.array(X_background)
    
    # 合并数据
    X_all = np.vstack([X_action, X_background])
    y_action = np.array([1] * len(X_action) + [0] * len(X_background))  # 1=action, 0=background
    
    # amplitude标签（背景设为-1，训练时忽略）
    le = LabelEncoder()
    y_amp_encoded = le.fit_transform(labels_amp)
    y_amp_all = np.concatenate([y_amp_encoded, np.full(n_neg, -1)])
    
    # 划分数据
    indices = np.random.permutation(len(X_all))
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_action_train, y_action_val = y_action[train_idx], y_action[val_idx]
    y_amp_train, y_amp_val = y_amp_all[train_idx], y_amp_all[val_idx]
    
    # 训练
    detector = detector.to(device)
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_action_train, dtype=torch.long),
        torch.tensor(y_amp_train, dtype=torch.long)
    ), batch_size=16, shuffle=True)
    
    criterion_action = nn.CrossEntropyLoss()
    criterion_amp = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(detector.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(epochs):
        detector.train()
        for xb, yb_action, yb_amp in train_loader:
            xb = xb.to(device)
            yb_action = yb_action.to(device)
            yb_amp = yb_amp.to(device)
            
            optimizer.zero_grad()
            out_action, out_amp = detector(xb)
            loss = criterion_action(out_action, yb_action) + criterion_amp(out_amp, yb_amp)
            loss.backward()
            optimizer.step()
    
    # 验证
    detector.eval()
    with torch.no_grad():
        x_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        out_action, _ = detector(x_val_tensor)
        _, pred = torch.max(out_action, 1)
        acc = (pred.cpu().numpy() == y_action_val).mean()
        print(f"动作检测验证准确率: {acc:.4f}")
    
    return detector, le, target_len


# ==================== 动作模式分割 ====================

def detect_actions_with_pattern(detector, signal, target_len, window_step=500, 
                                threshold=0.7, min_gap=1000, device='cpu'):
    """
    使用学习到的动作模式检测信号中的动作
    
    原理：
    1. 滑动窗口扫描信号
    2. 用训练好的检测器判断每个窗口是否包含动作
    3. 根据动作概率找到动作位置
    4. 合并重叠检测，精确定位边界
    
    返回：segments 列表，不包含 amplitude 预测（由分类器单独预测）
    """
    signal_norm = normalize_signal(signal)
    detector.eval()
    
    # 滑动窗口检测
    detections = []  # [(center_pos, action_prob, start, end)]
    
    for start in range(0, len(signal) - target_len + 1, window_step):
        window = signal_norm[start:start + target_len]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out_action, _ = detector(x)
            action_prob = torch.softmax(out_action, dim=1)[0, 1].item()  # P(action)
        
        center = start + target_len // 2
        detections.append((center, action_prob, start, start + target_len))
    
    # 找到高置信度的检测
    high_conf = [(c, p, s, e) for c, p, s, e in detections if p > threshold]
    
    if not high_conf:
        # 降低阈值再试
        threshold = max(0.5, threshold - 0.2)
        high_conf = [(c, p, s, e) for c, p, s, e in detections if p > threshold]
    
    if not high_conf:
        return []
    
    # 非极大值抑制：合并重叠的检测
    high_conf = sorted(high_conf, key=lambda x: -x[1])  # 按概率降序
    segments = []
    
    for center, prob, start, end in high_conf:
        # 检查是否与已有segment重叠
        overlap = False
        for seg_start, seg_end in segments:
            if not (end < seg_start or start > seg_end):
                overlap = True
                break
        
        if not overlap:
            # 精确化边界：在检测区域内找能量变化点
            refined_start, refined_end = refine_segment_boundaries(
                signal_norm, start, end, target_len
            )
            segments.append((refined_start, refined_end))
    
    # 按时间顺序排序
    segments = sorted(segments, key=lambda x: x[0])
    
    return segments


def refine_segment_boundaries(signal, approx_start, approx_end, target_len):
    """
    精确化segment边界
    在检测到的大致区域内，根据信号能量变化精确找到动作的起止点
    """
    # 扩展搜索范围
    search_margin = target_len // 4
    search_start = max(0, approx_start - search_margin)
    search_end = min(len(signal), approx_end + search_margin)
    
    segment = signal[search_start:search_end]
    
    # 计算能量包络
    window_size = 100
    energy = np.array([np.mean(segment[max(0,i-window_size//2):min(len(segment),i+window_size//2)]**2) 
                       for i in range(len(segment))])
    
    # 找能量阈值
    threshold = np.mean(energy) + 0.5 * np.std(energy)
    
    # 找起点（从左往右第一个超过阈值的点）
    above = energy > threshold
    start_idx = 0
    for i, v in enumerate(above):
        if v:
            start_idx = i
            break
    
    # 找终点（从右往左第一个超过阈值的点）
    end_idx = len(segment) - 1
    for i in range(len(above) - 1, -1, -1):
        if above[i]:
            end_idx = i
            break
    
    refined_start = search_start + start_idx
    refined_end = search_start + end_idx
    
    # 确保合理长度
    min_len = target_len // 4
    if refined_end - refined_start < min_len:
        refined_start = approx_start
        refined_end = approx_end
    
    return refined_start, refined_end


# ==================== 分类器评估 ====================

def evaluate_classifiers(X_raw, X_feat, y, task_name, target_len):
    """评估多种分类器"""
    print(f"\n{'='*60}")
    print(f"任务: {task_name}")
    print(f"{'='*60}")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # 准备数据
    X_raw_resized = np.array([resize_signal(normalize_signal(s), target_len) for s in X_raw])
    
    X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X_raw_resized, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    X_feat_train, X_feat_test = X_feat[np.isin(y_enc, y_train)], X_feat[np.isin(y_enc, y_test)]
    
    # 重新划分特征
    X_feat_train, X_feat_test, _, _ = train_test_split(
        X_feat, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    scaler = StandardScaler()
    X_feat_train_s = scaler.fit_transform(X_feat_train)
    X_feat_test_s = scaler.transform(X_feat_test)
    
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = len(le.classes_)
    
    # ML模型
    print("\n--- 机器学习模型 ---")
    ml_models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, random_state=42, verbosity=0),
        'SVM': SVC(kernel='rbf', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in ml_models.items():
        model.fit(X_feat_train_s, y_train)
        pred = model.predict(X_feat_test_s)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='weighted')
        results[name] = {'acc': acc, 'f1': f1, 'pred': pred, 'model': model, 'type': 'ml'}
        print(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}")
    
    # DL模型
    print("\n--- 深度学习模型 ---")
    X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(
        X_raw_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    dl_models = {
        'CNN': CNNClassifier(num_classes),
        'CNN+Transformer': CNNTransformer(num_classes)
    }
    
    for name, model in dl_models.items():
        trained, val_acc = train_model(model, X_train_dl, y_train_dl, X_val_dl, y_val_dl, epochs=50, device=device)
        trained.eval()
        with torch.no_grad():
            x_test = torch.tensor(X_raw_test, dtype=torch.float32).to(device)
            _, pred = torch.max(trained(x_test), 1)
            pred = pred.cpu().numpy()
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='weighted')
        results[name] = {'acc': acc, 'f1': f1, 'pred': pred, 'model': trained, 'type': 'dl'}
        print(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}")
    
    best = max(results, key=lambda k: results[k]['f1'])
    print(f"\n最佳: {best} (F1={results[best]['f1']:.4f})")
    
    return {'best': best, 'results': results, 'y_test': y_test, 'le': le, 'scaler': scaler}


# ==================== 可视化 ====================

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    """混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"保存: {save_path}")


def plot_segments(segments, labels, save_path, n_per_class=3):
    """绘制segment示例"""
    unique = sorted(set(labels))
    fig, axes = plt.subplots(len(unique), n_per_class, figsize=(15, 3*len(unique)))
    if len(unique) == 1:
        axes = axes.reshape(1, -1)
    
    for i, label in enumerate(unique):
        idxs = [j for j, l in enumerate(labels) if l == label][:n_per_class]
        for j, idx in enumerate(idxs):
            ax = axes[i, j]
            t = np.arange(len(segments[idx])) / 2000
            ax.plot(t, segments[idx], lw=0.5)
            ax.set_title(f'{label} #{j+1}')
            ax.set_xlabel('Time (s)')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"保存: {save_path}")


def plot_model_comparison(results, title, save_path):
    """模型对比"""
    names = list(results.keys())
    accs = [results[n]['acc'] for n in names]
    f1s = [results[n]['f1'] for n in names]
    
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 0.2, accs, 0.4, label='Accuracy', color='steelblue')
    ax.bar(x + 0.2, f1s, 0.4, label='F1', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"保存: {save_path}")


def plot_segmentation(signal, segments, amp_preds, fat_preds, amp_le, fat_le, save_path, title):
    """绘制分割结果"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    t = np.arange(len(signal)) / 2000
    
    # 信号 + 分割区域
    ax1 = axes[0]
    ax1.plot(t, signal, 'b-', lw=0.5, alpha=0.7)
    colors = {'full': 'green', 'half': 'orange', 'invalid': 'red'}
    
    for i, (start, end) in enumerate(segments):
        amp = amp_le.inverse_transform([amp_preds[i]])[0] if i < len(amp_preds) else 'unknown'
        ax1.axvspan(start/2000, end/2000, alpha=0.3, color=colors.get(amp, 'gray'))
        ax1.axvline(start/2000, color='r', ls='--', lw=0.5)
        ax1.axvline(end/2000, color='r', ls='--', lw=0.5)
    
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'{title} - Detected Actions')
    ax1.grid(True, alpha=0.3)
    
    # 分类结果时间线
    ax2 = axes[1]
    fat_colors = {'free': 'lightgreen', 'light': 'yellow', 'medium': 'orange', 'heavy': 'red'}
    
    for i, (start, end) in enumerate(segments):
        amp = amp_le.inverse_transform([amp_preds[i]])[0] if i < len(amp_preds) else '?'
        fat = fat_le.inverse_transform([fat_preds[i]])[0] if i < len(fat_preds) else '?'
        
        ax2.barh(1, (end-start)/2000, left=start/2000, height=0.3, color=colors.get(amp, 'gray'), alpha=0.7)
        ax2.text((start+end)/2/2000, 1, amp, ha='center', va='center', fontsize=8)
        
        ax2.barh(0, (end-start)/2000, left=start/2000, height=0.3, color=fat_colors.get(fat, 'gray'), alpha=0.7)
        ax2.text((start+end)/2/2000, 0, fat, ha='center', va='center', fontsize=8)
    
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Fatigue', 'Amplitude'])
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Classification Results')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"保存: {save_path}")


# ==================== 主函数 ====================

def main():
    print("="*60)
    print("EMG 动作模式学习与分割系统")
    print("="*60)
    
    TRAIN_DIR = 'train'
    TARGET_LEN = 2000
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # ========== 阶段1: 学习动作模式 ==========
    print("\n" + "="*60)
    print("阶段1: 学习 segment 动作模式")
    print("="*60)
    
    segments, labels_amp, labels_fat, subjects = load_all_segments(TRAIN_DIR)
    segments_filtered = [bandpass_filter(s) for s in segments]
    features = extract_all_features(segments_filtered)
    
    # 可视化segments
    plot_segments(segments_filtered, labels_amp, 'results/amplitude_segments.png')
    plot_segments(segments_filtered, labels_fat, 'results/fatigue_segments.png')
    
    # 训练动作模式检测器
    print("\n[训练动作模式检测器...]")
    detector = ActionPatternDetector(num_amp_classes=len(set(labels_amp)))
    detector, amp_le, target_len = train_pattern_detector(detector, segments_filtered, labels_amp, epochs=50, device=device)
    
    # 评估Amplitude分类
    print("\n[评估Amplitude分类器...]")
    amp_eval = evaluate_classifiers(segments_filtered, features, labels_amp, "Amplitude", TARGET_LEN)
    plot_confusion_matrix(
        amp_eval['y_test'], amp_eval['results'][amp_eval['best']]['pred'],
        amp_eval['le'].classes_, f"Amplitude ({amp_eval['best']})", 'results/amplitude_confusion_matrix.png'
    )
    plot_model_comparison(amp_eval['results'], "Amplitude Models", 'results/amplitude_model_comparison.png')
    
    # 评估Fatigue分类
    print("\n[评估Fatigue分类器...]")
    full_idx = [i for i, a in enumerate(labels_amp) if a == 'full']
    segs_full = [segments_filtered[i] for i in full_idx]
    feats_full = features[full_idx]
    labels_fat_full = [labels_fat[i] for i in full_idx]
    
    fat_eval = evaluate_classifiers(segs_full, feats_full, labels_fat_full, "Fatigue", TARGET_LEN)
    plot_confusion_matrix(
        fat_eval['y_test'], fat_eval['results'][fat_eval['best']]['pred'],
        fat_eval['le'].classes_, f"Fatigue ({fat_eval['best']})", 'results/fatigue_confusion_matrix.png'
    )
    plot_model_comparison(fat_eval['results'], "Fatigue Models", 'results/fatigue_model_comparison.png')
    
    # ========== 阶段2: 分割原始信号 ==========
    print("\n" + "="*60)
    print("阶段2: 使用动作模式分割原始信号")
    print("="*60)
    
    raw_files = get_raw_files(TRAIN_DIR)
    print(f"找到 {len(raw_files)} 个原始信号")
    
    # 获取分类器
    amp_model = amp_eval['results'][amp_eval['best']]['model']
    amp_scaler = amp_eval['scaler']
    amp_le = amp_eval['le']
    amp_type = amp_eval['results'][amp_eval['best']]['type']
    
    fat_model = fat_eval['results'][fat_eval['best']]['model']
    fat_scaler = fat_eval['scaler']
    fat_le = fat_eval['le']
    fat_type = fat_eval['results'][fat_eval['best']]['type']
    
    for i, raw_file in enumerate(raw_files[:5]):
        print(f"\n处理: {os.path.basename(raw_file)}")
        signal = load_signal(raw_file)
        signal_filtered = bandpass_filter(signal)
        
        # 用动作模式检测分割
        segments_detected = detect_actions_with_pattern(
            detector, signal_filtered, target_len, device=device
        )
        
        print(f"  检测到 {len(segments_detected)} 个动作")
        
        # 对每个检测到的动作预测 amplitude 和 fatigue
        amp_preds = []
        fat_preds = []
        
        for start, end in segments_detected:
            seg = signal_filtered[start:end]
            feat = extract_features(seg).reshape(1, -1)
            
            # 预测 amplitude
            if amp_type == 'ml':
                feat_s = amp_scaler.transform(feat)
                amp_pred = amp_model.predict(feat_s)[0]
            else:
                seg_resized = resize_signal(normalize_signal(seg), TARGET_LEN)
                x = torch.tensor(seg_resized, dtype=torch.float32).unsqueeze(0).to(device)
                amp_model.eval()
                with torch.no_grad():
                    _, pred = torch.max(amp_model(x), 1)
                    amp_pred = pred.cpu().numpy()[0]
            amp_preds.append(amp_pred)
            
            # 预测 fatigue
            if fat_type == 'ml':
                feat_s = fat_scaler.transform(feat)
                fat_pred = fat_model.predict(feat_s)[0]
            else:
                seg_resized = resize_signal(normalize_signal(seg), TARGET_LEN)
                x = torch.tensor(seg_resized, dtype=torch.float32).unsqueeze(0).to(device)
                fat_model.eval()
                with torch.no_grad():
                    _, pred = torch.max(fat_model(x), 1)
                    fat_pred = pred.cpu().numpy()[0]
            fat_preds.append(fat_pred)
        
        for j, (start, end) in enumerate(segments_detected):
            amp = amp_le.inverse_transform([amp_preds[j]])[0]
            fat = fat_le.inverse_transform([fat_preds[j]])[0]
            print(f"    [{start/2000:.2f}s - {end/2000:.2f}s] Amp={amp}, Fat={fat}")
        
        # 绘制分割结果
        save_path = f'results/segmentation_{os.path.basename(raw_file).replace(".csv", ".png")}'
        plot_segmentation(signal_filtered, segments_detected, amp_preds, fat_preds, 
                         amp_le, fat_le, save_path, os.path.basename(raw_file))
    
    # ========== 总结 ==========
    print("\n" + "="*60)
    print("完成!")
    print("="*60)
    print(f"\nAmplitude最佳: {amp_eval['best']} (F1={amp_eval['results'][amp_eval['best']]['f1']:.4f})")
    print(f"Fatigue最佳: {fat_eval['best']} (F1={fat_eval['results'][fat_eval['best']]['f1']:.4f})")
    print("\n输出文件:")
    print("  results/amplitude_confusion_matrix.png")
    print("  results/fatigue_confusion_matrix.png")
    print("  results/amplitude_segments.png")
    print("  results/fatigue_segments.png")
    print("  results/amplitude_model_comparison.png")
    print("  results/fatigue_model_comparison.png")
    print("  results/segmentation_*.png")


if __name__ == '__main__':
    main()
