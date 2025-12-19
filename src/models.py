"""
CNN+Transformer 深度学习模型用于 EMG 信号分类。
使用 CNN 提取局部特征，Transformer 建模时序关系。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math

# 位置编码缩放因子常量
PE_SCALE_FACTOR = 10000.0


class PositionalEncoding(nn.Module):
    """Transformer 位置编码"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(PE_SCALE_FACTOR) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNNTransformerBackbone(nn.Module):
    """
    CNN + Transformer 骨干网络
    - CNN: 提取局部时域特征
    - Transformer: 建模全局时序关系
    """
    
    def __init__(
        self,
        input_channels=1,
        cnn_channels=[32, 64, 128],
        kernel_sizes=[7, 5, 3],
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
        dim_feedforward=256,
        num_classes=3,
        dropout=0.1
    ):
        super().__init__()
        
        # CNN 层：提取局部特征
        cnn_layers = []
        in_ch = input_channels
        for out_ch, ks in zip(cnn_channels, kernel_sizes):
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout)
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        
        # 投影到 Transformer 维度
        self.proj = nn.Linear(cnn_channels[-1], d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # 全局池化 + 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq_len) 或 (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, seq_len)
        
        # CNN 特征提取
        x = self.cnn(x)  # (batch, channels, seq_len')
        
        # 转换维度用于 Transformer
        x = x.transpose(1, 2)  # (batch, seq_len', channels)
        x = self.proj(x)  # (batch, seq_len', d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer 编码
        x = self.transformer(x)  # (batch, seq_len', d_model)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (batch, d_model)
        
        # 分类
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits


class CNNTransformerClassifier:
    """
    CNN+Transformer EMG 信号分类器
    用于识别动作类型 (amplitude) 和疲劳程度 (fatigue)
    """
    
    def __init__(
        self,
        num_classes,
        target_length=4000,
        cnn_channels=[32, 64, 128],
        kernel_sizes=[7, 5, 3],
        d_model=128,
        nhead=4,
        num_transformer_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        device=None
    ):
        """
        初始化分类器
        
        Args:
            num_classes: 分类类别数
            target_length: 目标序列长度（用于统一输入长度）
            cnn_channels: CNN 各层通道数
            kernel_sizes: CNN 各层卷积核大小
            d_model: Transformer 模型维度
            nhead: 注意力头数
            num_transformer_layers: Transformer 层数
            dim_feedforward: FFN 隐藏层维度
            dropout: Dropout 概率
            device: 计算设备
        """
        self.num_classes = num_classes
        self.target_length = target_length
        self.cnn_channels = cnn_channels
        self.kernel_sizes = kernel_sizes
        self.d_model = d_model
        self.nhead = nhead
        self.num_transformer_layers = num_transformer_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = CNNTransformerBackbone(
            input_channels=1,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            d_model=d_model,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)
        
        self.label_to_idx = {}
        self.idx_to_label = {}
    
    def _normalize_length(self, signals):
        """
        统一信号长度
        - 过长则截取
        - 过短则填充
        """
        normalized = []
        for sig in signals:
            if len(sig) > self.target_length:
                # 截取中间部分
                start = (len(sig) - self.target_length) // 2
                normalized.append(sig[start:start + self.target_length])
            elif len(sig) < self.target_length:
                # 边缘填充
                pad_total = self.target_length - len(sig)
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                normalized.append(np.pad(sig, (pad_left, pad_right), mode='edge'))
            else:
                normalized.append(sig)
        return np.array(normalized)
    
    def _normalize_signal(self, signals):
        """信号标准化"""
        normalized = []
        for sig in signals:
            mean = np.mean(sig)
            std = np.std(sig) + 1e-8
            normalized.append((sig - mean) / std)
        return np.array(normalized)
    
    def fit(self, signals, labels, epochs=50, batch_size=16, lr=1e-3, val_split=0.2):
        """
        训练模型
        
        Args:
            signals: list of np.array, EMG 信号列表
            labels: list of str, 标签列表
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
            val_split: 验证集比例
        """
        # 构建标签映射
        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # 预处理信号
        signals = self._normalize_length(signals)
        signals = self._normalize_signal(signals)
        
        # 转换标签
        y = np.array([self.label_to_idx[label] for label in labels])
        
        # 划分训练集和验证集
        n_val = int(len(signals) * val_split)
        indices = np.random.permutation(len(signals))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train = signals[train_indices]
        y_train = y[train_indices]
        X_val = signals[val_indices]
        y_val = y[val_indices]
        
        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 训练循环
        best_val_acc = 0.0
        best_state = None
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * xb.size(0)
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == yb).sum().item()
                train_total += xb.size(0)
            
            scheduler.step()
            
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / train_total
            
            # 验证阶段
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    
                    logits = self.model(xb)
                    _, predicted = torch.max(logits, 1)
                    val_correct += (predicted == yb).sum().item()
                    val_total += xb.size(0)
            
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
        
        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        
        print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.4f}")
        return best_val_acc
    
    def predict(self, signals):
        """
        预测
        
        Args:
            signals: list of np.array, EMG 信号列表
        
        Returns:
            list of str: 预测标签
        """
        self.model.eval()
        
        # 预处理
        signals = self._normalize_length(signals)
        signals = self._normalize_signal(signals)
        
        x_tensor = torch.tensor(signals, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x_tensor)
            _, predicted = torch.max(logits, 1)
            pred_indices = predicted.cpu().numpy()
        
        return [self.idx_to_label[idx] for idx in pred_indices]
    
    def predict_proba(self, signals):
        """
        预测概率
        
        Args:
            signals: list of np.array, EMG 信号列表
        
        Returns:
            np.array: 概率矩阵 (n_samples, n_classes)
        """
        self.model.eval()
        
        # 预处理
        signals = self._normalize_length(signals)
        signals = self._normalize_signal(signals)
        
        x_tensor = torch.tensor(signals, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        
        return proba
    
    def get_classes(self):
        """获取类别标签列表"""
        return [self.idx_to_label[i] for i in range(self.num_classes)]
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'target_length': self.target_length,
            'cnn_channels': self.cnn_channels,
            'kernel_sizes': self.kernel_sizes,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_transformer_layers': self.num_transformer_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label
        }, filepath)
        print(f"模型已保存到 {filepath}")
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.num_classes = checkpoint['num_classes']
        self.target_length = checkpoint['target_length']
        self.cnn_channels = checkpoint['cnn_channels']
        self.kernel_sizes = checkpoint['kernel_sizes']
        self.d_model = checkpoint['d_model']
        self.nhead = checkpoint['nhead']
        self.num_transformer_layers = checkpoint['num_transformer_layers']
        self.dim_feedforward = checkpoint['dim_feedforward']
        self.dropout = checkpoint['dropout']
        self.label_to_idx = checkpoint['label_to_idx']
        self.idx_to_label = checkpoint['idx_to_label']
        
        self.model = CNNTransformerBackbone(
            input_channels=1,
            cnn_channels=self.cnn_channels,
            kernel_sizes=self.kernel_sizes,
            d_model=self.d_model,
            nhead=self.nhead,
            num_transformer_layers=self.num_transformer_layers,
            dim_feedforward=self.dim_feedforward,
            num_classes=self.num_classes,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {filepath} 加载")


class CRNNActivitySegmenter:
    """
    Lightweight placeholder CRNN-style segmenter used solely for
    backward-compatible testing. It exposes a minimal interface and
    returns zero-valued predictions matching the input window shape.
    """
    def __init__(self, sequence_length=200, step_size=50, device="cpu"):
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.device = device
    
    def predict(self, windows):
        """
        Generate zero predictions with the same shape as the input.
        
        Args:
            windows: np.array shaped (batch, sequence_length)
        
        Returns:
            np.array: binary predictions with identical shape
        """
        import numpy as np
        windows = np.asarray(windows)
        return np.zeros_like(windows, dtype=int)
