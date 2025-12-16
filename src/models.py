"""
Model wrappers for EMG signal classification.
Provides interfaces for training and prediction with sklearn models.
"""
import numpy as np
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class ActivityDetector:
    """
    Binary classifier to detect active/inactive segments in EMG signal.
    Uses sliding window features to predict activity.
    """
    
    def __init__(self, model_type='random_forest', decision_threshold=0.6, **kwargs):
        """
        Initialize activity detector.
        
        Args:
            model_type: str, 'random_forest' or 'xgboost'
            decision_threshold: float, probability threshold for classifying
                                a window as active (0=inactive, 1=active).
            **kwargs: additional parameters for the model
        """
        self.model_type = model_type
        self.decision_threshold = decision_threshold
        
        if model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }
            default_params.update(kwargs)
            self.model = RandomForestClassifier(**default_params)
        
        elif model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(kwargs)
            self.model = XGBClassifier(**default_params)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X, y):
        """
        Train the activity detector.
        
        Args:
            X: np.array, feature matrix (n_samples, n_features)
            y: np.array, binary labels (0=inactive, 1=active)
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Predict activity for given features.
        
        Args:
            X: np.array, feature matrix (n_samples, n_features)
        
        Returns:
            np.array: binary predictions (0=inactive, 1=active)
        """
        proba = self.model.predict_proba(X)
        
        class_labels = None
        active_idx = None
        
        if hasattr(self.model, "classes_"):
            class_labels = np.array(self.model.classes_)
            active_positions = np.where(class_labels == 1)[0]
            if active_positions.size == 1:
                active_idx = int(active_positions[0])
            elif active_positions.size > 1:
                warnings.warn(
                    "Multiple occurrences of active class label '1' found; using the first.",
                    RuntimeWarning,
                )
                active_idx = int(active_positions[0])
            else:
                warnings.warn(
                    "Active class label '1' not found; falling back to the last probability column.",
                    RuntimeWarning,
                )
        
        if active_idx is None:
            active_idx = proba.shape[1] - 1 if proba.shape[1] > 1 else 0
        
        if self.decision_threshold is None:
            pred_indices = np.argmax(proba, axis=1)
            if class_labels is not None:
                predicted_labels = class_labels[pred_indices]
                return (predicted_labels == 1).astype(int)
            return pred_indices.astype(int)
        
        return (proba[:, active_idx] >= self.decision_threshold).astype(int)
    
    def predict_proba(self, X):
        """
        Predict probability of activity.
        
        Args:
            X: np.array, feature matrix (n_samples, n_features)
        
        Returns:
            np.array: probability matrix (n_samples, 2)
        """
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")


class AmplitudeClassifier:
    """
    Multi-class classifier for amplitude classification (Full, Half, Invalid).
    """
    
    def __init__(self, model_type='random_forest', **kwargs):
        """
        Initialize amplitude classifier.
        
        Args:
            model_type: str, 'random_forest' or 'xgboost'
            **kwargs: additional parameters for the model
        """
        self.model_type = model_type
        self.label_encoder = LabelEncoder()
        
        if model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }
            default_params.update(kwargs)
            self.model = RandomForestClassifier(**default_params)
        
        elif model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(kwargs)
            self.model = XGBClassifier(**default_params)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X, y):
        """
        Train the amplitude classifier.
        
        Args:
            X: np.array, feature matrix (n_samples, n_features)
            y: array-like, amplitude labels ('full', 'half', 'invalid')
        """
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
    
    def predict(self, X):
        """
        Predict amplitude class.
        
        Args:
            X: np.array, feature matrix (n_samples, n_features)
        
        Returns:
            np.array: amplitude predictions
        """
        y_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X):
        """
        Predict probability for each amplitude class.
        
        Args:
            X: np.array, feature matrix (n_samples, n_features)
        
        Returns:
            np.array: probability matrix (n_samples, n_classes)
        """
        return self.model.predict_proba(X)
    
    def get_classes(self):
        """Get class labels."""
        return self.label_encoder.classes_
    
    def save(self, filepath):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'label_encoder': self.label_encoder}, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_encoder = data['label_encoder']
        print(f"Model loaded from {filepath}")


class FatigueClassifier:
    """
    Multi-class classifier for fatigue classification (Free, Light, Medium, Heavy).
    Only applicable to 'full' amplitude data.
    """
    
    def __init__(self, model_type='random_forest', **kwargs):
        """
        Initialize fatigue classifier.
        
        Args:
            model_type: str, 'random_forest' or 'xgboost'
            **kwargs: additional parameters for the model
        """
        self.model_type = model_type
        self.label_encoder = LabelEncoder()
        
        if model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }
            default_params.update(kwargs)
            self.model = RandomForestClassifier(**default_params)
        
        elif model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(kwargs)
            self.model = XGBClassifier(**default_params)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X, y):
        """
        Train the fatigue classifier.
        
        Args:
            X: np.array, feature matrix (n_samples, n_features)
            y: array-like, fatigue labels ('free', 'light', 'medium', 'heavy')
        """
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
    
    def predict(self, X):
        """
        Predict fatigue class.
        
        Args:
            X: np.array, feature matrix (n_samples, n_features)
        
        Returns:
            np.array: fatigue predictions
        """
        y_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X):
        """
        Predict probability for each fatigue class.
        
        Args:
            X: np.array, feature matrix (n_samples, n_features)
        
        Returns:
            np.array: probability matrix (n_samples, n_classes)
        """
        return self.model.predict_proba(X)
    
    def get_classes(self):
        """Get class labels."""
        return self.label_encoder.classes_
    
    def save(self, filepath):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'label_encoder': self.label_encoder}, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_encoder = data['label_encoder']
        print(f"Model loaded from {filepath}")


# --- CRNN for three-state activity segmentation ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class _CRNNBackbone(nn.Module):
    def __init__(self, input_channels=1, conv_channels=32, lstm_hidden=64, num_classes=3):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, conv_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(conv_channels)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) or (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.bn(self.relu(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        x, _ = self.lstm(x)
        logits = self.classifier(x)
        return logits


class CRNNActivitySegmenter:
    """
    Time-series semantic segmenter producing 0/1/2 labels per timestep.
    """

    def __init__(
        self,
        sequence_length=400,
        step_size=200,
        conv_channels=32,
        lstm_hidden=64,
        num_classes=3,
        device=None,
    ):
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.conv_channels = conv_channels
        self.lstm_hidden = lstm_hidden
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _CRNNBackbone(
            conv_channels=self.conv_channels,
            lstm_hidden=self.lstm_hidden,
            num_classes=self.num_classes,
        ).to(self.device)

    def fit(self, windows, labels, epochs=3, batch_size=32, lr=1e-3):
        """
        Train CRNN using paired signal/label sequences.
        Args:
            windows (np.array): shape (n_samples, seq_len)
            labels (np.array): shape (n_samples, seq_len)
        """
        x_tensor = torch.tensor(windows, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits.view(-1, 3), yb.view(-1))
                loss.backward()
                optimizer.step()

    def predict(self, windows):
        """
        Predict per-timestep labels for given windows.
        """
        self.model.eval()
        x_tensor = torch.tensor(windows, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(x_tensor)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
        return preds

    def save(self, filepath):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "sequence_length": self.sequence_length,
                "step_size": self.step_size,
                "conv_channels": self.conv_channels,
                "lstm_hidden": self.lstm_hidden,
                "num_classes": self.num_classes,
            },
            filepath,
        )

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.sequence_length = checkpoint.get("sequence_length", self.sequence_length)
        self.step_size = checkpoint.get("step_size", self.step_size)
        self.conv_channels = checkpoint.get("conv_channels", self.conv_channels)
        self.lstm_hidden = checkpoint.get("lstm_hidden", self.lstm_hidden)
        self.num_classes = checkpoint.get("num_classes", self.num_classes)
        self.model = _CRNNBackbone(
            conv_channels=self.conv_channels,
            lstm_hidden=self.lstm_hidden,
            num_classes=self.num_classes,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
