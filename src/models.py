"""
Model wrappers for EMG signal classification.
Provides interfaces for training and prediction with sklearn models.
"""
import numpy as np
import pickle
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
        if self.decision_threshold is not None:
            proba = self.model.predict_proba(X)
            # Probability for the active class (assumed to be column 1)
            return (proba[:, 1] >= self.decision_threshold).astype(int)
        return self.model.predict(X)
    
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
