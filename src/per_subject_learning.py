"""
Per-subject learning module for EMG signal classification.
Implements subject-specific feature extraction and model training for better accuracy.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle


class PerSubjectClassifier:
    """
    Classifier that learns separate patterns for each subject, then aggregates.
    """
    
    def __init__(self, model_type='random_forest', **kwargs):
        """
        Initialize per-subject classifier.
        
        Args:
            model_type: str, 'random_forest' or 'xgboost'
            **kwargs: additional parameters for the models
        """
        self.model_type = model_type
        self.subject_models = {}  # Maps subject_id -> trained model
        self.global_model = None  # Global model trained on all data
        self.label_encoder = LabelEncoder()
        self.model_kwargs = kwargs
        
        # Set default parameters
        if model_type == 'random_forest':
            self.default_params = {
                'n_estimators': 50,
                'max_depth': 10,
                'min_samples_split': 3,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            from xgboost import XGBClassifier
            self.default_params = {
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
        
        self.default_params.update(kwargs)
    
    def _create_model(self):
        """Create a new model instance."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(**self.default_params)
        else:
            from xgboost import XGBClassifier
            return XGBClassifier(**self.default_params)
    
    def fit(self, X, y, subject_ids):
        """
        Train per-subject models and a global model.
        
        Args:
            X: np.array, feature matrix (n_samples, n_features)
            y: array-like, labels
            subject_ids: array-like, subject ID for each sample
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Get unique subjects
        unique_subjects = np.unique(subject_ids)
        
        print(f"\nTraining per-subject models for {len(unique_subjects)} subjects...")
        
        # Train a model for each subject with sufficient data
        for subject_id in unique_subjects:
            subject_mask = subject_ids == subject_id
            X_subject = X[subject_mask]
            y_subject = y_encoded[subject_mask]
            
            # Only train if subject has enough samples
            if len(X_subject) >= 5:  # Minimum threshold
                print(f"  Subject {subject_id}: {len(X_subject)} samples")
                
                # Check if subject has multiple classes
                if len(np.unique(y_subject)) > 1:
                    model = self._create_model()
                    model.fit(X_subject, y_subject)
                    self.subject_models[subject_id] = model
                else:
                    print(f"    Warning: Subject {subject_id} has only one class, skipping per-subject model")
            else:
                print(f"  Subject {subject_id}: {len(X_subject)} samples (insufficient, skipping)")
        
        # Train global model on all data
        print(f"\nTraining global model on all {len(X)} samples...")
        self.global_model = self._create_model()
        self.global_model.fit(X, y_encoded)
        
        print(f"Per-subject training complete: {len(self.subject_models)} subject-specific models")
    
    def predict(self, X, subject_ids=None):
        """
        Predict using per-subject models when available, otherwise use global model.
        
        Args:
            X: np.array, feature matrix (n_samples, n_features)
            subject_ids: array-like, subject ID for each sample (optional)
        
        Returns:
            np.array: predictions
        """
        if subject_ids is None or len(self.subject_models) == 0:
            # Use global model only
            y_encoded = self.global_model.predict(X)
            return self.label_encoder.inverse_transform(y_encoded)
        
        # Use per-subject models when available
        predictions = np.zeros(len(X), dtype=object)
        
        for i, (x, subject_id) in enumerate(zip(X, subject_ids)):
            if subject_id in self.subject_models:
                # Use subject-specific model
                y_encoded = self.subject_models[subject_id].predict(x.reshape(1, -1))
            else:
                # Fall back to global model
                y_encoded = self.global_model.predict(x.reshape(1, -1))
            
            predictions[i] = self.label_encoder.inverse_transform(y_encoded)[0]
        
        return predictions
    
    def predict_proba(self, X, subject_ids=None):
        """
        Predict probabilities using per-subject models when available.
        
        Args:
            X: np.array, feature matrix (n_samples, n_features)
            subject_ids: array-like, subject ID for each sample (optional)
        
        Returns:
            np.array: probability matrix
        """
        if subject_ids is None or len(self.subject_models) == 0:
            # Use global model only
            return self.global_model.predict_proba(X)
        
        # Use per-subject models when available
        n_classes = len(self.label_encoder.classes_)
        probas = np.zeros((len(X), n_classes))
        
        for i, (x, subject_id) in enumerate(zip(X, subject_ids)):
            if subject_id in self.subject_models:
                # Use subject-specific model
                probas[i] = self.subject_models[subject_id].predict_proba(x.reshape(1, -1))[0]
            else:
                # Fall back to global model
                probas[i] = self.global_model.predict_proba(x.reshape(1, -1))[0]
        
        return probas
    
    def get_classes(self):
        """Get class labels."""
        return self.label_encoder.classes_
    
    def save(self, filepath):
        """Save models to file."""
        data = {
            'subject_models': self.subject_models,
            'global_model': self.global_model,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'model_kwargs': self.model_kwargs
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Per-subject model saved to {filepath}")
    
    def load(self, filepath):
        """Load models from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.subject_models = data['subject_models']
            self.global_model = data['global_model']
            self.label_encoder = data['label_encoder']
            self.model_type = data['model_type']
            self.model_kwargs = data.get('model_kwargs', {})
        print(f"Per-subject model loaded from {filepath}")
        print(f"  Subject-specific models: {len(self.subject_models)}")


def extract_subject_features(X, subject_ids):
    """
    Extract subject-specific features by computing statistics per subject.
    
    Args:
        X: np.array, feature matrix (n_samples, n_features)
        subject_ids: array-like, subject ID for each sample
    
    Returns:
        np.array: enhanced feature matrix with subject statistics
    """
    unique_subjects = np.unique(subject_ids)
    subject_stats = {}
    
    # Compute statistics for each subject
    for subject_id in unique_subjects:
        subject_mask = subject_ids == subject_id
        X_subject = X[subject_mask]
        
        # Compute mean and std for each feature
        mean_features = np.mean(X_subject, axis=0)
        std_features = np.std(X_subject, axis=0)
        
        subject_stats[subject_id] = {
            'mean': mean_features,
            'std': std_features
        }
    
    # Create enhanced features
    X_enhanced = []
    
    for i, (x, subject_id) in enumerate(zip(X, subject_ids)):
        # Original features
        features = list(x)
        
        # Add subject-relative features if subject stats available
        if subject_id in subject_stats:
            stats = subject_stats[subject_id]
            
            # Normalized features (z-score relative to subject)
            normalized = (x - stats['mean']) / (stats['std'] + 1e-10)
            features.extend(normalized)
        else:
            # If no subject stats, add zeros
            features.extend(np.zeros_like(x))
        
        X_enhanced.append(features)
    
    return np.array(X_enhanced)
