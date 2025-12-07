
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
import sys
sys.path.append("../")

class TestModels:
    """Unit tests for LunarSense-3 models"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load models and data"""
        self.model = joblib.load("03_models/fusion_baseline_xgb.pkl")
        self.scalers = joblib.load("03_models/feature_scalers.pkl")
        self.catalog = pd.read_csv("07_deployment/event_catalog.csv")

    def test_model_exists(self):
        """Test model loading"""
        assert self.model is not None
        assert hasattr(self.model, 'predict')

    def test_prediction_shape(self):
        """Test prediction output shape"""
        X = np.random.randn(10, 11)
        pred = self.model.predict(X)
        assert len(pred) == 10
        assert all(p in [0, 1] for p in pred)

    def test_probability_range(self):
        """Test probabilities in [0, 1]"""
        X = np.random.randn(10, 11)
        proba = self.model.predict_proba(X)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_catalog_columns(self):
        """Test event catalog structure"""
        required = ['event_id', 'prediction', 'confidence', 'uncertainty']
        for col in required:
            assert col in self.catalog.columns

    def test_confidence_uncertainty(self):
        """Test confidence-uncertainty relationship"""
        assert np.all(self.catalog['confidence'] >= 0)
        assert np.all(self.catalog['confidence'] <= 1)
        assert np.all(self.catalog['uncertainty'] >= 0)
        assert np.all(self.catalog['uncertainty'] <= 1)

    def test_balanced_predictions(self):
        """Test model isn't predicting all one class"""
        predictions = self.catalog['prediction'].values
        pos_rate = predictions.sum() / len(predictions)
        assert 0.2 < pos_rate < 0.8  # Not extremely biased

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
