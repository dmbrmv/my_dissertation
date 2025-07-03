"""Tests for evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import nse, kge, rmse, relative_error, create_metrics_dataframe


class TestNSE:
    """Test Nash-Sutcliffe Efficiency metric."""
    
    def test_perfect_prediction(self):
        """Test NSE with perfect predictions."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = nse(predictions, targets)
        assert abs(result - 1.0) < 1e-10
    
    def test_mean_prediction(self):
        """Test NSE when predictions equal mean of targets."""
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = np.full_like(targets, np.mean(targets))
        
        result = nse(predictions, targets)
        assert abs(result) < 1e-10
    
    def test_with_nans(self):
        """Test NSE with NaN values."""
        predictions = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        targets = np.array([1.1, 1.9, 3.1, 3.9, 4.9])
        
        result = nse(predictions, targets)
        assert not np.isnan(result)
    
    def test_mismatched_lengths(self):
        """Test NSE with mismatched array lengths."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="same length"):
            nse(predictions, targets)


class TestKGE:
    """Test Kling-Gupta Efficiency metric."""
    
    def test_perfect_prediction(self):
        """Test KGE with perfect predictions."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        kge_val, r, alpha, beta = kge(predictions, targets)
        
        assert abs(kge_val - 1.0) < 1e-10
        assert abs(r - 1.0) < 1e-10
        assert abs(alpha - 1.0) < 1e-10
        assert abs(beta - 1.0) < 1e-10
    
    def test_components_range(self):
        """Test that KGE components are in reasonable ranges."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.1, 1.9, 3.1, 3.9, 4.9])
        
        kge_val, r, alpha, beta = kge(predictions, targets)
        
        assert -1 <= r <= 1  # Correlation should be between -1 and 1
        assert alpha > 0     # Alpha should be positive
        assert beta > 0      # Beta should be positive
    
    def test_with_nans(self):
        """Test KGE with NaN values."""
        predictions = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        targets = np.array([1.1, 1.9, 3.1, 3.9, 4.9])
        
        kge_val, r, alpha, beta = kge(predictions, targets)
        assert not np.isnan(kge_val)


class TestRMSE:
    """Test Root Mean Square Error metric."""
    
    def test_perfect_prediction(self):
        """Test RMSE with perfect predictions."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = rmse(predictions, targets)
        assert abs(result) < 1e-10
    
    def test_constant_error(self):
        """Test RMSE with constant error."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([2.0, 3.0, 4.0, 5.0, 6.0])  # Off by 1
        
        result = rmse(predictions, targets)
        assert abs(result - 1.0) < 1e-10
    
    def test_positive_result(self):
        """Test that RMSE is always positive."""
        predictions = np.array([-5.0, -2.0, 1.0, 3.0, 10.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = rmse(predictions, targets)
        assert result >= 0


class TestRelativeError:
    """Test relative error metric."""
    
    def test_perfect_prediction(self):
        """Test relative error with perfect predictions."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = relative_error(predictions, targets)
        assert abs(result) < 1e-10
    
    def test_percentage_output(self):
        """Test that relative error returns percentage."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([2.0, 4.0, 6.0])  # 50% error
        
        result = relative_error(predictions, targets)
        assert abs(result - 50.0) < 1e-10
    
    def test_zero_targets_excluded(self):
        """Test that zero targets are excluded from calculation."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([0.0, 2.0, 3.0])  # First target is zero
        
        result = relative_error(predictions, targets)
        assert not np.isnan(result)


class TestCreateMetricsDataframe:
    """Test metrics DataFrame creation."""
    
    def test_basic_functionality(self):
        """Test basic metrics DataFrame creation."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.1, 1.9, 3.1, 3.9, 4.9])
        gauge_id = "test_gauge"
        
        result = create_metrics_dataframe(gauge_id, predictions, targets)
        
        assert len(result) == 1
        assert result.index[0] == gauge_id
        assert "NSE" in result.columns
        assert "KGE" in result.columns
        assert "RMSE" in result.columns
    
    def test_data_quality_metrics(self):
        """Test that data quality metrics are included."""
        predictions = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        targets = np.array([1.1, 1.9, 3.1, 3.9, 4.9])
        gauge_id = "test_gauge"
        
        result = create_metrics_dataframe(gauge_id, predictions, targets)
        
        assert "n_observations" in result.columns
        assert "n_valid" in result.columns
        assert "completeness" in result.columns
        assert result.loc[gauge_id, "n_observations"] == 5
        assert result.loc[gauge_id, "n_valid"] == 4
        assert result.loc[gauge_id, "completeness"] == 0.8
    
    def test_error_handling(self):
        """Test error handling in metrics calculation."""
        predictions = np.array([np.nan, np.nan, np.nan])
        targets = np.array([1.0, 2.0, 3.0])
        gauge_id = "test_gauge"
        
        result = create_metrics_dataframe(gauge_id, predictions, targets)
        
        # Should not raise exception, but fill with NaN
        assert len(result) == 1
        assert pd.isna(result.loc[gauge_id, "NSE"])
