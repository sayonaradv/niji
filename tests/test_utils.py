import numpy as np
import pytest
from numpy.testing import assert_array_equal

from stormy.utils import combine_labels


class TestCombineLabels:
    @pytest.fixture
    def sample_batch(self):
        """Sample dataset for testing."""
        return {
            "label1": [1, 0, 1, 0],
            "label2": [0, 1, 1, 0],
            "label3": [1, 1, 0, 1],
            "text": ["sample1", "sample2", "sample", "sample4"],
        }

    def test_basic_functionality(self, sample_batch):
        """Test basic label combination functionality."""
        result = combine_labels(sample_batch, ["label1", "label2"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)  # batch_size=4, num_labels=2

        # Check actual values
        expected = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        assert_array_equal(result, expected)

    def missing_column(self, sample_batch):
        """Test error handling for a single missing column."""
        with pytest.raises(KeyError):
            combine_labels(sample_batch, ["label1", "missing_label"])
