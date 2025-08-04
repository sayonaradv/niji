import pytest

from stormy.utils import combine_labels


class TestCombineLabels:
    @pytest.fixture
    def sample_batch(self) -> dict[str, list[int | str]]:
        """Sample dataset for testing."""
        return {
            "label1": [1, 0, 1, 0],
            "label2": [0, 1, 1, 0],
            "label3": [1, 1, 0, 1],
            "text": ["sample1", "sample2", "sample", "sample4"],
        }

    def test_basic_functionality(
        self, sample_batch: dict[str, list[int | str]]
    ) -> None:
        """Test basic label combination functionality."""
        result = combine_labels(sample_batch, ["label1", "label2"])

        assert isinstance(result, list)
        assert len(result) == 4  # batch_size=4
        assert len(result[0]) == 2  # num_labels=2

        # Check actual values
        expected = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
        assert result == expected

    def test_missing_column(self, sample_batch: dict[str, list[int | str]]) -> None:
        """Test error handling for a single missing column."""
        with pytest.raises(
            KeyError, match="Label columns not found in dataset: \\['missing_label'\\]"
        ):
            combine_labels(sample_batch, ["label1", "missing_label"])

    def test_all_labels(self, sample_batch: dict[str, list[int | str]]) -> None:
        """Test combining all label columns."""
        result = combine_labels(sample_batch, ["label1", "label2", "label3"])

        assert isinstance(result, list)
        assert len(result) == 4  # batch_size=4
        assert len(result[0]) == 3  # num_labels=3

        expected = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert result == expected

    def test_empty_label_list(self, sample_batch: dict[str, list[int | str]]) -> None:
        """Test with empty label list."""
        with pytest.raises(ValueError, match="label_columns cannot be empty"):
            combine_labels(sample_batch, [])

    def test_multiple_missing_columns(
        self, sample_batch: dict[str, list[int | str]]
    ) -> None:
        """Test error handling for multiple missing columns."""
        with pytest.raises(
            KeyError,
            match="Label columns not found in dataset: \\['missing1', 'missing2'\\]",
        ):
            combine_labels(sample_batch, ["label1", "missing1", "missing2"])
