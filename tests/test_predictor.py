import pytest

from ruffle.predictor import Ruffle


class TestRuffle:
    def test_init_with_invalid_model(self):
        with pytest.raises(ValueError, match="Unknown model.*Available"):
            Ruffle(model_name="nonexistent-model")

    def test_init_with_invalid_threshold(self):
        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            Ruffle(threshold=1.5)

        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            Ruffle(threshold=-0.1)

    def test_init_with_invalid_ckpt_path(self):
        with pytest.raises(FileNotFoundError, match="No such file or directory"):
            Ruffle(ckpt_path="invalid_path")
