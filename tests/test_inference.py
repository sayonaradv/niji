import pytest

from ruffle.inference import load_checkpoint


class TestLoadCheckpoint:
    """Test cases for the `load_checkpoint` function."""

    def test_raises_error_when_no_arguments_provided(self) -> None:
        with pytest.raises(
            ValueError, match="Must provide either 'model_name' or 'ckpt_path'"
        ):
            load_checkpoint()

    def test_raises_error_for_unknown_model_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            load_checkpoint(model_name="nonexistent_model")

    def test_raises_error_for_nonexistent_checkpoint_file(self) -> None:
        nonexistent_path = "/path/that/does/not/exist.ckpt"
        with pytest.raises(FileNotFoundError, match="Checkpoint file does not exist:"):
            load_checkpoint(ckpt_path=nonexistent_path)
