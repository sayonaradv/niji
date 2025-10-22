# Niji Code Review Improvements

## High Priority

- **Fix placeholder code**: Implement actual prediction logic in `Niji.predict` using the loaded model's forward pass and sigmoid activation.
- **Call model compilation**: In `module.py`, `configure_model` is defined but never invoked. Call it in `__init__` or a setup method to enable `torch.compile`.
- **Correct invalid examples**: The main block in `training.py` has `warmup_epochs=20` and `max_epochs=10`, violating the constraint. Update to valid values.
- **Add evaluation separation**: Move testing logic from `inference.py` to a new `eval.py` module.

## Medium Priority

- **Enhance error handling**: The `Niji.predict` method lacks input validation or error handling for invalid text inputs.
- **Improve documentation**: Add docstrings or comments for complex logic, like the LR scheduler in `schedulers.py`.
- **Testing coverage**: Tests exist for core components, but expand to cover CLI, server, and edge cases (e.g., missing data files).

## Low Priority

- **Configuration management**: The README mentions YAML configs, but no config handling is implemented. Add support for loading/saving training configs.
