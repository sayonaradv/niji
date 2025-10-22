# Niji Code Review Improvements

## High Priority

- **Correct invalid examples**: The main block in `training.py` has `warmup_epochs=20` and `max_epochs=10`, violating the constraint. Update to valid values.

## Medium Priority

- **Enhance error handling**: The `Niji.predict` method lacks input validation or error handling for invalid text inputs.
- **Improve documentation**: Add docstrings or comments for complex logic, like the LR scheduler in `schedulers.py`.
- **Testing coverage**: Tests exist for core components, but expand to cover CLI, server, and edge cases (e.g., missing data files).

## Low Priority

- **Configuration management**: The README mentions YAML configs, but no config handling is implemented. Add support for loading/saving training configs.
