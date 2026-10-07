# Tests

Unit and integration tests. Fixtures must use only synthetic data.

`test_dummy_data.py` validates the committed synthetic dataset using only the
Python standard library. `test_training.py` covers losses, metrics, splits,
schema validation, and short end-to-end training runs. `test_prediction.py`
checks checkpoint loading and prediction, and `test_visualization.py` checks
the interactive graph.
