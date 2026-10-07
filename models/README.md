# Models

Trained model artifacts and their model cards live here. The shipped research
checkpoint is in `heterogcn2_puloss_3layers/`.

Every checkpoint is a `model.pth` state dict with a `model.json` beside it.
`model.json` records the architecture, the node-feature layout (company-type
slots), the relationship-type IDs, hyperparameters, and training provenance.
Prediction needs both files.

Models trained with `scripts/train.py` are written outside this directory, to
`artifacts/training/<run>/` by default, in the same format.
