# No Data Is Not No Risk

Research code for predicting company risk from neighbouring companies in a
corporate hierarchy graph with a heterogeneous graph convolutional network.

## What's included

- Local training of the `HeteroGCN2` model on your own data.
- Annual risk prediction with the trained research checkpoint or your own model.
- An interactive graph visualization of known incidents and predicted risk.
- A small synthetic corporate hierarchy dataset of fictional companies.
- The trained research checkpoint, with the metadata needed to load it.

## Repository layout

```text
.
├── data/
│   └── dummy/               # Public synthetic corporate graph data
├── models/                  # Versioned model artifacts and model cards
├── scripts/                 # Thin command-line entry points
├── src/risk_propagation/    # Reusable Python package
└── tests/                   # Unit and integration tests
```

## Installation

Use Python 3.11–3.13 in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Train a model

Training requires your own data: the proprietary data behind the research
checkpoint is not published. Export it as `nodes.csv`, `edges.csv`, and
`risks.csv` following [`data/README.md`](data/README.md), which documents every
column and how it becomes training targets, labels, and features.

To try the pipeline first, train on the synthetic sample:

```bash
python scripts/train.py \
  --data-dir data/dummy \
  --year 2024 \
  --epochs 200 --batch-size 8 --patience 20 \
  --positive-prior 0.4 \
  --output-dir outputs/training/dummy-model
```

Each company with a 2024 row in `risks.csv` is a target. The model learns
whether that company had an incident from its neighbours' 2024 incidents, with
the target's own value masked. Messages pass through a separate graph
convolution for each relationship type and direction. All other defaults
reproduce the shipped research checkpoint: `HeteroGCN2`,
positive-unlabelled (PU) loss, 3 layers, hidden size 64, learning rate 0.001,
batch size 64, and 2 epochs.

The dummy command trains longer on smaller batches because the sample has only
54 companies. It also raises `--positive-prior`, the estimated true share of
risky companies, from its default of 0.12. About 30% of dummy
companies have an observed 2024 incident, and the prior must be at least that
share; training warns when it is not. With your own data, set it from your
best estimate and otherwise start from the defaults:

```bash
python scripts/train.py --data-dir path/to/your-data --year 2024
```

Run `python scripts/train.py --help` for all options. Each run directory
(`artifacts/training/<run>/` unless `--output-dir` is given) contains:

- `model.pth`: the weights with the best validation precision at 10% recall,
  with AUC-PR breaking ties.
- `model.json`: the architecture and node-feature layout needed to load the
  model, plus
  hyperparameters, a data summary, and every validation evaluation.
- `validation_predictions.csv`: predicted risk and observed label for each
  held-out company.

A trained model is used exactly like the shipped checkpoint:

```bash
python scripts/predict.py \
  --data-dir data/dummy \
  --model outputs/training/dummy-model/model.pth \
  --year 2025 \
  --feature-year 2024 \
  --output outputs/predictions_2025.csv
```

Metrics on the dummy data only show that the pipeline works; they say nothing
about real-world performance.

## Predict 2025 risk

Predict with the shipped research checkpoint:

```bash
python scripts/predict.py \
  --data-dir data/dummy \
  --model models/heterogcn2_puloss_3layers/model.pth \
  --year 2025 \
  --feature-year 2024 \
  --output outputs/predictions_2025.csv
```

The command predicts 2025 risk from known 2024 incident observations. Output
is ranked by predicted risk and also includes the synthetic observed 2025 label
for comparison.

## Visualize the graph

After generating predictions, create a standalone interactive graph:

```bash
python scripts/visualize.py \
  --data-dir data/dummy \
  --predictions outputs/predictions_2025.csv \
  --year 2025 \
  --output outputs/company-risk-2025.html
```

Open `outputs/company-risk-2025.html` in a browser. Hover over a node to see the
company name, predicted 2025 risk, and known 2024 incident count. Node color
runs from green at 0% predicted risk through yellow to red at 20% or more. A
small red dot indicates a known 2024 incident (no dot means none); a dark outer
ring identifies companies with no recorded incident before the visualized year.
