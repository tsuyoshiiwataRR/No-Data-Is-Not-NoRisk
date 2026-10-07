# Scripts

Thin command-line entry points that run from a source checkout. The logic they
call lives in `src/risk_propagation`.

`train.py` trains a `HeteroGCN2` model locally on data following `data/README.md`.
`predict.py` runs annual `HeteroGCN2` risk prediction from a source checkout.
`visualize.py` turns prediction output into a standalone interactive HTML graph.
