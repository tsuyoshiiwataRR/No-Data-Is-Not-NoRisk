# HeteroGCN2 PU checkpoint

The trained research checkpoint `heterogcn2_puloss_3layers`. It was trained
with the original `HeteroGCN2` implementation; parameter names are unchanged,
so `model.pth` loads without conversion. `model.json` holds the metadata
`load_checkpoint` needs.

## Architecture

- Model: `HeteroGCN2`
- Loss used for training: positive-unlabelled (`pu`)
- Graph-convolution layers: 3, each with one GCN2 convolution per
  relationship type and direction (8 types × 2 directions), summed
- Hidden size: 64
- Input features: 7
- Output: one risk logit per target company
- Training hyperparameters: learning rate 0.001, batch size 64, at most
  2 epochs, early-stopping patience 4

## Input compatibility

The seven inputs are the observed-risk value followed by six one-hot
company-type slots. The first slot is `Other`; the labels of the other five
company types are not public, so they are named `research_type_1` to
`research_type_5`. Companies without a matching type, including every dummy
company, use the `Other` slot. Each target company's own risk value is masked.

Relationship types use the eight `reltype_id` values documented in
`data/README.md`, in that order.

The checkpoint was trained on 2024 observations. The README example applies it
to predict 2025 risk from 2024 observations.

## Difference from the original implementation

The original implementation ran only the convolutions whose relationship type
appeared in the current batch. Every convolution also adds self-loop and
initial-residual terms to all nodes, so a company's score depended on the
other neighbourhoods in its batch. On the dummy graph, scoring companies one at
a time instead of together changed predicted probabilities by up to 0.40.
This repository's `HeteroGCN2` always runs all 16 convolutions, which matches
the large training batches in which every type was present. With all types
present, its output equals the original implementation's exactly.

## Integrity

- File: `model.pth`
- SHA-256: `5814897b6be731bdf59ace6f493075521f04d059e566bc748fbb3afba9335f71`
