# Input data schema

Training (`scripts/train.py`) and prediction (`scripts/predict.py`) read a
directory of three CSV files describing a corporate graph and annual risk
observations. This repository does **not** ship real training data: the
research model was trained on proprietary data that cannot be published. To
train your own model, export your data in the format below.

`dummy/` is a small, entirely synthetic example in this format. Use it to try
the commands and to check your own export against a working dataset. Never
commit real company graphs or real risk targets to this repository.

```text
my-data/
├── nodes.csv    # one row per company
├── edges.csv    # one row per directed relationship between two companies
└── risks.csv    # one row per company and year with an observed risk value
```

All files are UTF-8 CSV with a header row. Column order does not matter, and
extra columns are ignored. Identifiers are compared as exact strings.

## `nodes.csv`

| Column | Required | Type | Meaning |
| --- | --- | --- | --- |
| `company_id` | yes | string | Unique, non-empty company identifier. |
| `company_name` | no | string | Display name used in outputs. Defaults to `company_id`. |
| `company_type` | no | string | Categorical company type, such as `Public` or `Fund`. |

Every company in the graph must have a row, including companies that are only
neighbours and never prediction targets.

`company_type` becomes a one-hot node feature. A dedicated `Other` slot comes
first; types held by at most 1% of companies, empty values, and types unseen
during training share it. The resulting slot list is saved with the model in
`model.json`, so prediction encodes types exactly as training did. Without the
column, every company uses the `Other` slot.

## `edges.csv`

| Column | Required | Type | Meaning |
| --- | --- | --- | --- |
| `edge_id` | yes | string | Unique, non-empty relationship identifier. |
| `company_a` | yes | string | Source `company_id` of the directed relationship. |
| `company_b` | yes | string | Destination `company_id` of the directed relationship. |
| `reltype_id` | yes | integer | Relationship type; must be one of the IDs below. |
| `relationship_type` | no | string | Human-readable relationship name. |

Direction follows the relationship: for example, a parent (`company_a`) and its
subsidiary (`company_b`) with `reltype_id` 5. Store each relationship once. The
model passes messages in both directions, with separate weights for each
relationship type and direction.

`reltype_id` must be one of the eight relationship types the model supports.
Map any other relationship type to `0` (`Other`).

| ID | Relationship type |
| ---: | --- |
| 0 | Other |
| 1 | Current Investment |
| 2 | Prior Investment |
| 5 | Current Subsidiary |
| 6 | Prior Subsidiary |
| 7 | Merged Entity |
| 17 | Current Fund Sponsor |
| 34 | Current Fund Investment Advisor |

The model has one convolution per type and direction for exactly these eight
types, in this order, so the list cannot be extended without retraining.

## `risks.csv`

| Column | Required | Type | Meaning |
| --- | --- | --- | --- |
| `company_id` | yes | string | Company the observation belongs to. |
| `year` | yes | integer | Observation year. |
| `risk_label` | yes | non-negative integer | Number of observed risk incidents that year. |

Each `(company_id, year)` pair may appear at most once.

A `risk_label` greater than zero is an observed positive. A zero means no
incident was observed; it does **not** prove the company is low risk. The
default positive-unlabelled (PU) loss treats zeros as unlabelled for this
reason.

A missing row has a different meaning from a zero. A company without a row
for a year is outside the observed universe for that year: it is never a
training target, and its risk feature is 0 when it appears as a neighbour.
Include a row with `risk_label` 0 for every company you monitored that year.

## How training uses the files

For `--year Y` and `--feature-year F` (default `F = Y`):

| Concept | Definition |
| --- | --- |
| Training targets | Companies with a `risks.csv` row for year `Y`. |
| Target label | `1` if the company's year-`Y` `risk_label` is greater than 0, else `0`. |
| Neighbourhood | Companies within `--layers` relationship hops of the target, in either direction. At most `--max-neighbours` edges per company, direction, and hop are sampled. |
| Node features | The year-`F` `risk_label` count (0 if missing), followed by the one-hot `company_type` slots. The target's own risk feature is always set to 0 so its label cannot leak. |
| Train/validation split | Targets are split by `--val-fraction` with `--seed`, stratified so both sets contain positives. The raw files contain no split. |

Training requires at least two positive and two zero-label targets in year `Y`.
With the default PU loss, `--positive-prior` must be at least the share of
targets with an observed positive. It estimates the true share of risky
companies, including unobserved ones.
A useful model needs many more: the research checkpoint was trained on a
production-scale graph.

The shipped checkpoint was trained on one year with same-year neighbour
features (`--year 2024`). The resulting model is then applied to a later year with
`scripts/predict.py`, for example `--year 2025`.

## Minimal example

`nodes.csv`

```csv
company_id,company_name,company_type
H1,Example Holdings,Public
S1,Example Services,Private
S2,Example Logistics,Private
```

`edges.csv`

```csv
edge_id,company_a,company_b,reltype_id,relationship_type
E1,H1,S1,5,Current Subsidiary
E2,H1,S2,1,Current Investment
```

`risks.csv`

```csv
company_id,year,risk_label
H1,2024,0
S1,2024,2
S2,2024,0
```

## Validation

Loading fails with an explicit error if a required file or column is missing,
an identifier is empty or duplicated, an edge or risk row refers to an unknown
company, `reltype_id` is unsupported, `year` or `risk_label` is not an integer,
or `risk_label` is negative.
