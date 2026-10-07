# Dummy dataset

This directory contains a small, entirely synthetic corporate hierarchy graph.
Every company name and every value is fictional and was created for this
repository. It contains no transformed, sampled, or anonymized production data.

The graph has 54 companies in five fictional groups, joined into one connected
network by synthetic cross-group relationships, and five years of synthetic
incident observations (2021–2025). Two groups intentionally make
neighbourhood-risk patterns easy to explore:

- **Silver Birch Holdings** is a low-recent-risk area. Its focal operating
  company, Silver Birch Services (`L401`), has several direct neighbours with
  a mixture of historical incidents and no history, but none of those
  neighbours recorded an incident in 2024 or 2025.
- **Iron Peak Holdings** is a high-recent-risk area. Its focal operating
  company, Iron Peak Operations (`R501`), has several direct neighbours with
  incidents recorded in both 2024 and 2025.

These patterns are intentionally synthetic teaching examples, not assertions
that a recorded zero proves a company is low-risk.

## Files

The general input format, including the optional `company_type` column that
this sample omits, is documented in [`../README.md`](../README.md).

- `nodes.csv`: company identifiers and display names.
- `edges.csv`: directed company relationships using the model's eight
  supported relationship types.
- `risks.csv`: one synthetic observed-risk value per company and year.

## Node schema

| Column | Type | Meaning |
| --- | --- | --- |
| `company_id` | string | Stable fictional identifier. |
| `company_name` | string | Fictional display name. |

Raw nodes intentionally contain no train/validation/test designation. Training
code is responsible for creating reproducible splits for a requested year.

## Edge schema

| Column | Type | Meaning |
| --- | --- | --- |
| `edge_id` | string | Stable fictional edge identifier. |
| `company_a` | string | Source company in the directed relationship. |
| `company_b` | string | Destination company in the directed relationship. |
| `reltype_id` | integer | Relationship type ID, listed below. |
| `relationship_type` | string | Human-readable relationship name. |

The model supports seven frequent relationship types; every rarer type is
mapped to `0` (`Other`). The dummy graph includes all eight:

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

The IDs and names are the relationship categories the shipped checkpoint was
trained with. The rows themselves are synthetic. The model passes messages in
both directions itself, so reverse edges are not duplicated in the raw table.

## Risk schema

| Column | Type | Meaning |
| --- | --- | --- |
| `company_id` | string | Company associated with the observation. |
| `year` | integer | Observation year, from 2021 through 2025. |
| `risk_label` | non-negative integer | Number of synthetic recorded incidents that year. |

A value greater than zero is an observed positive risk signal. A zero means no
incident was recorded; it does **not** prove that the company was low-risk.
This distinction reflects the positive-unlabelled nature of the project.

For a selected five-year window, a visualization can classify a company as
`visible` if any of its risk values is greater than zero and `invisible` if all
five are zero. Training can use the 2024 graph and labels, while evaluation can
apply that model to propagate known 2025 signals toward masked 2025 targets.
The observation year is selected by code rather than encoded as a split in
these raw files.

## Intended use

The dataset is designed for examples, tests, local training smoke tests, and
the interactive visualization. Its size and invented labels make it
unsuitable for drawing real-world risk conclusions or benchmarking accuracy.
