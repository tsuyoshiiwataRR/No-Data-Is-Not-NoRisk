"""Risk prediction workflows and command-line interface."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from risk_propagation.data import (
    RESEARCH_CHECKPOINT_FEATURES,
    NodeFeatureSpec,
    extract_neighbourhood,
    load_graph_dataset,
)
from risk_propagation.model import HeteroGCN2, load_checkpoint


@dataclass(frozen=True)
class RiskPrediction:
    company_id: str
    company_name: str
    feature_year: int
    prediction_year: int
    predicted_risk: float
    observed_risk_label: int | None


def predict_risks(
    model: HeteroGCN2,
    data_dir: str | Path,
    prediction_year: int,
    feature_year: int | None = None,
    max_depth: int | None = None,
    feature_spec: NodeFeatureSpec = RESEARCH_CHECKPOINT_FEATURES,
) -> list[RiskPrediction]:
    """Predict risk for every company from neighbours in the selected year.

    ``max_depth`` defaults to the model's number of graph-convolution layers.
    """

    feature_year = prediction_year if feature_year is None else feature_year
    max_depth = model.nb_layers if max_depth is None else max_depth
    dataset = load_graph_dataset(data_dir)
    predictions: list[RiskPrediction] = []

    model.eval()
    with torch.no_grad():
        for company in dataset.companies.values():
            graph = extract_neighbourhood(
                dataset, company.company_id, feature_year, prediction_year, max_depth, feature_spec
            )
            batch = torch.zeros(graph.num_nodes, dtype=torch.long)
            logit = model(graph.x, graph.edge_index, graph.edge_attr, batch)[0, 0]
            predictions.append(
                RiskPrediction(
                    company_id=company.company_id,
                    company_name=company.company_name,
                    feature_year=feature_year,
                    prediction_year=prediction_year,
                    predicted_risk=torch.sigmoid(logit).item(),
                    observed_risk_label=dataset.risks.get((company.company_id, prediction_year)),
                )
            )
    return predictions


def write_predictions(predictions: list[RiskPrediction], output_path: str | Path) -> Path:
    """Write predictions in descending risk order."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'company_id',
        'company_name',
        'feature_year',
        'prediction_year',
        'predicted_risk',
        'observed_risk_label',
    ]
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in sorted(predictions, key=lambda item: item.predicted_risk, reverse=True):
            writer.writerow(
                {
                    **prediction.__dict__,
                    'predicted_risk': f'{prediction.predicted_risk:.6f}',
                    'observed_risk_label': (
                        '' if prediction.observed_risk_label is None else prediction.observed_risk_label
                    ),
                }
            )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Predict annual company risk from graph neighbourhoods')
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing nodes/edges/risks CSVs')
    parser.add_argument(
        '--model',
        type=Path,
        required=True,
        help='Path to a HeteroGCN2 model.pth checkpoint with its model.json beside it',
    )
    parser.add_argument('--year', type=int, default=2025, help='Risk year to predict (default: 2025)')
    parser.add_argument('--feature-year', type=int, help='Neighbour risk year (default: same as --year)')
    parser.add_argument('--max-depth', type=int, help='Neighbourhood depth (default: number of model layers)')
    parser.add_argument('--output', type=Path, required=True, help='Destination CSV path')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model, feature_spec = load_checkpoint(args.model)
    predictions = predict_risks(
        model,
        args.data_dir,
        args.year,
        feature_year=args.feature_year,
        max_depth=args.max_depth,
        feature_spec=feature_spec,
    )
    output_path = write_predictions(predictions, args.output)

    print(f'Predicted {args.year} risk for {len(predictions)} companies using {predictions[0].feature_year} features.')
    print(f'Predictions written to {output_path}')
    print('Highest predicted risks:')
    for prediction in sorted(predictions, key=lambda item: item.predicted_risk, reverse=True)[:5]:
        print(f'  {prediction.company_name}: {prediction.predicted_risk:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
