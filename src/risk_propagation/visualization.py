"""Interactive company graph visualization."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go

from risk_propagation.data import GraphDataset, load_graph_dataset

# Predicted risks at or above this value are drawn fully red.
RISK_COLOR_SCALE_MAXIMUM = 0.2


@dataclass(frozen=True)
class PredictionValue:
    company_id: str
    company_name: str
    prediction_year: int
    predicted_risk: float
    feature_year: int | None = None


def load_predictions(predictions_path: str | Path, prediction_year: int) -> dict[str, PredictionValue]:
    """Load one prediction year from the predictor's CSV output."""

    predictions_path = Path(predictions_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(f'Predictions file does not exist: {predictions_path}')

    predictions: dict[str, PredictionValue] = {}
    with predictions_path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            row_year = int(row['prediction_year'])
            if row_year != prediction_year:
                continue
            company_id = row['company_id']
            predicted_risk = float(row['predicted_risk'])
            if company_id in predictions:
                raise ValueError(f'Duplicate prediction for {company_id!r} in {prediction_year}')
            if not 0.0 <= predicted_risk <= 1.0:
                raise ValueError(f'Predicted risk for {company_id!r} is outside [0, 1]')
            predictions[company_id] = PredictionValue(
                company_id=company_id,
                company_name=row['company_name'],
                prediction_year=row_year,
                predicted_risk=predicted_risk,
                feature_year=int(row['feature_year']) if row.get('feature_year') else None,
            )

    if not predictions:
        raise ValueError(f'No predictions for {prediction_year} found in {predictions_path}')
    return predictions


def _layout(dataset: GraphDataset) -> dict[str, tuple[float, float]]:
    graph = nx.Graph()
    graph.add_nodes_from(dataset.companies)
    graph.add_edges_from((edge.company_a, edge.company_b) for edge in dataset.relationships)
    return nx.spring_layout(graph, seed=42, k=0.75, iterations=300)


def build_graph_figure(
    dataset: GraphDataset,
    predictions: dict[str, PredictionValue],
    prediction_year: int,
    feature_year: int | None = None,
) -> go.Figure:
    """Build a hoverable graph of predicted risk and known feature-year incidents."""

    missing_predictions = set(dataset.companies) - set(predictions)
    if missing_predictions:
        missing = ', '.join(sorted(missing_predictions))
        raise ValueError(f'Missing {prediction_year} predictions for: {missing}')

    prediction_feature_years = {value.feature_year for value in predictions.values() if value.feature_year is not None}
    if feature_year is None:
        if len(prediction_feature_years) > 1:
            raise ValueError('Predictions contain multiple feature years; pass feature_year explicitly')
        feature_year = prediction_feature_years.pop() if prediction_feature_years else prediction_year

    positions = _layout(dataset)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for edge in dataset.relationships:
        x0, y0 = positions[edge.company_a]
        x1, y1 = positions[edge.company_b]
        edge_x.extend((x0, x1, None))
        edge_y.extend((y0, y1, None))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line={'width': 1.2, 'color': '#b8c2cc'},
        hoverinfo='skip',
        showlegend=False,
    )

    company_ids = list(dataset.companies)
    node_x = [positions[company_id][0] for company_id in company_ids]
    node_y = [positions[company_id][1] for company_id in company_ids]
    risk_values = [predictions[company_id].predicted_risk for company_id in company_ids]
    past_incident_counts = [
        sum(
            risk_value
            for (risk_company_id, year), risk_value in dataset.risks.items()
            if risk_company_id == company_id and year < prediction_year
        )
        for company_id in company_ids
    ]
    feature_year_incidents = [dataset.risks.get((company_id, feature_year), 0) for company_id in company_ids]
    custom_data = [
        [dataset.companies[company_id].company_name, risk, past_incidents, feature_incidents, company_id]
        for company_id, risk, past_incidents, feature_incidents in zip(
            company_ids, risk_values, past_incident_counts, feature_year_incidents, strict=True
        )
    ]

    never_observed_ids = [
        company_id
        for company_id, past_incidents in zip(company_ids, past_incident_counts, strict=True)
        if past_incidents == 0
    ]
    history_ring = go.Scatter(
        x=[positions[company_id][0] for company_id in never_observed_ids],
        y=[positions[company_id][1] for company_id in never_observed_ids],
        mode='markers',
        marker={
            'size': 31,
            'symbol': 'circle-open',
            'color': '#111827',
            'line': {'width': 3, 'color': '#111827'},
        },
        text=never_observed_ids,
        hoverinfo='skip',
        name=f'Invisible (no incident before {prediction_year})',
    )

    feature_year_incident_ids = [
        company_id
        for company_id, incident_count in zip(company_ids, feature_year_incidents, strict=True)
        if incident_count > 0
    ]
    feature_year_marker = go.Scatter(
        x=[positions[company_id][0] for company_id in feature_year_incident_ids],
        y=[positions[company_id][1] for company_id in feature_year_incident_ids],
        mode='markers',
        marker={
            'size': 9,
            'symbol': 'circle',
            'color': '#dc2626',
            'line': {'width': 1.5, 'color': '#ffffff'},
        },
        text=feature_year_incident_ids,
        hoverinfo='skip',
        name=f'Known incident in {feature_year}',
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=company_ids,
        textposition='top center',
        textfont={'size': 10, 'color': '#374151'},
        customdata=custom_data,
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>'
            'Company ID: %{customdata[4]}<br>'
            f'Predicted {prediction_year} risk: '
            '%{customdata[1]:.1%}<br>'
            f'Recorded incidents before {prediction_year}: '
            f'%{{customdata[2]}}<br>Known incidents in feature year {feature_year}: '
            '%{customdata[3]}<extra></extra>'
        ),
        marker={
            'size': 23,
            'color': risk_values,
            'cmin': 0,
            'cmax': RISK_COLOR_SCALE_MAXIMUM,
            'colorscale': [[0, '#16a34a'], [0.5, '#facc15'], [1, '#dc2626']],
            'line': {'width': 1.5, 'color': '#ffffff'},
            'colorbar': {
                'title': {'text': f'Predicted {prediction_year}<br>risk'},
                'tickvals': [0, 0.05, 0.1, 0.15, RISK_COLOR_SCALE_MAXIMUM],
                'ticktext': ['0%', '5%', '10%', '15%', f'≥{RISK_COLOR_SCALE_MAXIMUM:.0%}'],
                'thickness': 18,
            },
        },
        showlegend=False,
    )

    # Plot the incident dots last so they remain visible over the risk-coloured nodes.
    figure = go.Figure(data=[edge_trace, history_ring, node_trace, feature_year_marker])
    figure.update_layout(
        title={
            'text': f'Company risk propagation — {prediction_year}',
            'x': 0.02,
            'xanchor': 'left',
            'font': {'size': 24, 'color': '#111827'},
        },
        annotations=[
            {
                'text': (
                    'Color shows predicted risk. A dark outer ring marks companies with no earlier recorded incident; '
                    f'a red dot marks a known incident in the {feature_year} feature year.'
                ),
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.02,
                'y': 1.04,
                'showarrow': False,
                'font': {'size': 13, 'color': '#4b5563'},
            }
        ],
        hovermode='closest',
        hoverlabel={'bgcolor': '#111827', 'font': {'color': '#ffffff', 'size': 13}},
        legend={
            'orientation': 'h',
            'x': 0.02,
            'y': 0.02,
            'bgcolor': 'rgba(255,255,255,0.85)',
        },
        margin={'l': 25, 'r': 100, 't': 100, 'b': 25},
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        xaxis={'visible': False},
        yaxis={'visible': False},
        width=1200,
        height=800,
    )
    return figure


def write_graph_html(figure: go.Figure, output_path: str | Path) -> Path:
    """Write a fully standalone interactive HTML document."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        config={
            'displaylogo': False,
            'responsive': True,
            'scrollZoom': True,
            'toImageButtonOptions': {'format': 'png', 'filename': output_path.stem, 'scale': 2},
        },
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate an interactive company-risk graph')
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory containing nodes/edges/risks CSVs')
    parser.add_argument('--predictions', type=Path, required=True, help='Prediction CSV produced by predict.py')
    parser.add_argument('--year', type=int, default=2025, help='Prediction year to visualize (default: 2025)')
    parser.add_argument('--output', type=Path, required=True, help='Destination HTML path')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_graph_dataset(args.data_dir)
    predictions = load_predictions(args.predictions, args.year)
    figure = build_graph_figure(dataset, predictions, args.year)
    output_path = write_graph_html(figure, args.output)
    print(f'Interactive graph written to {output_path}')
    print('Open the HTML file in a browser to explore the network.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
