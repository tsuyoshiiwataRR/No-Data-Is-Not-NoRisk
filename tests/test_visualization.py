import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
HAS_VISUALIZATION_DEPENDENCIES = all(importlib.util.find_spec(name) for name in ('networkx', 'plotly'))


@unittest.skipUnless(HAS_VISUALIZATION_DEPENDENCIES, 'networkx and plotly are not installed')
class VisualizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / 'src'))
        from risk_propagation.data import load_graph_dataset
        from risk_propagation.visualization import PredictionValue, build_graph_figure

        cls.dataset = load_graph_dataset(PROJECT_ROOT / 'data' / 'dummy')
        cls.predictions = {
            company_id: PredictionValue(
                company_id, company.company_name, 2025, index / (len(cls.dataset.companies) - 1), feature_year=2024
            )
            for index, (company_id, company) in enumerate(cls.dataset.companies.items())
        }
        cls.figure = build_graph_figure(cls.dataset, cls.predictions, 2025)

    def test_loads_predictions_for_the_selected_year(self) -> None:
        from risk_propagation.visualization import load_predictions

        with tempfile.TemporaryDirectory() as temporary_directory:
            predictions_path = Path(temporary_directory) / 'predictions.csv'
            predictions_path.write_text(
                'company_id,company_name,feature_year,prediction_year,predicted_risk\n'
                'H001,Asteria Group,2023,2024,0.1\n'
                'H001,Asteria Group,2024,2025,0.4\n',
                encoding='utf-8',
            )
            predictions = load_predictions(predictions_path, 2025)
        self.assertEqual({'H001'}, set(predictions))
        self.assertAlmostEqual(0.4, predictions['H001'].predicted_risk)
        self.assertEqual(2024, predictions['H001'].feature_year)

    def test_hover_contains_company_name_and_predicted_risk(self) -> None:
        node_trace = self.figure.data[2]
        self.assertIn('Predicted 2025 risk', node_trace.hovertemplate)
        self.assertIn('Known incidents in feature year 2024', node_trace.hovertemplate)
        company_names = {row[0] for row in node_trace.customdata}
        self.assertIn('Asteria Group', company_names)

    def test_risk_colors_and_no_history_marker_are_independent(self) -> None:
        history_ring = self.figure.data[1]
        node_trace = self.figure.data[2]
        self.assertEqual(0, node_trace.marker.cmin)
        self.assertEqual(0.2, node_trace.marker.cmax)
        self.assertIn('A107', set(history_ring.text))
        self.assertNotIn('A103', set(history_ring.text))

    def test_feature_year_incidents_have_a_distinct_red_dot_marker(self) -> None:
        feature_year_marker = self.figure.data[3]
        self.assertEqual('Known incident in 2024', feature_year_marker.name)
        self.assertEqual('#dc2626', feature_year_marker.marker.color)
        expected = {
            company_id for (company_id, year), value in self.dataset.risks.items() if year == 2024 and value > 0
        }
        self.assertTrue(expected)
        self.assertEqual(expected, set(feature_year_marker.text))
        # A103's only incident is in 2022, so it has no 2024 dot.
        self.assertNotIn('A103', set(feature_year_marker.text))

    def test_risk_color_scale_is_fully_red_from_20_percent_regardless_of_predictions(self) -> None:
        from risk_propagation.visualization import PredictionValue, build_graph_figure

        for risk in (0.05, 0.9):
            predictions = {
                company_id: PredictionValue(company_id, company.company_name, 2025, risk)
                for company_id, company in self.dataset.companies.items()
            }
            marker = build_graph_figure(self.dataset, predictions, 2025).data[2].marker
            self.assertEqual(0.2, marker.cmax)
            self.assertEqual('≥20%', marker.colorbar.ticktext[-1])

    def test_writes_a_standalone_html_document(self) -> None:
        from risk_propagation.visualization import write_graph_html

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = write_graph_html(self.figure, Path(temporary_directory) / 'graph.html')
            html = output_path.read_text(encoding='utf-8')
        self.assertIn('<html>', html)
        self.assertIn('plotly.js', html)
        self.assertIn('Asteria Group', html)


if __name__ == '__main__':
    unittest.main()
