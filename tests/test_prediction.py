import importlib.util
import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
HAS_ML_DEPENDENCIES = all(importlib.util.find_spec(name) for name in ('torch', 'torch_geometric'))


@unittest.skipUnless(HAS_ML_DEPENDENCIES, 'torch and torch-geometric are not installed')
class PredictionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / 'src'))
        from risk_propagation.data import load_graph_dataset
        from risk_propagation.model import load_checkpoint

        cls.data_dir = PROJECT_ROOT / 'data' / 'dummy'
        cls.model_path = PROJECT_ROOT / 'models' / 'heterogcn2_puloss_3layers' / 'model.pth'
        cls.dataset = load_graph_dataset(cls.data_dir)
        cls.model, cls.feature_spec = load_checkpoint(cls.model_path)

    def test_checkpoint_dimensions_are_loaded(self) -> None:
        from risk_propagation.model import HeteroGCN2

        self.assertIsInstance(self.model, HeteroGCN2)
        self.assertEqual((64, 7), tuple(self.model.first_layer.weight.shape))
        self.assertEqual(3, self.model.nb_layers)
        self.assertEqual(8, self.model.num_edge_types)
        self.assertEqual(16, len(self.model.conv_0.convs))
        self.assertEqual(7, self.feature_spec.size)

    def test_scores_do_not_depend_on_batch_composition(self) -> None:
        import torch
        from torch_geometric.data import Batch

        from risk_propagation.data import extract_neighbourhood

        graphs = [
            extract_neighbourhood(self.dataset, company_id, 2024, 2025) for company_id in ('A101', 'R501', 'L401')
        ]
        batch = Batch.from_data_list(graphs)
        with torch.no_grad():
            batched = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)[:, 0]
            single = torch.cat(
                [
                    self.model(
                        graph.x, graph.edge_index, graph.edge_attr, torch.zeros(graph.num_nodes, dtype=torch.long)
                    )[:, 0]
                    for graph in graphs
                ]
            )
        self.assertTrue(torch.allclose(batched, single, atol=1e-5))

    def test_relationship_types_change_the_prediction(self) -> None:
        import torch

        from risk_propagation.data import extract_neighbourhood

        graph = extract_neighbourhood(self.dataset, 'R501', 2024, 2025)
        batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        reversed_attr = -graph.edge_attr
        with torch.no_grad():
            original = self.model(graph.x, graph.edge_index, graph.edge_attr, batch)
            reversed_direction = self.model(graph.x, graph.edge_index, reversed_attr, batch)
        self.assertFalse(torch.allclose(original, reversed_direction))

    def test_neighbourhood_masks_target_risk_and_uses_other_type(self) -> None:
        from risk_propagation.data import extract_neighbourhood

        graph = extract_neighbourhood(self.dataset, 'N304', feature_year=2025, prediction_year=2025)
        self.assertEqual(0.0, graph.x[0, 0].item())
        self.assertTrue((graph.x[:, 1] == 1).all().item())
        self.assertTrue((graph.x[:, 2:] == 0).all().item())
        self.assertTrue((graph.x[1:, 0] > 0).any().item())

    def test_predicts_a_finite_probability_for_every_company(self) -> None:
        from risk_propagation.prediction import predict_risks

        predictions = predict_risks(self.model, self.data_dir, prediction_year=2025)
        self.assertEqual(len(self.dataset.companies), len(predictions))
        for prediction in predictions:
            self.assertEqual(2025, prediction.feature_year)
            self.assertEqual(2025, prediction.prediction_year)
            self.assertTrue(math.isfinite(prediction.predicted_risk))
            self.assertGreaterEqual(prediction.predicted_risk, 0.0)
            self.assertLessEqual(prediction.predicted_risk, 1.0)


if __name__ == '__main__':
    unittest.main()
