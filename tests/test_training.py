import csv
import importlib.util
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'dummy'
HAS_ML_DEPENDENCIES = all(importlib.util.find_spec(name) for name in ('torch', 'torch_geometric'))
HAS_SKLEARN = importlib.util.find_spec('sklearn') is not None


def write_dataset(directory: Path, nodes: list[dict], edges: list[dict], risks: list[dict]) -> Path:
    for name, rows in (('nodes.csv', nodes), ('edges.csv', edges), ('risks.csv', risks)):
        with (directory / name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return directory


def typed_dataset(directory: Path) -> Path:
    """A fictional chain of eight companies with a company_type column."""

    company_types = ['Public', 'Private', 'Private', 'Fund', 'Public', 'Private', '', 'Public']
    nodes = [
        {'company_id': f'C{index}', 'company_name': f'Example {index}', 'company_type': company_type}
        for index, company_type in enumerate(company_types)
    ]
    edges = [
        {'edge_id': f'E{index}', 'company_a': f'C{index}', 'company_b': f'C{index + 1}', 'reltype_id': '5'}
        for index in range(len(nodes) - 1)
    ]
    risks = [
        {'company_id': f'C{index}', 'year': '2024', 'risk_label': str(int(index % 2 == 0))}
        for index in range(len(nodes))
    ]
    return write_dataset(directory, nodes, edges, risks)


@unittest.skipUnless(HAS_ML_DEPENDENCIES, 'torch and torch-geometric are not installed')
class TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / 'src'))

    def test_metrics_on_a_hand_checked_ranking(self) -> None:
        from risk_propagation.training import compute_metrics

        metrics = compute_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
        self.assertAlmostEqual((1 + 2 / 3) / 2, metrics['auc_pr'])
        self.assertAlmostEqual(0.75, metrics['auc_roc'])
        self.assertAlmostEqual(1.0, metrics['precision_at_10pct_recall'])
        self.assertAlmostEqual(0.5, metrics['recall_at_10pct_recall'])
        self.assertAlmostEqual((1 + 0.5) / (1 + 1 / math.log2(3)), metrics['ndcg'])

    def test_tied_scores_are_flagged_together(self) -> None:
        from risk_propagation.training import compute_metrics

        metrics = compute_metrics([0, 1, 0, 0], [0.5, 0.5, 0.1, 0.1])
        self.assertAlmostEqual(0.5, metrics['precision_at_10pct_recall'])
        self.assertAlmostEqual(0.5, metrics['precision_at_top_10pct'])
        self.assertAlmostEqual(2.5 / 3, metrics['auc_roc'])

    def test_single_class_metrics_are_zero(self) -> None:
        from risk_propagation.training import compute_metrics

        self.assertTrue(all(value == 0.0 for value in compute_metrics([0, 0, 0], [0.1, 0.5, 0.2]).values()))

    @unittest.skipUnless(HAS_SKLEARN, 'scikit-learn is not installed')
    def test_metrics_match_scikit_learn_with_ties(self) -> None:
        from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score

        from risk_propagation.training import compute_metrics

        rng = random.Random(7)
        for _ in range(20):
            labels = [int(rng.random() < 0.3) for _ in range(40)]
            labels[0], labels[1] = 1, 0
            scores = [round(rng.random(), 1) for _ in labels]
            metrics = compute_metrics(labels, scores)
            self.assertAlmostEqual(average_precision_score(labels, scores), metrics['auc_pr'])
            self.assertAlmostEqual(roc_auc_score(labels, scores), metrics['auc_roc'])
            self.assertAlmostEqual(ndcg_score([labels], [scores]), metrics['ndcg'])
            self.assertAlmostEqual(ndcg_score([labels], [scores], k=4), metrics['ndcg_at_10pct'])

    def test_pu_loss_matches_cross_entropy_terms_and_reverses_negative_risk(self) -> None:
        import torch
        import torch.nn.functional as F

        from risk_propagation.training import pu_loss

        logits = torch.tensor([2.0, -1.0, 0.5])
        labels = torch.tensor([1.0, 0.0, 0.0])
        expected = 0.12 * F.softplus(-logits[0]) + (F.softplus(logits[1:]).mean() - 0.12 * F.softplus(logits[0]))
        self.assertAlmostEqual(expected.item(), pu_loss(logits, labels, positive_prior=0.12).item(), places=6)

        confident_positive = torch.tensor([-6.0, -6.0], requires_grad=True)
        loss = pu_loss(confident_positive, torch.tensor([1.0, 0.0]), positive_prior=0.9)
        loss.backward()
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(confident_positive.grad).all().item())

    def test_split_is_stratified_disjoint_and_reproducible(self) -> None:
        from risk_propagation.data import labelled_company_ids, load_graph_dataset
        from risk_propagation.training import split_targets

        dataset = load_graph_dataset(DATA_DIR)
        split = split_targets(dataset, 2024, val_fraction=0.2, seed=42)
        self.assertEqual(split, split_targets(dataset, 2024, val_fraction=0.2, seed=42))
        self.assertNotEqual(split, split_targets(dataset, 2024, val_fraction=0.2, seed=1))
        self.assertFalse(set(split.train_ids) & set(split.validation_ids))
        self.assertEqual(set(labelled_company_ids(dataset, 2024)), set(split.train_ids) | set(split.validation_ids))
        for ids in (split.train_ids, split.validation_ids):
            labels = {dataset.risks[company_id, 2024] > 0 for company_id in ids}
            self.assertEqual({True, False}, labels)

    def test_split_rejects_a_year_without_enough_positives(self) -> None:
        from risk_propagation.data import load_graph_dataset
        from risk_propagation.training import split_targets

        with self.assertRaisesRegex(ValueError, 'Year 2030 needs at least two companies'):
            split_targets(load_graph_dataset(DATA_DIR), 2030, val_fraction=0.2, seed=42)

    def test_company_types_become_one_hot_slots_with_other_first(self) -> None:
        from risk_propagation.data import build_node_feature_spec, extract_neighbourhood, load_graph_dataset

        with tempfile.TemporaryDirectory() as directory:
            dataset = load_graph_dataset(typed_dataset(Path(directory)))
        spec = build_node_feature_spec(dataset)
        self.assertEqual(('Other', 'Fund', 'Private', 'Public'), spec.company_types)
        self.assertEqual(5, spec.size)

        graph = extract_neighbourhood(dataset, 'C6', 2024, 2024, max_depth=1, feature_spec=spec)
        self.assertEqual(['C6', 'C5', 'C7'], graph.node_ids)
        self.assertEqual([0.0, 1.0, 0.0, 0.0, 0.0], graph.x[0].tolist())
        self.assertEqual([0.0, 0.0, 0.0, 1.0, 0.0], graph.x[1].tolist())
        self.assertEqual([0.0, 0.0, 0.0, 0.0, 1.0], graph.x[2].tolist())
        self.assertEqual([1.0], graph.y.tolist())

    def test_dummy_nodes_produce_a_single_other_slot(self) -> None:
        from risk_propagation.data import build_node_feature_spec, load_graph_dataset

        self.assertEqual(('Other',), build_node_feature_spec(load_graph_dataset(DATA_DIR)).company_types)

    def test_neighbour_sampling_caps_edges_per_company(self) -> None:
        from risk_propagation.data import extract_neighbourhood, load_graph_dataset

        with tempfile.TemporaryDirectory() as directory:
            nodes = [{'company_id': 'HUB'}, *({'company_id': f'S{index}'} for index in range(10))]
            edges = [
                {'edge_id': f'E{index}', 'company_a': 'HUB', 'company_b': f'S{index}', 'reltype_id': '5'}
                for index in range(10)
            ]
            risks = [{'company_id': 'HUB', 'year': '2024', 'risk_label': '0'}]
            dataset = load_graph_dataset(write_dataset(Path(directory), nodes, edges, risks))

        capped = extract_neighbourhood(dataset, 'HUB', 2024, 2024, max_depth=1, max_neighbours=4)
        self.assertEqual(5, capped.num_nodes)
        self.assertEqual('S0', dataset.companies['S0'].company_name)
        repeated = extract_neighbourhood(dataset, 'HUB', 2024, 2024, max_depth=1, max_neighbours=4)
        self.assertEqual(capped.node_ids, repeated.node_ids)
        self.assertEqual(11, extract_neighbourhood(dataset, 'HUB', 2024, 2024, max_depth=1).num_nodes)

    def test_missing_columns_point_to_the_schema_documentation(self) -> None:
        from risk_propagation.data import load_graph_dataset

        with tempfile.TemporaryDirectory() as directory:
            write_dataset(
                Path(directory),
                [{'company_id': 'C1'}],
                [{'edge_id': 'E1', 'company_a': 'C1', 'company_b': 'C1', 'reltype_id': '0'}],
                [{'company_id': 'C1', 'year': '2024', 'risk': '1'}],
            )
            with self.assertRaisesRegex(ValueError, r"missing required column\(s\) \['risk_label'\].*data/README.md"):
                load_graph_dataset(directory)

    def test_unsupported_relationship_type_explains_mapping_to_other(self) -> None:
        from risk_propagation.data import load_graph_dataset

        with tempfile.TemporaryDirectory() as directory:
            write_dataset(
                Path(directory),
                [{'company_id': 'C1'}, {'company_id': 'C2'}],
                [{'edge_id': 'E1', 'company_a': 'C1', 'company_b': 'C2', 'reltype_id': '99'}],
                [{'company_id': 'C1', 'year': '2024', 'risk_label': '1'}],
            )
            with self.assertRaisesRegex(ValueError, 'map other relationship types to 0'):
                load_graph_dataset(directory)

    def test_training_writes_a_checkpoint_usable_for_prediction(self) -> None:
        from risk_propagation.model import HeteroGCN2, load_checkpoint
        from risk_propagation.prediction import predict_risks
        from risk_propagation.training import TrainingConfig, train_model

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / 'run'
            result = train_model(
                TrainingConfig(
                    data_dir=DATA_DIR,
                    year=2024,
                    output_dir=output_dir,
                    nb_layers=2,
                    hidden_size=8,
                    batch_size=16,
                    epochs=3,
                )
            )

            self.assertEqual(output_dir / 'model.pth', result.model_path)
            self.assertEqual(3, len(result.evaluations))
            metadata = json.loads((output_dir / 'model.json').read_text(encoding='utf-8'))
            self.assertEqual('HeteroGCN2', metadata['architecture'])
            self.assertEqual([0, 1, 2, 5, 6, 7, 17, 34], metadata['edge_type_ids'])
            self.assertEqual(['Other'], metadata['node_features']['company_types'])
            self.assertEqual(2024, metadata['training_data']['label_year'])
            self.assertEqual(result.best_metrics, metadata['best_validation_metrics'])

            with (output_dir / 'validation_predictions.csv').open(newline='', encoding='utf-8') as handle:
                validation_rows = list(csv.DictReader(handle))
            self.assertEqual(sorted(result.split.validation_ids), sorted(row['company_id'] for row in validation_rows))

            model, feature_spec = load_checkpoint(result.model_path)
            self.assertIsInstance(model, HeteroGCN2)
            self.assertEqual(2, model.nb_layers)
            predictions = predict_risks(model, DATA_DIR, 2025, feature_year=2024, feature_spec=feature_spec)
            self.assertEqual(54, len(predictions))
            self.assertTrue(all(0.0 <= prediction.predicted_risk <= 1.0 for prediction in predictions))

    def test_training_with_company_types_round_trips_the_feature_layout(self) -> None:
        from risk_propagation.model import load_checkpoint
        from risk_propagation.training import TrainingConfig, train_model

        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / 'data'
            data_dir.mkdir()
            result = train_model(
                TrainingConfig(
                    data_dir=typed_dataset(data_dir),
                    year=2024,
                    output_dir=Path(directory) / 'run',
                    hidden_size=4,
                    epochs=1,
                    val_fraction=0.25,
                )
            )
            model, feature_spec = load_checkpoint(result.model_path)
            self.assertEqual(result.feature_spec, feature_spec)
            self.assertEqual(5, model.first_layer.in_features)

    def test_feature_layout_mismatch_is_reported(self) -> None:
        import shutil

        from risk_propagation.model import load_checkpoint

        shipped_dir = PROJECT_ROOT / 'models' / 'heterogcn2_puloss_3layers'
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / 'model.pth'
            shutil.copy(shipped_dir / 'model.pth', model_path)
            with self.assertRaisesRegex(FileNotFoundError, 'model.json'):
                load_checkpoint(model_path)

            metadata = json.loads((shipped_dir / 'model.json').read_text(encoding='utf-8'))
            metadata['node_features']['company_types'] = ['Other']
            (Path(directory) / 'model.json').write_text(json.dumps(metadata), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'expects 7 node features'):
                load_checkpoint(model_path)

    def test_warns_when_positive_prior_is_below_the_observed_positive_share(self) -> None:
        from risk_propagation.training import TrainingConfig, train_model

        with tempfile.TemporaryDirectory() as directory:
            config = TrainingConfig(data_dir=DATA_DIR, year=2024, output_dir=directory, hidden_size=4, epochs=1)
            with self.assertLogs('risk_propagation.training', level='WARNING') as logs:
                train_model(config)
        self.assertIn('above --positive-prior 0.12', '\n'.join(logs.output))

    def test_invalid_configuration_is_rejected(self) -> None:
        from risk_propagation.training import TrainingConfig

        with self.assertRaisesRegex(ValueError, 'loss must be one of'):
            TrainingConfig(data_dir=DATA_DIR, year=2024, loss='mse')
        with self.assertRaisesRegex(ValueError, 'val_fraction'):
            TrainingConfig(data_dir=DATA_DIR, year=2024, val_fraction=1.0)
        self.assertEqual(2024, TrainingConfig(data_dir=DATA_DIR, year=2024).feature_year)

    def test_command_line_reports_data_errors_without_a_traceback(self) -> None:
        from risk_propagation.training import main

        with tempfile.TemporaryDirectory() as directory, self.assertRaises(SystemExit) as raised:
            main(['--data-dir', directory, '--year', '2024'])
        self.assertEqual(2, raised.exception.code)


if __name__ == '__main__':
    unittest.main()
