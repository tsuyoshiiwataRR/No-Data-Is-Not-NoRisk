import csv
import unittest
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / 'data' / 'dummy'
EXPECTED_RELATIONSHIPS = {
    0: 'Other',
    1: 'Current Investment',
    2: 'Prior Investment',
    5: 'Current Subsidiary',
    6: 'Prior Subsidiary',
    7: 'Merged Entity',
    17: 'Current Fund Sponsor',
    34: 'Current Fund Investment Advisor',
}
EXPECTED_YEARS = set(range(2021, 2026))


def read_csv(name: str) -> tuple[list[str], list[dict[str, str]]]:
    with (DATA_DIR / name).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


class DummyDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.node_fields, cls.nodes = read_csv('nodes.csv')
        cls.edge_fields, cls.edges = read_csv('edges.csv')
        cls.risk_fields, cls.risks = read_csv('risks.csv')
        cls.node_ids = {row['company_id'] for row in cls.nodes}

    def test_nodes_only_contain_identifier_and_name(self) -> None:
        self.assertEqual(['company_id', 'company_name'], self.node_fields)
        self.assertEqual(54, len(self.nodes))
        self.assertEqual(len(self.nodes), len(self.node_ids))
        self.assertTrue(all(row['company_name'] for row in self.nodes))

    def test_edges_use_all_supported_relationship_types(self) -> None:
        self.assertEqual(
            ['edge_id', 'company_a', 'company_b', 'reltype_id', 'relationship_type'],
            self.edge_fields,
        )
        edge_ids = [row['edge_id'] for row in self.edges]
        self.assertEqual(len(edge_ids), len(set(edge_ids)))

        seen_relationships = set()
        for edge in self.edges:
            self.assertIn(edge['company_a'], self.node_ids)
            self.assertIn(edge['company_b'], self.node_ids)
            self.assertNotEqual(edge['company_a'], edge['company_b'])
            relationship_id = int(edge['reltype_id'])
            self.assertEqual(EXPECTED_RELATIONSHIPS[relationship_id], edge['relationship_type'])
            seen_relationships.add(relationship_id)

        self.assertEqual(set(EXPECTED_RELATIONSHIPS), seen_relationships)

    def test_graph_is_connected_when_relationship_direction_is_ignored(self) -> None:
        neighbours = {company_id: set() for company_id in self.node_ids}
        for edge in self.edges:
            neighbours[edge['company_a']].add(edge['company_b'])
            neighbours[edge['company_b']].add(edge['company_a'])

        reachable = {next(iter(self.node_ids))}
        frontier = list(reachable)
        while frontier:
            company_id = frontier.pop()
            newly_reachable = neighbours[company_id] - reachable
            reachable.update(newly_reachable)
            frontier.extend(newly_reachable)

        self.assertEqual(self.node_ids, reachable)

    def test_verdant_harbor_is_directly_connected_to_iron_peak_operations(self) -> None:
        self.assertIn(
            ('H002', 'R501', '1', 'Current Investment'),
            {
                (edge['company_a'], edge['company_b'], edge['reltype_id'], edge['relationship_type'])
                for edge in self.edges
            },
        )

    def test_every_company_has_one_risk_value_for_each_year(self) -> None:
        self.assertEqual(['company_id', 'year', 'risk_label'], self.risk_fields)
        self.assertEqual(len(self.node_ids) * len(EXPECTED_YEARS), len(self.risks))

        years_by_company: dict[str, set[int]] = defaultdict(set)
        observations = set()
        for risk in self.risks:
            company_id = risk['company_id']
            year = int(risk['year'])
            risk_label = int(risk['risk_label'])
            self.assertIn(company_id, self.node_ids)
            self.assertIn(year, EXPECTED_YEARS)
            self.assertGreaterEqual(risk_label, 0)
            self.assertNotIn((company_id, year), observations)
            observations.add((company_id, year))
            years_by_company[company_id].add(year)

        self.assertEqual(self.node_ids, set(years_by_company))
        self.assertTrue(all(years == EXPECTED_YEARS for years in years_by_company.values()))

    def test_dataset_has_both_visible_and_invisible_companies(self) -> None:
        maximum_risk_by_company = defaultdict(int)
        for risk in self.risks:
            company_id = risk['company_id']
            maximum_risk_by_company[company_id] = max(maximum_risk_by_company[company_id], int(risk['risk_label']))

        visible = {company_id for company_id, value in maximum_risk_by_company.items() if value > 0}
        invisible = self.node_ids - visible
        self.assertTrue(visible)
        self.assertTrue(invisible)

    def test_silver_birch_is_a_mixed_history_low_recent_risk_area(self) -> None:
        recent_years = (2024, 2025)
        neighbours = {'L402', 'L403', 'L404', 'L405', 'L409', 'L413'}
        risk_by_company_year = {(row['company_id'], int(row['year'])): int(row['risk_label']) for row in self.risks}

        self.assertEqual(neighbours, {edge['company_b'] for edge in self.edges if edge['company_a'] == 'L401'})
        self.assertTrue(any(risk_by_company_year[company_id, year] > 0 for company_id in neighbours for year in (2021, 2022, 2023)))
        self.assertTrue(any(all(risk_by_company_year[company_id, year] == 0 for year in EXPECTED_YEARS) for company_id in neighbours))
        self.assertTrue(all(risk_by_company_year[company_id, year] == 0 for company_id in neighbours for year in recent_years))

    def test_iron_peak_has_many_recently_observed_risk_neighbours(self) -> None:
        recent_years = (2024, 2025)
        neighbours = {'R502', 'R503', 'R504', 'R505', 'R510', 'R513'}
        risk_by_company_year = {(row['company_id'], int(row['year'])): int(row['risk_label']) for row in self.risks}

        self.assertEqual(neighbours, {edge['company_b'] for edge in self.edges if edge['company_a'] == 'R501'})
        recently_observed = {
            company_id
            for company_id in neighbours
            if all(risk_by_company_year[company_id, year] > 0 for year in recent_years)
        }
        self.assertGreaterEqual(len(recently_observed), 5)


if __name__ == '__main__':
    unittest.main()
