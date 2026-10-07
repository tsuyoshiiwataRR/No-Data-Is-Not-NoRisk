"""Data loading, node-feature layouts, and graph neighbourhoods.

The expected input files and their columns are documented in ``data/README.md``.
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import Data

SCHEMA_DOCUMENTATION = 'data/README.md'
EDGE_TYPE_IDS = (0, 1, 2, 5, 6, 7, 17, 34)
OTHER_COMPANY_TYPE = 'Other'
RARE_COMPANY_TYPE_THRESHOLD = 0.01
MAX_PROPAGATION_NEIGHBOURS = 100

NODE_COLUMNS = ('company_id',)
EDGE_COLUMNS = ('edge_id', 'company_a', 'company_b', 'reltype_id')
RISK_COLUMNS = ('company_id', 'year', 'risk_label')


@dataclass(frozen=True)
class Company:
    company_id: str
    company_name: str
    company_type: str | None = None


@dataclass(frozen=True)
class Relationship:
    edge_id: str
    company_a: str
    company_b: str
    reltype_id: int


@dataclass
class GraphDataset:
    """Validated in-memory representation of the three raw CSV tables."""

    companies: dict[str, Company]
    relationships: list[Relationship]
    risks: dict[tuple[str, int], int]
    edges_from_a: dict[str, list[int]]
    edges_from_b: dict[str, list[int]]


@dataclass(frozen=True)
class NodeFeatureSpec:
    """Node-feature layout: observed risk followed by one-hot company-type slots.

    The first slot is always ``Other``. It receives companies whose type is
    missing, rare in the training data, or unknown to the model.
    """

    company_types: tuple[str, ...] = (OTHER_COMPANY_TYPE,)

    def __post_init__(self) -> None:
        if not self.company_types or self.company_types[0] != OTHER_COMPANY_TYPE:
            raise ValueError(f'The first company-type slot must be {OTHER_COMPANY_TYPE!r}')
        if len(set(self.company_types)) != len(self.company_types):
            raise ValueError('Company-type slots must be unique')

    @property
    def size(self) -> int:
        return 1 + len(self.company_types)

    def company_type_position(self, company_type: str | None) -> int:
        """Return the feature column of a company's one-hot type indicator."""

        if company_type in self.company_types:
            return 1 + self.company_types.index(company_type)
        return 1


# The shipped research checkpoint used six company-type slots. Their original
# labels are not public, so public data can only populate the ``Other`` slot.
RESEARCH_CHECKPOINT_FEATURES = NodeFeatureSpec(
    (OTHER_COMPANY_TYPE, *(f'research_type_{index}' for index in range(1, 6)))
)


def _read_csv(path: Path, required_columns: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f'Required data file does not exist: {path} (see {SCHEMA_DOCUMENTATION})')
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in required_columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f'{path} is missing required column(s) {missing} (see {SCHEMA_DOCUMENTATION})')
        return list(reader)


def _parse_int(value: str, description: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f'{description} must be an integer, got {value!r}') from None


def load_graph_dataset(data_dir: str | Path) -> GraphDataset:
    """Load and validate nodes, relationships, and annual observed risks."""

    data_dir = Path(data_dir)
    node_rows = _read_csv(data_dir / 'nodes.csv', NODE_COLUMNS)
    edge_rows = _read_csv(data_dir / 'edges.csv', EDGE_COLUMNS)
    risk_rows = _read_csv(data_dir / 'risks.csv', RISK_COLUMNS)

    companies: dict[str, Company] = {}
    for row in node_rows:
        company_id = row['company_id']
        if not company_id or company_id in companies:
            raise ValueError(f'Duplicate or empty company_id: {company_id!r}')
        companies[company_id] = Company(
            company_id,
            row.get('company_name') or company_id,
            row.get('company_type') or None,
        )

    relationships: list[Relationship] = []
    edge_ids: set[str] = set()
    edges_from_a: dict[str, list[int]] = defaultdict(list)
    edges_from_b: dict[str, list[int]] = defaultdict(list)
    for row in edge_rows:
        edge_id = row['edge_id']
        company_a = row['company_a']
        company_b = row['company_b']
        reltype_id = _parse_int(row['reltype_id'], f'reltype_id of edge {edge_id!r}')
        if not edge_id or edge_id in edge_ids:
            raise ValueError(f'Duplicate or empty edge_id: {edge_id!r}')
        if company_a not in companies or company_b not in companies:
            raise ValueError(f'Edge {edge_id!r} refers to an unknown company')
        if reltype_id not in EDGE_TYPE_IDS:
            raise ValueError(
                f'Edge {edge_id!r} has unsupported reltype_id {reltype_id}; '
                f'use one of {EDGE_TYPE_IDS} and map other relationship types to 0 (Other)'
            )
        edge_ids.add(edge_id)
        relationships.append(Relationship(edge_id, company_a, company_b, reltype_id))
        edge_index = len(relationships) - 1
        edges_from_a[company_a].append(edge_index)
        edges_from_b[company_b].append(edge_index)

    risks: dict[tuple[str, int], int] = {}
    for row in risk_rows:
        company_id = row['company_id']
        key = (company_id, _parse_int(row['year'], f'year of risk observation for {company_id!r}'))
        risk_label = _parse_int(row['risk_label'], f'risk_label for {key!r}')
        if key[0] not in companies:
            raise ValueError(f'Risk observation refers to unknown company {key[0]!r}')
        if key in risks:
            raise ValueError(f'Duplicate risk observation for company/year {key!r}')
        if risk_label < 0:
            raise ValueError(f'Risk label must be non-negative for {key!r}')
        risks[key] = risk_label

    return GraphDataset(companies, relationships, risks, dict(edges_from_a), dict(edges_from_b))


def build_node_feature_spec(
    dataset: GraphDataset,
    rare_threshold: float = RARE_COMPANY_TYPE_THRESHOLD,
) -> NodeFeatureSpec:
    """Create one-hot slots for company types held by more than ``rare_threshold`` of companies.

    Matches the preprocessing of the shipped checkpoint, which folded rare types into ``Other``.
    Datasets without a ``company_type`` column produce the single ``Other`` slot.
    """

    counts = Counter(company.company_type for company in dataset.companies.values() if company.company_type)
    minimum_count = len(dataset.companies) * rare_threshold
    frequent = sorted(
        company_type
        for company_type, count in counts.items()
        if count > minimum_count and company_type != OTHER_COMPANY_TYPE
    )
    return NodeFeatureSpec((OTHER_COMPANY_TYPE, *frequent))


def labelled_company_ids(dataset: GraphDataset, year: int) -> list[str]:
    """Return companies with a risk observation in ``year``, i.e. eligible training targets."""

    return sorted(company_id for company_id in dataset.companies if (company_id, year) in dataset.risks)


def _capped_edges(edge_indices: list[int], max_neighbours: int | None, rng: random.Random) -> list[int]:
    if max_neighbours is None or len(edge_indices) <= max_neighbours:
        return edge_indices
    return rng.sample(edge_indices, max_neighbours)


def extract_neighbourhood(
    dataset: GraphDataset,
    target_id: str,
    feature_year: int,
    prediction_year: int,
    max_depth: int = 3,
    feature_spec: NodeFeatureSpec = RESEARCH_CHECKPOINT_FEATURES,
    max_neighbours: int | None = MAX_PROPAGATION_NEIGHBOURS,
    rng: random.Random | None = None,
) -> Data:
    """Build the target-centred graph used by HeteroGCN2.

    Node features are the observed ``feature_year`` risk followed by one-hot
    company-type slots from ``feature_spec``. The default spec matches the
    shipped research checkpoint. The target's own risk value is masked to
    prevent direct feature leakage, and ``y`` holds its ``prediction_year``
    observation.

    As when the shipped checkpoint was trained, each node contributes at most
    ``max_neighbours`` edges per direction on each hop. Larger edge lists are sampled with ``rng``
    (a fixed seed when omitted, so repeated predictions are identical).
    """

    import torch
    from torch_geometric.data import Data

    if target_id not in dataset.companies:
        raise KeyError(f'Unknown target company: {target_id!r}')
    rng = rng or random.Random(0)

    known_node_ids = {target_id}
    collected_a_to_b: list[int] = []
    collected_b_to_a: list[int] = []

    for _ in range(max_depth):
        candidate_a_to_b = [
            edge_index
            for node_id in sorted(known_node_ids)
            for edge_index in _capped_edges(dataset.edges_from_a.get(node_id, []), max_neighbours, rng)
            if dataset.relationships[edge_index].company_b not in known_node_ids
        ]
        candidate_b_to_a = [
            edge_index
            for node_id in sorted(known_node_ids)
            for edge_index in _capped_edges(dataset.edges_from_b.get(node_id, []), max_neighbours, rng)
            if dataset.relationships[edge_index].company_a not in known_node_ids
        ]
        newly_discovered = {dataset.relationships[index].company_b for index in candidate_a_to_b} | {
            dataset.relationships[index].company_a for index in candidate_b_to_a
        }
        newly_discovered -= known_node_ids
        collected_a_to_b.extend(candidate_a_to_b)
        collected_b_to_a.extend(candidate_b_to_a)
        known_node_ids.update(newly_discovered)
        if not newly_discovered:
            break

    node_ids = [target_id, *sorted(known_node_ids - {target_id})]
    node_to_local = {node_id: index for index, node_id in enumerate(node_ids)}

    x = torch.zeros((len(node_ids), feature_spec.size), dtype=torch.float32)
    for index, node_id in enumerate(node_ids):
        x[index, 0] = float(dataset.risks.get((node_id, feature_year), 0))
        x[index, feature_spec.company_type_position(dataset.companies[node_id].company_type)] = 1.0
    x[0, 0] = 0.0

    local_edges: list[tuple[int, int]] = []
    edge_attributes: list[list[float]] = []
    edge_type_to_position = {reltype_id: index for index, reltype_id in enumerate(EDGE_TYPE_IDS)}

    for edge_index in collected_a_to_b:
        edge = dataset.relationships[edge_index]
        local_edges.append((node_to_local[edge.company_b], node_to_local[edge.company_a]))
        attributes = [0.0] * len(EDGE_TYPE_IDS)
        attributes[edge_type_to_position[edge.reltype_id]] = 1.0
        edge_attributes.append(attributes)
    for edge_index in collected_b_to_a:
        edge = dataset.relationships[edge_index]
        local_edges.append((node_to_local[edge.company_a], node_to_local[edge.company_b]))
        attributes = [0.0] * len(EDGE_TYPE_IDS)
        attributes[edge_type_to_position[edge.reltype_id]] = -1.0
        edge_attributes.append(attributes)

    if local_edges:
        edge_index = torch.tensor(local_edges, dtype=torch.long).T.contiguous()
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(EDGE_TYPE_IDS)), dtype=torch.float32)

    observed_risk = dataset.risks.get((target_id, prediction_year))
    y = torch.tensor([float(observed_risk or 0)], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, node_ids=node_ids)
