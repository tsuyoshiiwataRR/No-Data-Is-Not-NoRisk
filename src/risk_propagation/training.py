"""Local HeteroGCN2 training: data splits, losses, ranking metrics, and checkpoint writing.

The input files must follow the schema documented in ``data/README.md``.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from risk_propagation.data import (
    EDGE_TYPE_IDS,
    MAX_PROPAGATION_NEIGHBOURS,
    SCHEMA_DOCUMENTATION,
    GraphDataset,
    NodeFeatureSpec,
    build_node_feature_spec,
    extract_neighbourhood,
    labelled_company_ids,
    load_graph_dataset,
)
from risk_propagation.model import ARCHITECTURE, MODEL_METADATA_FILENAME, HeteroGCN2

logger = logging.getLogger(__name__)

ESTIMATED_POSITIVE_PRIOR = 0.12
# Checkpoints are selected on precision at 10% recall; AUC-PR breaks ties, which
# are common on small validation sets.
SELECTION_METRICS = ('precision_at_10pct_recall', 'auc_pr')
DEFAULT_OUTPUT_ROOT = Path('artifacts') / 'training'
METADATA_FORMAT_VERSION = 1


@dataclass
class TrainingConfig:
    """Training settings. Defaults reproduce the shipped research checkpoint."""

    data_dir: Path
    year: int
    feature_year: int | None = None
    output_dir: Path | None = None
    loss: str = 'pu'
    nb_layers: int = 3
    hidden_size: int = 64
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 2
    eval_every: int = 1000
    patience: int = 4
    positive_prior: float = ESTIMATED_POSITIVE_PRIOR
    val_fraction: float = 0.2
    max_neighbours: int = MAX_PROPAGATION_NEIGHBOURS
    seed: int = 42
    device: str = 'cpu'

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = None if self.output_dir is None else Path(self.output_dir)
        if self.feature_year is None:
            self.feature_year = self.year
        if self.loss not in LOSSES:
            raise ValueError(f'loss must be one of {sorted(LOSSES)}, got {self.loss!r}')
        if not 0 < self.val_fraction < 1:
            raise ValueError('val_fraction must be between 0 and 1')
        if not 0 < self.positive_prior < 1:
            raise ValueError('positive_prior must be between 0 and 1')
        if self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive')
        for name in ('nb_layers', 'hidden_size', 'batch_size', 'epochs', 'eval_every', 'patience', 'max_neighbours'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be at least 1')


@dataclass(frozen=True)
class TargetSplit:
    train_ids: list[str]
    validation_ids: list[str]


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    model_path: Path
    feature_spec: NodeFeatureSpec
    split: TargetSplit
    best_metrics: dict[str, float]
    evaluations: list[dict[str, float]]


def pu_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_prior: float = ESTIMATED_POSITIVE_PRIOR,
    beta: float = 0.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Non-negative positive-unlabelled (nnPU) loss with gradient reversal.

    A label of 1 is an observed positive; 0 is unlabelled rather than negative.
    ``positive_prior`` is the estimated share of truly risky companies,
    including those without an observed incident.
    """

    logits_p = logits[labels == 1]
    logits_u = logits[labels == 0]
    zero = logits.sum() * 0.0

    if logits_p.numel() > 0:
        risk_p_plus = F.binary_cross_entropy_with_logits(logits_p, torch.ones_like(logits_p))
        risk_p_minus = F.binary_cross_entropy_with_logits(logits_p, torch.zeros_like(logits_p))
    else:
        risk_p_plus = risk_p_minus = zero

    if logits_u.numel() > 0:
        risk_u_minus = F.binary_cross_entropy_with_logits(logits_u, torch.zeros_like(logits_u))
    else:
        risk_u_minus = zero

    negative_risk = risk_u_minus - positive_prior * risk_p_minus
    if negative_risk.item() >= -beta:
        return positive_prior * risk_p_plus + negative_risk
    return -gamma * negative_risk


def ce_loss(
    logits: torch.Tensor, labels: torch.Tensor, positive_prior: float = ESTIMATED_POSITIVE_PRIOR
) -> torch.Tensor:
    """Binary cross-entropy that treats unlabelled companies as negatives."""

    return F.binary_cross_entropy_with_logits(logits, labels)


LOSSES: dict[str, Callable[..., torch.Tensor]] = {'pu': pu_loss, 'ce': ce_loss}


def _ranked_tie_groups(labels: Sequence[int], scores: Sequence[float]) -> list[tuple[int, int]]:
    """Return ``(size, positives)`` for each group of equal scores, highest score first."""

    groups: list[list[float]] = []
    for index in sorted(range(len(scores)), key=lambda item: scores[item], reverse=True):
        if groups and scores[index] == groups[-1][0]:
            groups[-1][1] += 1
            groups[-1][2] += labels[index]
        else:
            groups.append([scores[index], 1, labels[index]])
    return [(int(size), int(positives)) for _, size, positives in groups]


def _ndcg(groups: list[tuple[int, int]], n_positive: int, k: int) -> float:
    discounts = [1.0 / math.log2(position + 2) for position in range(k)]
    dcg = 0.0
    position = 0
    for size, positives in groups:
        if position >= k:
            break
        dcg += positives / size * sum(discounts[position : position + size])
        position += size
    return dcg / sum(discounts[: min(n_positive, k)])


def _precision_recall_until(
    groups: list[tuple[int, int]], n_positive: int, reached: Callable[[int, int], bool]
) -> tuple[float, float]:
    """Flag tie groups from the top until ``reached(true_positives, flagged)`` holds."""

    true_positives = flagged = 0
    for size, positives in groups:
        true_positives += positives
        flagged += size
        if reached(true_positives, flagged):
            break
    return true_positives / flagged, true_positives / n_positive


def compute_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    """Ranking metrics used for model selection.

    ``precision_at_10pct_recall`` is the precision among the highest-ranked
    companies needed to recover 10% of observed positives. ``*_top_10pct``
    metrics flag the highest-ranked 10% of companies. Companies with tied
    scores are always flagged together. All metrics are 0 when the labels
    contain a single class.
    """

    if len(labels) != len(scores):
        raise ValueError('labels and scores must have the same length')
    labels = [int(label > 0) for label in labels]
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    names = (
        'auc_pr',
        'auc_roc',
        'ndcg',
        'ndcg_at_10pct',
        'precision_at_10pct_recall',
        'recall_at_10pct_recall',
        'precision_at_top_10pct',
        'recall_at_top_10pct',
    )
    if n_positive == 0 or n_negative == 0:
        return dict.fromkeys(names, 0.0)

    groups = _ranked_tie_groups(labels, scores)
    top_k = max(1, int(len(labels) * 0.1))

    average_precision = 0.0
    true_positives = flagged = 0
    for size, positives in groups:
        true_positives += positives
        flagged += size
        average_precision += positives / n_positive * true_positives / flagged

    positive_rank_sum = 0.0
    rank_start = 1
    for size, positives in reversed(groups):
        positive_rank_sum += positives * (rank_start + (size - 1) / 2)
        rank_start += size
    auc_roc = (positive_rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)

    precision_at_recall, recall_at_recall = _precision_recall_until(
        groups, n_positive, lambda true_positives, _: true_positives / n_positive >= 0.1
    )
    precision_top, recall_top = _precision_recall_until(groups, n_positive, lambda _, flagged: flagged >= top_k)

    return dict(
        zip(
            names,
            (
                average_precision,
                auc_roc,
                _ndcg(groups, n_positive, len(labels)),
                _ndcg(groups, n_positive, top_k),
                precision_at_recall,
                recall_at_recall,
                precision_top,
                recall_top,
            ),
        )
    )


def split_targets(dataset: GraphDataset, year: int, val_fraction: float, seed: int) -> TargetSplit:
    """Split companies observed in ``year`` into train and validation targets.

    The split is seeded and stratified so both sets contain observed positives.
    """

    company_ids = labelled_company_ids(dataset, year)
    positives = [company_id for company_id in company_ids if dataset.risks[company_id, year] > 0]
    unlabelled = [company_id for company_id in company_ids if dataset.risks[company_id, year] == 0]
    if len(positives) < 2 or len(unlabelled) < 2:
        raise ValueError(
            f'Year {year} needs at least two companies with risk_label > 0 and two with risk_label = 0 in '
            f'risks.csv to create train and validation sets; found {len(positives)} and {len(unlabelled)}'
        )

    rng = random.Random(seed)
    train_ids: list[str] = []
    validation_ids: list[str] = []
    for group in (positives, unlabelled):
        rng.shuffle(group)
        n_validation = min(len(group) - 1, max(1, round(len(group) * val_fraction)))
        validation_ids.extend(group[:n_validation])
        train_ids.extend(group[n_validation:])
    return TargetSplit(sorted(train_ids), sorted(validation_ids))


class NeighbourhoodDataset(Dataset):
    """Extract a target company's neighbourhood each time it is requested."""

    def __init__(
        self,
        dataset: GraphDataset,
        target_ids: list[str],
        feature_year: int,
        label_year: int,
        max_depth: int,
        feature_spec: NodeFeatureSpec,
        max_neighbours: int,
        rng: random.Random | None = None,
    ) -> None:
        self.dataset = dataset
        self.target_ids = target_ids
        self.feature_year = feature_year
        self.label_year = label_year
        self.max_depth = max_depth
        self.feature_spec = feature_spec
        self.max_neighbours = max_neighbours
        self.rng = rng

    def __len__(self) -> int:
        return len(self.target_ids)

    def __getitem__(self, index: int):
        return extract_neighbourhood(
            self.dataset,
            self.target_ids[index],
            self.feature_year,
            self.label_year,
            max_depth=self.max_depth,
            feature_spec=self.feature_spec,
            max_neighbours=self.max_neighbours,
            rng=self.rng,
        )


def predict_scores(model: HeteroGCN2, loader: DataLoader, device: torch.device) -> tuple[list[float], list[int]]:
    """Return predicted risk probabilities and observed binary labels for each target."""

    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)[:, 0]
            scores.extend(torch.sigmoid(logits).tolist())
            labels.extend((batch.y > 0).int().tolist())
    return scores, labels


def default_output_dir(config: TrainingConfig) -> Path:
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    return (
        DEFAULT_OUTPUT_ROOT
        / f'heterogcn2_{config.loss}_layers_{config.nb_layers}_size_{config.hidden_size}_{timestamp}'
    )


def _model_metadata(
    config: TrainingConfig,
    feature_spec: NodeFeatureSpec,
    training_data: dict[str, int | str],
    best_step: int,
    best_metrics: dict[str, float],
    evaluations: list[dict[str, float]],
) -> dict:
    hyperparameters = {
        key: value
        for key, value in asdict(config).items()
        if key not in {'data_dir', 'year', 'feature_year', 'output_dir', 'device'}
    }
    return {
        'format_version': METADATA_FORMAT_VERSION,
        'architecture': ARCHITECTURE,
        'node_features': {
            'columns': ['observed_risk', *(f'company_type={name}' for name in feature_spec.company_types)],
            'company_types': list(feature_spec.company_types),
        },
        'edge_type_ids': list(EDGE_TYPE_IDS),
        'hyperparameters': hyperparameters,
        'training_data': training_data,
        'selection_metrics': list(SELECTION_METRICS),
        'best_step': best_step,
        'best_validation_metrics': best_metrics,
        'evaluations': evaluations,
        'created_at': datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec='seconds'),
    }


def _write_checkpoint(output_dir: Path, state_dict: dict[str, torch.Tensor], metadata: dict) -> Path:
    model_path = output_dir / 'model.pth'
    torch.save(state_dict, model_path)
    (output_dir / MODEL_METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return model_path


def _write_validation_predictions(
    path: Path, dataset: GraphDataset, company_ids: list[str], label_year: int, scores: list[float]
) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['company_id', 'company_name', 'label_year', 'predicted_risk', 'observed_risk_label'])
        for company_id, score in sorted(zip(company_ids, scores), key=lambda item: item[1], reverse=True):
            writer.writerow(
                [
                    company_id,
                    dataset.companies[company_id].company_name,
                    label_year,
                    f'{score:.6f}',
                    dataset.risks[company_id, label_year],
                ]
            )


def train_model(config: TrainingConfig) -> TrainingResult:
    """Train HeteroGCN2 on one label year and keep the best validation checkpoint."""

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    dataset = load_graph_dataset(config.data_dir)
    if not any(year == config.feature_year for _, year in dataset.risks):
        logger.warning('risks.csv has no %d observations; every neighbour risk feature will be 0', config.feature_year)
    feature_spec = build_node_feature_spec(dataset)
    split = split_targets(dataset, config.year, config.val_fraction, config.seed)
    training_data = {
        'data_dir': str(config.data_dir),
        'label_year': config.year,
        'feature_year': config.feature_year,
        'companies': len(dataset.companies),
        'relationships': len(dataset.relationships),
        'train_targets': len(split.train_ids),
        'train_positives': sum(dataset.risks[company_id, config.year] > 0 for company_id in split.train_ids),
        'validation_targets': len(split.validation_ids),
        'validation_positives': sum(dataset.risks[company_id, config.year] > 0 for company_id in split.validation_ids),
    }
    logger.info(
        'Loaded %d companies and %d relationships; %d train / %d validation targets for %d (features from %d)',
        training_data['companies'],
        training_data['relationships'],
        training_data['train_targets'],
        training_data['validation_targets'],
        config.year,
        config.feature_year,
    )
    logger.info('Node features: %s', ['observed_risk', *feature_spec.company_types])
    observed_positive_share = training_data['train_positives'] / training_data['train_targets']
    if config.loss == 'pu' and observed_positive_share > config.positive_prior:
        logger.warning(
            '%.0f%% of training targets are observed positives, above --positive-prior %.2f. The prior is the '
            'true share of risky companies, so it should be at least the observed share; the PU loss will '
            'otherwise push all scores down',
            100 * observed_positive_share,
            config.positive_prior,
        )

    def make_dataset(target_ids: list[str], rng: random.Random | None) -> NeighbourhoodDataset:
        return NeighbourhoodDataset(
            dataset,
            target_ids,
            config.feature_year,
            config.year,
            config.nb_layers,
            feature_spec,
            config.max_neighbours,
            rng,
        )

    train_loader = DataLoader(
        make_dataset(split.train_ids, random.Random(config.seed)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    # Validation neighbourhoods use a fixed sample so evaluations are comparable.
    validation_loader = DataLoader(make_dataset(split.validation_ids, None), batch_size=config.batch_size)

    model = HeteroGCN2(feature_spec.size, config.hidden_size, config.nb_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = LOSSES[config.loss]

    output_dir = config.output_dir or default_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info('Writing outputs to %s', output_dir)

    best_score: tuple[float, ...] = (-1.0,)
    best_step = 0
    best_metrics: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] = {}
    evaluations: list[dict[str, float]] = []
    evaluations_without_improvement = 0
    losses_since_evaluation: list[float] = []
    step = 0
    stop = False

    for epoch in range(1, config.epochs + 1):
        for batch_index, batch in enumerate(train_loader, start=1):
            model.train()
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)[:, 0]
            loss = criterion(logits, (batch.y > 0).float(), positive_prior=config.positive_prior)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            losses_since_evaluation.append(loss.item())

            if step % config.eval_every != 0 and batch_index != len(train_loader):
                continue

            scores, labels = predict_scores(model, validation_loader, device)
            metrics = compute_metrics(labels, scores)
            train_loss = sum(losses_since_evaluation) / len(losses_since_evaluation)
            losses_since_evaluation.clear()
            evaluations.append({'epoch': epoch, 'step': step, 'train_loss': train_loss, **metrics})
            logger.info(
                'epoch %d step %d | train loss %.4f | val precision@10%% recall %.4f, AUC-PR %.4f, AUC-ROC %.4f',
                epoch,
                step,
                train_loss,
                metrics['precision_at_10pct_recall'],
                metrics['auc_pr'],
                metrics['auc_roc'],
            )

            score = tuple(metrics[name] for name in SELECTION_METRICS)
            if score > best_score:
                best_score = score
                best_step = step
                best_metrics = metrics
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                evaluations_without_improvement = 0
                _write_checkpoint(
                    output_dir,
                    best_state,
                    _model_metadata(config, feature_spec, training_data, best_step, best_metrics, evaluations),
                )
            else:
                evaluations_without_improvement += 1
                if evaluations_without_improvement >= config.patience:
                    logger.info('Early stopping: no improvement for %d evaluations', config.patience)
                    stop = True
                    break
        if stop:
            break

    model.load_state_dict(best_state)
    model_path = _write_checkpoint(
        output_dir,
        best_state,
        _model_metadata(config, feature_spec, training_data, best_step, best_metrics, evaluations),
    )
    scores, _ = predict_scores(model, validation_loader, device)
    _write_validation_predictions(
        output_dir / 'validation_predictions.csv', dataset, split.validation_ids, config.year, scores
    )
    logger.info('Best checkpoint from step %d saved to %s', best_step, model_path)
    return TrainingResult(output_dir, model_path, feature_spec, split, best_metrics, evaluations)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Train the HeteroGCN2 company-risk model locally. Each company observed in --year is a target; the model '
            "predicts whether it has risk_label > 0 from its neighbours' --feature-year risk."
        ),
        epilog=(
            f'Provide your own nodes.csv, edges.csv, and risks.csv as documented in {SCHEMA_DOCUMENTATION}. '
            'data/dummy contains a small synthetic example. Defaults reproduce the shipped research checkpoint.'
        ),
    )
    data = parser.add_argument_group('data')
    data.add_argument('--data-dir', type=Path, required=True, help='Directory containing nodes/edges/risks CSVs')
    data.add_argument('--year', type=int, required=True, help='Year whose risk_label values are training targets')
    data.add_argument('--feature-year', type=int, help='Year of neighbour risk features (default: --year)')
    data.add_argument(
        '--val-fraction',
        type=float,
        default=0.2,
        help='Share of targets held out for validation (default: %(default)s)',
    )
    data.add_argument(
        '--max-neighbours',
        type=int,
        default=MAX_PROPAGATION_NEIGHBOURS,
        help='Maximum edges sampled per company, direction, and hop (default: %(default)s)',
    )
    data.add_argument(
        '--seed', type=int, default=42, help='Seed for the split, sampling, and initialisation (default: %(default)s)'
    )

    model = parser.add_argument_group('model and optimisation')
    model.add_argument(
        '--layers',
        type=int,
        default=3,
        help='Graph-convolution layers (also the neighbourhood depth) (default: %(default)s)',
    )
    model.add_argument(
        '--hidden-size', type=int, default=64, help='Hidden size of each graph-convolution layer (default: %(default)s)'
    )
    model.add_argument(
        '--loss',
        choices=sorted(LOSSES),
        default='pu',
        help='Positive-unlabelled or cross-entropy loss (default: %(default)s)',
    )
    model.add_argument(
        '--positive-prior',
        type=float,
        default=ESTIMATED_POSITIVE_PRIOR,
        help='Estimated share of truly risky companies (PU loss only) (default: %(default)s)',
    )
    model.add_argument('--learning-rate', type=float, default=0.001, help='Adam learning rate (default: %(default)s)')
    model.add_argument('--batch-size', type=int, default=64, help='Target companies per batch (default: %(default)s)')
    model.add_argument(
        '--epochs', type=int, default=2, help='Maximum passes over the training targets (default: %(default)s)'
    )
    model.add_argument(
        '--eval-every',
        type=int,
        default=1000,
        help='Validate every N steps, and always at the end of each epoch (default: %(default)s)',
    )
    model.add_argument(
        '--patience', type=int, default=4, help='Stop after N validations without improvement (default: %(default)s)'
    )
    model.add_argument(
        '--device', default='cpu', help='Torch device, for example cpu, cuda, or mps (default: %(default)s)'
    )

    output = parser.add_argument_group('output')
    output.add_argument(
        '--output-dir',
        type=Path,
        help=f'Run directory (default: {DEFAULT_OUTPUT_ROOT}/heterogcn2_<loss>_layers_<n>_size_<n>_<timestamp>)',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    try:
        config = TrainingConfig(
            data_dir=args.data_dir,
            year=args.year,
            feature_year=args.feature_year,
            output_dir=args.output_dir,
            loss=args.loss,
            nb_layers=args.layers,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            eval_every=args.eval_every,
            patience=args.patience,
            positive_prior=args.positive_prior,
            val_fraction=args.val_fraction,
            max_neighbours=args.max_neighbours,
            seed=args.seed,
            device=args.device,
        )
        result = train_model(config)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f'error: {error}\n')

    print(f'Best validation metrics ({len(result.split.validation_ids)} companies):')
    for name, value in result.best_metrics.items():
        print(f'  {name}: {value:.4f}')
    print(f'Model written to {result.model_path} (feature layout in {MODEL_METADATA_FILENAME})')
    print(
        f'Predict with: python scripts/predict.py --data-dir {config.data_dir} --model {result.model_path} '
        f'--year <year> --output outputs/predictions.csv'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
