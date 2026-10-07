"""Graph neural-network architecture and checkpoint loading."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import GCN2Conv, HeteroConv

from risk_propagation.data import EDGE_TYPE_IDS, NodeFeatureSpec

ARCHITECTURE = 'HeteroGCN2'
MODEL_METADATA_FILENAME = 'model.json'
GCN2_ALPHA = 0.3
NODE_TYPE = 'node'
# Edge-attribute sign marking a message along (fwd) or against (bwd) a relationship.
DIRECTIONS = {'fwd': 1.0, 'bwd': -1.0}


def edge_type_key(position: int, direction: str) -> tuple[str, str, str]:
    """Heterogeneous edge type for a relationship-type position in ``EDGE_TYPE_IDS``."""

    return (NODE_TYPE, f'type_{position}_{direction}', NODE_TYPE)


def first_node_indices(batch: torch.Tensor) -> torch.Tensor:
    """Return the index of each graph's first node, which is its target company."""

    return torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=batch.device),
            torch.where(batch[1:] != batch[:-1])[0] + 1,
        ]
    )


class WrappedGCN2Conv(nn.Module):
    """GCN2Conv with the ``(x, edge_index, ...)`` argument order HeteroConv passes."""

    def __init__(self, channels: int, alpha: float) -> None:
        super().__init__()
        self.conv = GCN2Conv(channels, alpha=alpha)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return self.conv(x, x0, edge_index)


class HeteroGCN2(nn.Module):
    """GCN2 with a separate graph convolution per relationship type and direction.

    Parameter names match the original implementation, so the shipped
    checkpoint loads directly. ``edge_attr`` holds +1
    (``fwd``) or -1 (``bwd``) at the relationship type's position in
    ``EDGE_TYPE_IDS``; see ``extract_neighbourhood``.

    Unlike the original implementation, every relationship-type convolution runs
    on every batch, with an empty edge set when the type is absent. Each
    convolution also adds self-loop and initial-residual terms to all nodes, so
    skipping absent types made a company's score depend on which other
    neighbourhoods shared its batch. The shipped checkpoint was trained and
    evaluated on large batches in which every type was present; always running
    all convolutions reproduces that setting for any batch size.
    """

    def __init__(
        self,
        node_feature_size: int,
        conv2_size: int,
        nb_layers: int,
        num_edge_types: int = len(EDGE_TYPE_IDS),
    ) -> None:
        super().__init__()
        self.nb_layers = nb_layers
        self.num_edge_types = num_edge_types
        self.first_layer = nn.Linear(node_feature_size, conv2_size)
        self.first_relu = nn.ReLU()
        for layer_index in range(nb_layers):
            if layer_index >= 1:
                setattr(self, f'activ_{layer_index}', nn.ReLU())
            convs = {
                edge_type_key(position, direction): WrappedGCN2Conv(conv2_size, alpha=GCN2_ALPHA)
                for position in range(num_edge_types)
                for direction in DIRECTIONS
            }
            setattr(self, f'conv_{layer_index}', HeteroConv(convs, aggr='sum'))
        self.last_relu = nn.ReLU()
        self.last_linear = nn.Linear(conv2_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.first_relu(self.first_layer(x))
        edge_index_dict = {
            edge_type_key(position, direction): edge_index[:, edge_attr[:, position] == sign]
            for position in range(self.num_edge_types)
            for direction, sign in DIRECTIONS.items()
        }
        x0_dict = {NODE_TYPE: x}
        x_dict = {NODE_TYPE: x}
        for layer_index in range(self.nb_layers):
            if layer_index >= 1:
                x_dict[NODE_TYPE] = getattr(self, f'activ_{layer_index}')(x_dict[NODE_TYPE])
            x_dict = getattr(self, f'conv_{layer_index}')(x_dict, edge_index_dict, x0_dict=x0_dict)
        x = self.last_linear(self.last_relu(x_dict[NODE_TYPE]))
        return x[first_node_indices(batch)]


def load_checkpoint(checkpoint_path: str | Path) -> tuple[HeteroGCN2, NodeFeatureSpec]:
    """Load a ``model.pth`` state dict and the feature layout from ``model.json`` beside it."""

    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.with_name(MODEL_METADATA_FILENAME)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Model checkpoint does not exist: {checkpoint_path}')
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Model metadata does not exist: {metadata_path}')

    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('architecture') != ARCHITECTURE:
        raise ValueError(f'{metadata_path} describes {metadata.get("architecture")!r}, not {ARCHITECTURE!r}')
    if tuple(metadata['edge_type_ids']) != EDGE_TYPE_IDS:
        raise ValueError(f'{metadata_path} uses edge types {metadata["edge_type_ids"]}, expected {EDGE_TYPE_IDS}')
    feature_spec = NodeFeatureSpec(tuple(metadata['node_features']['company_types']))

    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(state_dict, dict) or 'first_layer.weight' not in state_dict:
        raise ValueError(f'Unsupported checkpoint structure in {checkpoint_path}')
    hidden_size, node_feature_size = state_dict['first_layer.weight'].shape
    if node_feature_size != feature_spec.size:
        raise ValueError(
            f'Checkpoint {checkpoint_path} expects {node_feature_size} node features, '
            f'but {metadata_path} defines {feature_spec.size}'
        )
    nb_layers = len({key.split('.')[0] for key in state_dict if key.startswith('conv_')})
    model = HeteroGCN2(node_feature_size, hidden_size, nb_layers, num_edge_types=len(EDGE_TYPE_IDS))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, feature_spec
