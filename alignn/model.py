"""
ALIGNN Model Implementation
Based on reference_alignn/alignn/models/alignn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn import AvgPooling
import numpy as np
from typing import Union, Tuple

from .config import ALIGNNConfig
from .utils import RBFExpansion


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, BatchNorm1d, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),  # BatchNorm1d to match reference
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, BatchNorm1d, silu layer."""
        return self.layer(x)


class EdgeGatedGraphConv(nn.Module):
    """
    Edge-gated graph convolution
    Based on https://arxiv.org/abs/1711.07553
    """

    def __init__(self, input_features: int, output_features: int, residual: bool = True):
        super().__init__()
        self.residual = residual

        # Gate functions
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        # Update functions
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            g: DGL graph
            node_feats: Node features
            edge_feats: Edge features

        Returns:
            Updated node and edge features
        """
        g = g.local_var()

        # Compute edge gates
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        # Sigmoid gating
        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)

        # Message passing
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"),
            fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))

        # Normalize
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # Apply activation
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        # Residual connection
        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """ALIGNN convolution layer"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            g: Crystal graph
            lg: Line graph
            x: Node features
            y: Edge features
            z: Angle features

        Returns:
            Updated x, y, z
        """
        g = g.local_var()
        lg = lg.local_var()

        # Update nodes and edges on crystal graph
        x, m = self.node_update(g, x, y)

        # Update edges (as nodes in line graph)
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class ALIGNN(nn.Module):
    """
    Atomistic Line Graph Neural Network

    Main ALIGNN model for material property prediction
    """

    def __init__(self, config: ALIGNNConfig = None):
        super().__init__()

        if config is None:
            config = ALIGNNConfig(name="alignn")

        self.config = config
        self.classification = config.classification

        # Atom embedding
        self.atom_embedding = MLPLayer(
            config.atom_input_features,
            config.hidden_features
        )

        # Edge embedding (with RBF expansion)
        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features)
        )

        # Angle embedding
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features)
        )

        # ALIGNN layers
        self.alignn_layers = nn.ModuleList([
            ALIGNNConv(config.hidden_features, config.hidden_features)
            for _ in range(config.alignn_layers)
        ])

        # GCN layers
        self.gcn_layers = nn.ModuleList([
            EdgeGatedGraphConv(config.hidden_features, config.hidden_features)
            for _ in range(config.gcn_layers)
        ])

        # Readout
        self.readout = AvgPooling()
        self.readout_feat = AvgPooling()

        # Output layer
        if self.classification:
            self.fc = nn.Linear(config.hidden_features, config.num_classes)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)

        # Extra features support (lattice properties, etc.)
        if config.extra_features != 0:
            # Credit for extra_features work:
            # Gong et al., https://doi.org/10.48550/arXiv.2208.05039
            self.extra_feature_embedding = MLPLayer(
                config.extra_features, config.extra_features
            )
            self.fc3 = nn.Linear(
                config.hidden_features + config.extra_features,
                config.output_features,
            )
            self.fc1 = MLPLayer(
                config.extra_features + config.hidden_features,
                config.extra_features + config.hidden_features,
            )
            self.fc2 = MLPLayer(
                config.extra_features + config.hidden_features,
                config.extra_features + config.hidden_features,
            )

        # Link function
        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            # Initialize bias for log link
            avg_val = 0.7
            self.fc.bias.data = torch.tensor(np.log(avg_val), dtype=torch.float)
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor], dgl.DGLGraph]):
        """
        Forward pass

        Args:
            g: Input graph(s)
               Can be (graph, line_graph, lattice) or just graph

        Returns:
            Predictions
        """
        # Unpack input
        if isinstance(g, (list, tuple)) and len(g) == 3:
            g, lg, lat = g
        else:
            raise ValueError("Expected input as (graph, line_graph, lattice)")

        lg = lg.local_var()

        # Angle features (fixed)
        z = self.angle_embedding(lg.edata.pop("h"))

        # Extra features embedding
        if self.config.extra_features != 0:
            features = g.ndata["extra_features"]
            features = self.extra_feature_embedding(features)

        g = g.local_var()

        # Initial node features: atom feature network
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # Initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN layers: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # GCN layers: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # Norm-activation-pool-classify
        h = self.readout(g, x)

        # Handle extra features
        if self.config.extra_features != 0:
            h_feat = self.readout_feat(g, features)
            h = torch.cat((h, h_feat), 1)
            h = self.fc1(h)
            h = self.fc2(h)
            out = self.fc3(h)
        else:
            out = self.fc(h)

        # Apply link function
        if self.link:
            out = self.link(out)

        # Apply softmax for classification
        if self.classification:
            out = self.softmax(out)

        return torch.squeeze(out)
