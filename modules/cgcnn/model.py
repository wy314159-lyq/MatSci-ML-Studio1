"""
CGCNN Model Implementation
Based on: Xie & Grossman, Physical Review Letters 120, 145301 (2018)
Official implementation: https://github.com/txie-93/cgcnn

Core neural network architecture for crystal property prediction.
"""

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional operation on crystal graphs.

    This layer implements the graph convolution operation that aggregates
    information from neighboring atoms in the crystal structure.
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------
        atom_fea_len : int
            Number of atom hidden features
        nbr_fea_len : int
            Number of bond/edge features
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len

        # Linear layer for concatenated features
        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len,
            2 * self.atom_fea_len
        )

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.softplus2 = nn.Softplus()

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        Parameters
        ----------
        atom_in_fea : torch.Tensor (N, atom_fea_len)
            Atom features before convolution
        nbr_fea : torch.Tensor (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx : torch.LongTensor (N, M)
            Indices of M neighbors of each atom

        Returns
        -------
        atom_out_fea : torch.Tensor (N, atom_fea_len)
            Atom features after convolution
        """
        N, M = nbr_fea_idx.shape

        # Gather neighbor features
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]

        # Concatenate: [center_atom, neighbor_atom, edge_feature]
        total_nbr_fea = torch.cat([
            atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
            atom_nbr_fea,
            nbr_fea
        ], dim=2)

        # Apply linear transformation
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)

        # Split into filter and core
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        # Aggregate neighbor information
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)

        # Residual connection
        out = self.softplus2(atom_in_fea + nbr_sumed)

        return out


class CrystalGraphConvNet(nn.Module):
    """
    Crystal Graph Convolutional Neural Network.

    Main model for predicting material properties from crystal structures.
    """

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        classification=False,
        n_classes=2,
        strict_compatibility=False
    ):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------
        orig_atom_fea_len : int
            Number of atom features in the input
        nbr_fea_len : int
            Number of bond features
        atom_fea_len : int, optional
            Number of hidden atom features in conv layers (default: 64)
        n_conv : int, optional
            Number of convolutional layers (default: 3)
        h_fea_len : int, optional
            Number of hidden features after pooling (default: 128)
        n_h : int, optional
            Number of hidden layers after pooling (default: 1)
        classification : bool, optional
            If True, model performs classification (default: False)
        n_classes : int, optional
            Number of classes for classification (default: 2)
        strict_compatibility : bool, optional
            If True, force architecture to match official implementation exactly
            (classification always uses 2 classes, enables official weight loading)
            (default: False)
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification

        # Strict compatibility mode: force n_classes=2 for classification
        if strict_compatibility and classification:
            if n_classes != 2:
                print(f"[WARNING] strict_compatibility=True forces n_classes=2 "
                      f"(you specified {n_classes})")
            self.n_classes = 2
        else:
            self.n_classes = n_classes

        # Embedding layer
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # Convolutional layers
        self.convs = nn.ModuleList([
            ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)
        ])

        # Pooling to fully connected
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        # Hidden layers after pooling
        if n_h > 1:
            self.fcs = nn.ModuleList([
                nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)
            ])
            self.softpluses = nn.ModuleList([
                nn.Softplus() for _ in range(n_h - 1)
            ])

        # Output layer
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, self.n_classes)  # Use self.n_classes
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        Parameters
        ----------
        atom_fea : torch.Tensor (N, orig_atom_fea_len)
            Atom features from atom type
        nbr_fea : torch.Tensor (N, M, nbr_fea_len)
            Bond features of each atom's M neighbors
        nbr_fea_idx : torch.LongTensor (N, M)
            Indices of M neighbors of each atom
        crystal_atom_idx : list of torch.LongTensor
            Mapping from crystal idx to atom idx

        Returns
        -------
        prediction : torch.Tensor (N0,)
            Predicted property for each crystal
        """
        # Embed atom features
        atom_fea = self.embedding(atom_fea)

        # Apply convolutional layers
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)

        # Pool to crystal-level features
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        # Apply pooling transformation
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        # Apply hidden layers
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        # Output layer
        out = self.fc_out(crys_fea)

        if self.classification:
            out = self.logsoftmax(out)

        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pool atom features to crystal features.

        Uses mean pooling to aggregate atom-level features.

        Parameters
        ----------
        atom_fea : torch.Tensor (N, atom_fea_len)
            Atom feature vectors
        crystal_atom_idx : list of torch.LongTensor
            Mapping from crystal idx to atom idx

        Returns
        -------
        crys_fea : torch.Tensor (N0, atom_fea_len)
            Crystal feature vectors
        """
        total_atoms_in_batch = sum([len(idx_map) for idx_map in crystal_atom_idx])
        expected_atoms = atom_fea.data.shape[0]
        assert total_atoms_in_batch == expected_atoms, \
            f"Batch data mismatch! Total atoms in crystal_atom_idx: {total_atoms_in_batch}, " \
            f"but atom_fea has {expected_atoms} atoms. " \
            f"This usually happens with very large batch sizes. Try reducing batch size to 16-32."

        summed_fea = [
            torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
            for idx_map in crystal_atom_idx
        ]

        return torch.cat(summed_fea, dim=0)
