"""
ALIGNN Graph Builder
Convert crystal structures to DGL graphs for ALIGNN
Based on reference_alignn/alignn/graphs.py
"""

import numpy as np
import torch
import dgl
from typing import Optional, Tuple, Dict, Any
from collections import defaultdict
import math


def compute_bond_cosines(edges):
    """
    Compute bond angle cosines from bond displacement vectors

    For line graph edge (a,b), (b,c) representing angle a-b-c:
    cos(theta) = (r_ba · r_bc) / (||r_ba|| ||r_bc||)

    Args:
        edges: DGL edges with 'r' attribute

    Returns:
        Dictionary with 'h' key containing cosine values
    """
    # Negate src bond: r_ba = -r_ab
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]

    # Compute cosine
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )

    # Clamp to [-1, 1] to handle numerical errors
    bond_cosine = torch.clamp(bond_cosine, -1, 1)

    return {"h": bond_cosine}


def temp_graph_jarvis(
    atoms,
    cutoff: float = 4.0,
    atom_features: str = "atomic_number",
    dtype: str = "float32"
):
    """
    Construct a graph for a given cutoff using jarvis lattice method.

    This is aligned with reference_alignn/alignn/graphs.py:temp_graph (lines 23-83)

    Args:
        atoms: Atoms object (jarvis.core.atoms.Atoms)
        cutoff: Cutoff radius in Angstroms
        atom_features: Atom feature type
        dtype: Data type string

    Returns:
        g: DGL graph
        u, v, r: Edge lists and displacement vectors
    """
    TORCH_DTYPES = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat": torch.bfloat16,
    }
    torch_dtype = TORCH_DTYPES.get(dtype, torch.float32)

    u, v, r, d, images, atom_feats = [], [], [], [], [], []
    elements = atoms.elements

    # Import jarvis function for node attributes
    try:
        from jarvis.core.specie import get_node_attributes
    except ImportError:
        raise ImportError("jarvis-tools is required for radius_graph_jarvis strategy")

    # Loop over each atom in the structure
    for ii, i in enumerate(atoms.cart_coords):
        # Get neighbors within the cutoff distance using jarvis lattice method
        neighs = atoms.lattice.get_points_in_sphere(
            atoms.frac_coords, i, cutoff, distance_vector=True
        )

        # Filter out self-loops (exclude cases where atom is bonded to itself)
        valid_indices = neighs[2] != ii

        u.extend([ii] * np.sum(valid_indices))
        d.extend(neighs[1][valid_indices])
        v.extend(neighs[2][valid_indices])
        images.extend(neighs[3][valid_indices])
        r.extend(neighs[4][valid_indices])

        feat = list(get_node_attributes(elements[ii], atom_features=atom_features))
        atom_feats.append(feat)

    # Create DGL graph
    g = dgl.graph((np.array(u), np.array(v)))
    atom_feats = np.array(atom_feats)

    # Add data to the graph with the specified dtype
    g.ndata["atom_features"] = torch.tensor(atom_feats, dtype=torch_dtype)
    g.ndata["Z"] = torch.tensor(atom_feats, dtype=torch.int64)
    g.edata["r"] = torch.tensor(np.array(r), dtype=torch_dtype)
    g.edata["d"] = torch.tensor(d, dtype=torch_dtype)
    g.edata["images"] = torch.tensor(images, dtype=torch_dtype)
    g.ndata["pos"] = torch.tensor(atoms.cart_coords, dtype=torch_dtype)
    g.ndata["frac_coords"] = torch.tensor(atoms.frac_coords, dtype=torch_dtype)

    return g, u, v, r


def radius_graph_jarvis(
    atoms,
    cutoff: float = 4.0,
    cutoff_extra: float = 0.5,
    atom_features: str = "atomic_number",
    line_graph: bool = True,
    dtype: str = "float32",
    max_attempts: int = 10,
):
    """
    Construct radius graph with jarvis tools.

    This is aligned with reference_alignn/alignn/graphs.py:radius_graph_jarvis (lines 85-126)

    Args:
        atoms: Atoms object (jarvis.core.atoms.Atoms)
        cutoff: Initial cutoff radius in Angstroms
        cutoff_extra: Extra cutoff increment if graph is incomplete
        atom_features: Atom feature type
        line_graph: Whether to compute line graph
        dtype: Data type string
        max_attempts: Maximum attempts to construct graph

    Returns:
        g: DGL graph (and lg if line_graph=True)
    """
    count = 0
    while count <= max_attempts:
        count += 1
        g, u, v, r = temp_graph_jarvis(
            atoms=atoms,
            cutoff=cutoff,
            atom_features=atom_features,
            dtype=dtype,
        )
        # Check if all atoms are included as nodes
        if g.num_nodes() == len(atoms.elements):
            break
        # Increment the cutoff if the graph is incomplete
        cutoff += cutoff_extra

    if count >= max_attempts:
        raise ValueError(f"Failed to construct graph after {max_attempts} attempts for structure")

    # Optional: Create a line graph if requested
    if line_graph:
        lg = g.line_graph(shared=True)
        lg.apply_edges(compute_bond_cosines)
        return g, lg

    return g


def canonize_edge(src_id: int, dst_id: int, src_image: tuple, dst_image: tuple) -> Tuple:
    """
    Compute canonical edge representation

    Sort vertex IDs and shift periodic images so first vertex is in (0,0,0)

    Args:
        src_id: Source atom ID
        dst_id: Destination atom ID
        src_image: Source periodic image
        dst_image: Destination periodic image

    Returns:
        Canonical (src_id, dst_id, src_image, dst_image)
    """
    # Store directed edges with src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # Shift so src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0), "Source must be in (0,0,0) after canonization"

    return src_id, dst_id, src_image, dst_image


def radius_graph_edges(
    atoms,
    cutoff: float = 8.0,
    bond_tol: float = 0.5,
    id: Optional[str] = None,
    atol: float = 1e-5,
    cutoff_extra: float = 0.5,
    use_canonize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct radius graph with reciprocal space supercell search.

    This is the reference implementation from alignn/graphs.py:267-355
    that properly handles periodic boundaries using reciprocal lattice vectors.

    Args:
        atoms: Atoms object (jarvis.core.atoms.Atoms)
        cutoff: Cutoff radius in Angstroms
        bond_tol: Tolerance for bond cutoff
        id: Structure ID for logging
        atol: Absolute tolerance for distance comparison
        cutoff_extra: Extra cutoff increment if graph is disconnected
        use_canonize: Whether to canonize edges

    Returns:
        u: Source node indices
        v: Destination node indices
        r: Bond displacement vectors
        cell_images: Periodic image vectors
    """
    def temp_graph(cutoff_val):
        """Inner function to construct graph for given cutoff"""
        # Convert to torch tensors
        cart_coords = torch.tensor(atoms.cart_coords).type(torch.get_default_dtype())
        frac_coords = torch.tensor(atoms.frac_coords).type(torch.get_default_dtype())
        lattice_mat = torch.tensor(atoms.lattice_mat).type(torch.get_default_dtype())

        X_src = cart_coords
        num_atoms = X_src.shape[0]

        # Determine supercell range using reciprocal lattice
        # Key: Use 2π * inv(lattice)^T to get reciprocal vectors
        recp = 2 * math.pi * torch.linalg.inv(lattice_mat).T
        recp_len = torch.sqrt(torch.sum(recp**2, dim=1))

        # Calculate required supercell range
        maxr = torch.ceil((cutoff_val + bond_tol) * recp_len / (2 * math.pi))
        nmin = torch.floor(torch.min(frac_coords, dim=0)[0]) - maxr
        nmax = torch.ceil(torch.max(frac_coords, dim=0)[0]) + maxr

        # Construct supercell index list using cartesian product
        all_ranges = [
            torch.arange(x, y, dtype=torch.get_default_dtype())
            for x, y in zip(nmin, nmax)
        ]
        cell_images = torch.cartesian_prod(*all_ranges)

        # Tile periodic images into X_dst
        # X_dst[i*num_atoms + j] is atom j in cell_image i
        X_dst = (cell_images @ lattice_mat)[:, None, :] + X_src
        X_dst = X_dst.reshape(-1, 3)

        # Compute pairwise distances between (0,0,0) cell and all periodic images
        dist = torch.cdist(X_src, X_dst, compute_mode="donot_use_mm_for_euclid_dist")

        # Create neighbor mask: within cutoff and not self-loop
        neighbor_mask = torch.bitwise_and(
            dist <= cutoff_val,
            ~torch.isclose(
                dist,
                torch.tensor([0]).type(torch.get_default_dtype()),
                atol=atol,
            ),
        )

        # Extract edge list
        u, v = torch.where(neighbor_mask)

        # Map back to cell images
        cell_images_edges = cell_images[v // num_atoms]

        # Compute bond displacement vectors
        r = (X_dst[v] - X_src[u]).float()

        # Map v back to atom indices (v % num_atoms)
        v = v % num_atoms

        # Create DGL graph to check connectivity
        g = dgl.graph((u, v))

        return g, u, v, r, cell_images_edges

    # Adaptive cutoff: ensure all atoms are connected
    current_cutoff = cutoff
    while True:
        g, u, v, r, cell_images_edges = temp_graph(current_cutoff)

        # Check if graph is fully connected (all atoms have edges)
        if g.num_nodes() == len(atoms.elements):
            return u, v, r, cell_images_edges
        else:
            # Increase cutoff and retry
            current_cutoff += cutoff_extra


def nearest_neighbor_edges(
    atoms,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    id: Optional[str] = None,
    use_canonize: bool = False
) -> Tuple[Dict, np.ndarray]:
    """
    Construct k-nearest neighbor edge list

    Note: This is the simplified k-NN approach. For strict periodic boundary
    handling, use radius_graph_edges() instead.

    Args:
        atoms: Atoms object with get_all_neighbors method
        cutoff: Initial cutoff radius
        max_neighbors: Number of nearest neighbors
        id: Structure ID for logging
        use_canonize: Whether to canonize edges

    Returns:
        edges: Dictionary mapping (src_id, dst_id) to set of dst_images
        images: Array of periodic images
    """
    # Get all neighbors within cutoff
    all_neighbors = atoms.get_all_neighbors(r=cutoff)

    # Check minimum number of neighbors
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    # Recursively increase cutoff if needed
    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff

        return nearest_neighbor_edges(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id
        )

    # Build edge dictionary
    edges = defaultdict(set)

    for site_idx, neighborlist in enumerate(all_neighbors):
        # Sort by distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # Distance to k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]

        # Keep all edges out to k-th neighbor shell
        mask = distances <= max_dist
        ids = ids[mask]
        images = images[mask]
        distances = distances[mask]

        # Track cell-resolved edges
        for dst, image in zip(ids, images):
            if use_canonize:
                src_id, dst_id, src_image, dst_image = canonize_edge(
                    site_idx, dst, (0, 0, 0), tuple(image)
                )
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    return edges, images


def build_undirected_edgedata(atoms, edges: Dict) -> Tuple:
    """
    Build undirected graph data from edge set

    Args:
        atoms: Atoms object
        edges: Dictionary mapping (src_id, dst_id) to set of dst_images

    Returns:
        u: Source node indices
        v: Destination node indices
        r: Bond displacement vectors
        all_images: Periodic images
    """
    u, v, r, all_images = [], [], [], []

    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            # Fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image

            # Cartesian displacement vector: src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )

            # Add edges for both directions
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
                all_images.append(dst_image)

    # Convert to tensors
    u = torch.tensor(np.array(u))
    v = torch.tensor(np.array(v))
    r = torch.tensor(np.array(r)).type(torch.get_default_dtype())
    all_images = torch.tensor(np.array(all_images)).type(torch.get_default_dtype())

    return u, v, r, all_images


def get_node_attributes(element: str, atom_features: str = "cgcnn") -> list:
    """
    Get node attributes for an element

    Args:
        element: Element symbol
        atom_features: Feature type

    Returns:
        Feature vector
    """
    try:
        from jarvis.core.specie import get_node_attributes as jarvis_get_node_attributes
        return jarvis_get_node_attributes(element, atom_features=atom_features)
    except ImportError:
        # Fallback: just use atomic number if jarvis not available
        from ase.data import atomic_numbers
        if atom_features == "atomic_number":
            return [atomic_numbers[element]]
        elif atom_features == "cgcnn":
            # Return 92-dimensional one-hot encoding
            z = atomic_numbers[element]
            feat = [0.0] * 92
            if z <= 92:
                feat[z - 1] = 1.0
            return feat
        else:
            raise NotImplementedError(f"Feature type {atom_features} not implemented without jarvis")


class ALIGNNGraphBuilder:
    """
    ALIGNN Graph Builder

    Convert crystal structures to DGL graphs for ALIGNN
    """

    def __init__(
        self,
        cutoff: float = 8.0,
        max_neighbors: int = 12,
        atom_features: str = "cgcnn",
        compute_line_graph: bool = True,
        use_canonize: bool = True,
        neighbor_strategy: str = "k-nearest",
        cutoff_extra: float = 0.5,
        dtype: str = "float32"
    ):
        """
        Initialize graph builder

        Args:
            cutoff: Cutoff radius (Angstrom)
            max_neighbors: Maximum number of neighbors (k-nearest only)
            atom_features: Atom feature type
            compute_line_graph: DEPRECATED - Line graphs are always generated by Dataset
            use_canonize: Whether to canonize edges (default True for robustness)
            neighbor_strategy: "k-nearest" (simplified), "radius" (reference), or "radius_graph_jarvis" (jarvis native)
            cutoff_extra: Extra cutoff increment for adaptive adjustment
            dtype: Data type

        Note:
            - atoms_to_graph() always returns only g, never (g, lg)
            - Line graph generation is handled uniformly by StructureDataset
            - This matches reference implementation behavior (graphs.py:959-997)
            - use_canonize=True ensures undirected edges and robustness (reference default)
        """
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.atom_features = atom_features
        self.compute_line_graph = compute_line_graph  # Stored but ignored
        self.use_canonize = use_canonize
        self.neighbor_strategy = neighbor_strategy
        self.cutoff_extra = cutoff_extra

        # Set dtype (locally, NOT globally)
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat": torch.bfloat16
        }
        self.dtype = dtype_map.get(dtype, torch.float32)
        # NOTE: Do NOT call torch.set_default_dtype() here - it causes global side effects

    def atoms_to_graph(self, atoms, id: Optional[str] = None):
        """
        Convert Atoms object to DGL graph

        Note: This method ALWAYS returns only the graph g, never (g, lg).
        Line graphs should be generated by the Dataset class for consistency.

        Args:
            atoms: Atoms object (from jarvis or similar)
            id: Structure ID for logging

        Returns:
            g: DGL graph (never returns line graph tuple)
        """
        if self.neighbor_strategy == "k-nearest":
            return self._build_knn_graph(atoms, id)
        elif self.neighbor_strategy == "radius_graph":
            return self._build_radius_graph(atoms, id)
        elif self.neighbor_strategy == "voronoi":
            return self._build_voronoi_graph(atoms, id)
        elif self.neighbor_strategy == "radius_graph_jarvis":
            return self._build_jarvis_graph(atoms, id)
        else:
            raise NotImplementedError(
                f"Neighbor strategy {self.neighbor_strategy} not implemented. "
                f"Choose from: 'k-nearest', 'radius_graph', 'voronoi', 'radius_graph_jarvis'"
            )

    def _build_knn_graph(self, atoms, id: Optional[str] = None):
        """
        Build k-nearest neighbor graph

        Returns only the graph g. Line graph generation is handled by Dataset.
        """

        # Get edges
        edges, _ = nearest_neighbor_edges(
            atoms=atoms,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            id=id,
            use_canonize=self.use_canonize
        )

        # Build undirected edge data
        u, v, r, images = build_undirected_edgedata(atoms, edges)

        # Store atomic numbers (not vectorized features)
        # Dataset will convert these to features using attribute lookup
        atomic_numbers = []
        for element in atoms.elements:
            from jarvis.core.specie import chem_data
            atomic_numbers.append(chem_data[element]["Z"])

        atomic_numbers = np.array(atomic_numbers)
        node_features = torch.tensor(atomic_numbers).type(torch.get_default_dtype())

        # Create graph
        g = dgl.graph((u, v))
        g.ndata["atom_features"] = node_features  # Store atomic numbers, not vectors
        g.edata["r"] = r
        g.edata["images"] = images

        # Add additional node data
        g.ndata["V"] = torch.tensor([atoms.volume for _ in range(atoms.num_atoms)])
        g.ndata["frac_coords"] = torch.tensor(atoms.frac_coords).type(
            torch.get_default_dtype()
        )

        # Note: compute_line_graph parameter is IGNORED
        # Always return only g, never (g, lg)
        return g

    def _build_radius_graph(self, atoms, id: Optional[str] = None):
        """
        Build radius graph using reciprocal space supercell search.

        This is the reference implementation that properly handles periodic boundaries.
        Returns only the graph g. Line graph generation is handled by Dataset.
        """
        # Get edges using radius graph method
        u, v, r, images = radius_graph_edges(
            atoms=atoms,
            cutoff=self.cutoff,
            bond_tol=0.5,
            id=id,
            atol=1e-5,
            cutoff_extra=self.cutoff_extra,
            use_canonize=self.use_canonize
        )

        # Store atomic numbers (not vectorized features)
        # Dataset will convert these to features using attribute lookup
        atomic_numbers = []
        for element in atoms.elements:
            from jarvis.core.specie import chem_data
            atomic_numbers.append(chem_data[element]["Z"])

        atomic_numbers = np.array(atomic_numbers)
        node_features = torch.tensor(atomic_numbers).type(torch.get_default_dtype())

        # Create graph
        g = dgl.graph((u, v))
        g.ndata["atom_features"] = node_features  # Store atomic numbers, not vectors
        g.edata["r"] = r
        g.edata["images"] = images

        # Add additional node data
        g.ndata["V"] = torch.tensor([atoms.volume for _ in range(atoms.num_atoms)])
        g.ndata["frac_coords"] = torch.tensor(atoms.frac_coords).type(
            torch.get_default_dtype()
        )

        # Note: compute_line_graph parameter is IGNORED
        # Always return only g, never (g, lg)
        return g

    def _build_voronoi_graph(self, atoms, id: Optional[str] = None):
        """
        Build graph using Voronoi tessellation for neighbor finding.

        Voronoi-based neighbor finding identifies neighbors based on
        Voronoi cell face-sharing, which is chemically more meaningful
        than pure distance-based cutoffs.

        Returns only the graph g. Line graph generation is handled by Dataset.
        """
        try:
            from jarvis.core.specie import chem_data
        except ImportError:
            raise ImportError("jarvis-tools is required for voronoi strategy")

        # Get Voronoi neighbors from jarvis
        try:
            # jarvis Atoms has voronoi_neighbors method
            voronoi_data = atoms.voronoi_neighbors
        except Exception as e:
            raise ValueError(f"Failed to compute Voronoi neighbors: {e}")

        # Build edge lists from Voronoi data
        u, v, r_list, images_list = [], [], [], []

        for site_idx, neighbors in enumerate(voronoi_data):
            for neighbor in neighbors:
                # neighbor format: (neighbor_idx, distance, image)
                dst_idx = neighbor[0] if isinstance(neighbor, (list, tuple)) else neighbor.get('site_index', 0)
                dist = neighbor[1] if isinstance(neighbor, (list, tuple)) else neighbor.get('distance', 0)
                image = neighbor[2] if isinstance(neighbor, (list, tuple)) and len(neighbor) > 2 else (0, 0, 0)

                # Compute displacement vector
                dst_frac = atoms.frac_coords[dst_idx] + np.array(image)
                src_frac = atoms.frac_coords[site_idx]
                d = atoms.lattice.cart_coords(dst_frac - src_frac)

                # Add both directions
                u.append(site_idx)
                v.append(dst_idx)
                r_list.append(d)
                images_list.append(image)

                u.append(dst_idx)
                v.append(site_idx)
                r_list.append(-d)
                images_list.append(tuple(-np.array(image)))

        # Store atomic numbers (not vectorized features)
        atomic_numbers = []
        for element in atoms.elements:
            atomic_numbers.append(chem_data[element]["Z"])

        atomic_numbers = np.array(atomic_numbers)
        node_features = torch.tensor(atomic_numbers).type(self.dtype)

        # Create graph
        g = dgl.graph((u, v))
        g.ndata["atom_features"] = node_features
        g.edata["r"] = torch.tensor(np.array(r_list)).type(self.dtype)
        g.edata["images"] = torch.tensor(np.array(images_list)).type(self.dtype)

        # Add additional node data
        g.ndata["V"] = torch.tensor([atoms.volume for _ in range(atoms.num_atoms)])
        g.ndata["frac_coords"] = torch.tensor(atoms.frac_coords).type(self.dtype)

        return g

    def _build_jarvis_graph(self, atoms, id: Optional[str] = None):
        """
        Build graph using jarvis native lattice method.

        This is aligned with reference_alignn/alignn/graphs.py:radius_graph_jarvis
        Returns only the graph g. Line graph generation is handled by Dataset.

        Note: This method uses jarvis's get_points_in_sphere which has different
        behavior than the radius_graph method. It's useful for compatibility with
        official ALIGNN pretrained models.
        """
        # Determine dtype string from self.dtype
        dtype_str = "float32"
        if self.dtype == torch.float16:
            dtype_str = "float16"
        elif self.dtype == torch.float64:
            dtype_str = "float64"
        elif self.dtype == torch.bfloat16:
            dtype_str = "bfloat"

        # Use radius_graph_jarvis but only get the graph (not line graph)
        # since line graph is handled by Dataset
        g = radius_graph_jarvis(
            atoms=atoms,
            cutoff=self.cutoff,
            cutoff_extra=self.cutoff_extra,
            atom_features=self.atom_features,
            line_graph=False,  # Don't create line graph here
            dtype=dtype_str,
            max_attempts=10
        )

        # Add volume data (consistent with other methods)
        g.ndata["V"] = torch.tensor([atoms.volume for _ in range(atoms.num_atoms)])

        return g

    def batch_to_graphs(self, atoms_list: list, ids: Optional[list] = None) -> list:
        """
        Convert list of Atoms to graphs

        Args:
            atoms_list: List of Atoms objects
            ids: Optional list of IDs

        Returns:
            List of graphs (or graph tuples if line graph enabled)
        """
        if ids is None:
            ids = [None] * len(atoms_list)

        graphs = []
        for atoms, id in zip(atoms_list, ids):
            graph = self.atoms_to_graph(atoms, id)
            graphs.append(graph)

        return graphs
