#! python3
# -*- encoding: utf-8 -*-
# ===================================
# File    :   mcts.py
# Author  :   Zehong Zhang
# Contact :   zhang@fmp-berlin.de
# ===================================
"""
Monte Carlo Tree Search (MCTS) engine for protein complex assembly.

The assembly proceeds by iteratively superposing pre-computed dimer structures
onto a growing partial complex. At each step, candidate chains are evaluated
for steric clashes and scored by a composite metric combining interface geometry
(pLDDT-weighted contact count) with XL-MS crosslink satisfaction.

Public API
----------
main(args)
    Main entry point called by ``complex_assembly_main.py`` or the notebook.
"""

import argparse
import copy
import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio.SVDSuperimposer import SVDSuperimposer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

def setup_logger(log_file: str) -> logging.Logger:
    """
    Create and configure a logger that writes INFO-level messages to both
    a file and the console.

    Duplicate handlers are suppressed so the function is safe to call
    multiple times (e.g. when re-running notebook cells).

    Parameters
    ----------
    log_file : str
        Destination log file path.  Parent directories are created if
        they do not already exist.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers on repeated calls
    if root_logger.handlers:
        return root_logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger


# ─────────────────────────────────────────────────────────────
# PDB I/O
# ─────────────────────────────────────────────────────────────

def parse_atm_record(line: str) -> dict:
    """
    Parse a single ATOM record from a PDB file.

    Parameters
    ----------
    line : str
        One ATOM line from a PDB file (must be ≥ 66 characters).

    Returns
    -------
    dict
        Keys: ``name``, ``atm_no``, ``atm_name``, ``atm_alt``,
        ``res_name``, ``chain``, ``res_no``, ``insert``, ``resid``,
        ``x``, ``y``, ``z``, ``occ``, ``B``.
    """
    record = defaultdict()
    record["name"]     = line[0:6].strip()
    record["atm_no"]   = int(line[6:11])
    record["atm_name"] = line[12:16].strip()
    record["atm_alt"]  = line[17]
    record["res_name"] = line[17:20].strip()
    record["chain"]    = line[21]
    record["res_no"]   = int(line[22:26])
    record["insert"]   = line[26].strip()
    record["resid"]    = line[22:29]
    record["x"]        = float(line[30:38])
    record["y"]        = float(line[38:46])
    record["z"]        = float(line[46:54])
    record["occ"]      = float(line[54:60])
    record["B"]        = float(line[60:66])
    return record


def read_pdb(pdbfile: str) -> tuple:
    """
    Read a PDB file and organise ATOM records by chain.

    For each chain the function collects:

    - Raw ATOM lines (for later PDB writing).
    - Atom coordinates as a flat list of ``[x, y, z]`` triplets.
    - **0-based** indices into that coordinate list for Cα atoms.
    - **0-based** indices for Cβ atoms (or Cα for glycine residues).

    Parameters
    ----------
    pdbfile : str
        Path to a PDB file.

    Returns
    -------
    pdb_chains : dict[str, list[str]]
        ``{chain_id: [atom_line, ...]}``.
    chain_coords : dict[str, list[list[float]]]
        ``{chain_id: [[x, y, z], ...]}``.
    chain_CA_inds : dict[str, list[int]]
        ``{chain_id: [ca_index, ...]}``.
    chain_CB_inds : dict[str, list[int]]
        ``{chain_id: [cb_index, ...]}``.
    """
    pdb_chains: dict    = {}
    chain_coords: dict  = {}
    chain_CA_inds: dict = {}
    chain_CB_inds: dict = {}
    coord_ind: int      = 0

    with open(pdbfile) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue

            record = parse_atm_record(line)
            cid    = record["chain"]

            if cid in pdb_chains:
                pdb_chains[cid].append(line)
                chain_coords[cid].append([record["x"], record["y"], record["z"]])
                coord_ind += 1
            else:
                pdb_chains[cid]    = [line]
                chain_coords[cid]  = [[record["x"], record["y"], record["z"]]]
                chain_CA_inds[cid] = []
                chain_CB_inds[cid] = []
                coord_ind = 0

            if record["atm_name"] == "CA":
                chain_CA_inds[cid].append(coord_ind)
            if record["atm_name"] == "CB" or (
                record["atm_name"] == "CA" and record["res_name"] == "GLY"
            ):
                chain_CB_inds[cid].append(coord_ind)

    return pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds


# ─────────────────────────────────────────────────────────────
# Clash detection
# ─────────────────────────────────────────────────────────────

def check_overlaps(
    path_coords: dict,
    path_CA_inds: dict,
    new_coords: np.ndarray,
    new_CA_inds: np.ndarray,
    clash_threshold_A: float = 5.0,
    clash_fraction: float = 0.5,
) -> bool:
    """
    Detect severe steric clashes between a new chain and the existing assembly.

    The test compares Cα atoms only.  If more than ``clash_fraction`` of the
    Cα atoms of the shorter chain lie within ``clash_threshold_A`` Å of any
    Cα in the other chain, the pair is considered clashing.

    Parameters
    ----------
    path_coords : dict[str, np.ndarray]
        Coordinates of chains already in the assembly.
    path_CA_inds : dict[str, np.ndarray]
        Cα indices for each chain in the assembly.
    new_coords : np.ndarray
        Coordinate array for the candidate new chain.
    new_CA_inds : np.ndarray
        Cα indices within ``new_coords``.
    clash_threshold_A : float, optional
        Distance cut-off in Å below which two atoms are considered clashing
        (default 5.0 Å).
    clash_fraction : float, optional
        Fraction of Cα atoms that must clash to trigger rejection
        (default 0.5, i.e. 50 %).

    Returns
    -------
    bool
        ``True`` if a severe clash is detected; ``False`` otherwise.
    """
    new_CAs = new_coords[new_CA_inds]
    n_new   = len(new_CAs)

    for chain, coords in path_coords.items():
        p_CAs = coords[path_CA_inds[chain]]

        combined = np.vstack([new_CAs, p_CAs])
        diff     = combined[:, None, :] - combined[None, :, :]
        dists    = np.sqrt((diff ** 2).sum(axis=-1))

        close = dists[:n_new, n_new:] <= clash_threshold_A
        if close.sum() > clash_fraction * min(n_new, len(p_CAs)):
            return True

    return False


# ─────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────

def cal_crosslink_distance(
    path_coords: dict,
    path_CA_inds: dict,
    a_chain: str,
    a_res: int,
    b_chain: str,
    b_res: int,
    crosslinker_length: int = 35,
) -> tuple:
    """
    Compute the Cα–Cα distance between two residues and evaluate whether
    the corresponding crosslink is satisfied.

    If either chain is absent from the current assembly the distance is
    returned as ``np.inf`` and the crosslink is marked unsatisfied.

    Parameters
    ----------
    path_coords : dict[str, np.ndarray]
        Atomic coordinates indexed by chain ID.
    path_CA_inds : dict[str, list[int]]
        Cα atom indices indexed by chain ID.
    a_chain : str
        Chain ID of the first crosslinked residue.
    a_res : int
        1-based sequence position of the first residue.
    b_chain : str
        Chain ID of the second crosslinked residue.
    b_res : int
        1-based sequence position of the second residue.
    crosslinker_length : int, optional
        Maximum allowed Cα–Cα distance (Å) for a satisfied crosslink
        (default 35 Å).

    Returns
    -------
    distance : float
        Cα–Cα distance rounded to two decimal places, or ``np.inf`` if
        a chain is missing.
    satisfied : bool
        ``True`` if ``distance <= crosslinker_length``.
    """
    if a_chain not in path_coords or b_chain not in path_coords:
        return np.inf, False

    a_idx   = path_CA_inds[a_chain][a_res - 1]
    b_idx   = path_CA_inds[b_chain][b_res - 1]
    a_coord = path_coords[a_chain][a_idx]
    b_coord = path_coords[b_chain][b_idx]

    dist = np.linalg.norm(a_coord - b_coord)
    return round(float(dist), 2), dist <= crosslinker_length


def score_crosslinks(
    ucrosslinks: pd.DataFrame,
    path_coords: dict,
    path_CA_inds: dict,
    crosslinker_length: int = 35,
    inter_prop: float = 1.0,
) -> tuple:
    """
    Evaluate how well the current structural model satisfies XL-MS restraints.

    Scores are computed for:

    - **All** crosslinks (intra- and inter-chain).
    - **Inter-chain** crosslinks only.
    - A weighted combination of the two.

    Parameters
    ----------
    ucrosslinks : pd.DataFrame
        Crosslink table with columns ``ChainA``, ``ResidueA``,
        ``ChainB``, ``ResidueB``.
    path_coords : dict[str, np.ndarray]
        Atomic coordinates indexed by chain ID.
    path_CA_inds : dict[str, list[int]]
        Cα atom indices indexed by chain ID.
    crosslinker_length : int, optional
        Maximum allowed Cα–Cα distance for a satisfied crosslink (Å),
        default 35.
    inter_prop : float, optional
        Weight given to the inter-chain score in the final combination
        (0 = total only, 1 = inter-chain only; default 1.0).

    Returns
    -------
    score_total : float
        Fraction of *all* crosslinks satisfied.
    score_inter : float
        Fraction of *inter-chain* crosslinks satisfied.
    score_final : float
        ``(1 - inter_prop) * score_total + inter_prop * score_inter``.
    """
    if len(ucrosslinks) == 0:
        return 1.0, 1.0, 1.0

    n_satisfied       = 0
    n_inter_satisfied = 0
    n_inter_total     = 0

    for _, row in ucrosslinks.iterrows():
        _, satisfied = cal_crosslink_distance(
            path_coords, path_CA_inds,
            row["ChainA"], int(row["ResidueA"]),
            row["ChainB"], int(row["ResidueB"]),
            crosslinker_length,
        )
        n_satisfied += satisfied
        if row["ChainA"] != row["ChainB"]:
            n_inter_total     += 1
            n_inter_satisfied += satisfied

    score_total = n_satisfied / len(ucrosslinks)
    score_inter = 1.0 if n_inter_total == 0 else n_inter_satisfied / n_inter_total
    score_final = (1.0 - inter_prop) * score_total + inter_prop * score_inter

    return score_total, score_inter, score_final


def score_structure(
    path_coords: dict,
    path_CB_inds: dict,
    path_plddt: dict,
    contact_threshold_A: float = 8.0,
) -> float:
    """
    Compute a structure-based quality score for the current assembly.

    For every pair of chains the function counts Cβ–Cβ contacts within
    ``contact_threshold_A`` Å and weights each contact by the mean pLDDT
    of the two contacting residues.  The per-pair contribution is
    ``log10(n_contacts + 1) × mean_pLDDT`` to dampen the effect of very
    large interfaces.

    Parameters
    ----------
    path_coords : dict[str, np.ndarray]
        Atomic coordinates indexed by chain ID.
    path_CB_inds : dict[str, np.ndarray]
        Cβ atom indices (Cα for glycine) indexed by chain ID.
    path_plddt : dict[str, np.ndarray]
        Per-residue pLDDT confidence scores indexed by chain ID.
    contact_threshold_A : float, optional
        Cβ–Cβ distance cut-off in Å for defining an interface contact
        (default 8.0 Å).

    Returns
    -------
    float
        Cumulative structure score across all inter-chain interfaces.
        Returns 0.0 if no inter-chain contacts are found.
    """
    score  = 0.0
    chains = list(path_coords.keys())

    for i, c1 in enumerate(chains):
        coords1 = path_coords[c1][path_CB_inds[c1]]
        plddt1  = path_plddt[c1]

        for c2 in chains[i + 1:]:
            coords2 = path_coords[c2][path_CB_inds[c2]]
            plddt2  = path_plddt[c2]

            combined = np.vstack([coords1, coords2])
            diff     = combined[:, None, :] - combined[None, :, :]
            dists    = np.sqrt((diff ** 2).sum(axis=-1))

            contacts = np.argwhere(dists[:len(coords1), len(coords1):] <= contact_threshold_A)
            if len(contacts) == 0:
                continue

            mean_plddt = np.mean(
                np.concatenate([plddt1[contacts[:, 0]], plddt2[contacts[:, 1]]])
            )
            score += np.log10(len(contacts) + 1) * mean_plddt

    return score


def score_complex(
    path_coords: dict,
    path_CA_inds: dict,
    path_CB_inds: dict,
    path_plddt: dict,
    ucrosslinks: pd.DataFrame,
    crosslinker_length: int = 35,
) -> float:
    """
    Compute the overall score of a protein complex assembly.

    The final score is the product of a structural quality score and a
    crosslink satisfaction score:

    .. math::

        \\text{Score}_{\\text{final}} =
            \\text{Score}_{\\text{struct}} \\times \\text{Score}_{\\text{XL}}

    Parameters
    ----------
    path_coords : dict[str, np.ndarray]
        Atomic coordinates indexed by chain ID.
    path_CA_inds : dict[str, np.ndarray]
        Cα indices indexed by chain ID.
    path_CB_inds : dict[str, np.ndarray]
        Cβ indices (Cα for glycine) indexed by chain ID.
    path_plddt : dict[str, np.ndarray]
        Per-residue pLDDT scores indexed by chain ID.
    ucrosslinks : pd.DataFrame
        Crosslink table (``ChainA``, ``ResidueA``, ``ChainB``, ``ResidueB``).
    crosslinker_length : int, optional
        Maximum allowed Cα–Cα distance for a satisfied crosslink (Å),
        default 35.

    Returns
    -------
    float
        Combined assembly score (≥ 0).
    """
    s_struct = score_structure(path_coords, path_CB_inds, path_plddt)
    _, _, s_xl = score_crosslinks(
        ucrosslinks, path_coords, path_CA_inds, crosslinker_length
    )
    return s_struct * s_xl


# ─────────────────────────────────────────────────────────────
# MCTS node
# ─────────────────────────────────────────────────────────────

class MonteCarloTreeSearchNode:
    """
    A single node in the MCTS tree for protein complex assembly.

    Each node represents one chain that has been docked onto the growing
    assembly.  The node stores the rotated coordinates of that chain, its
    pLDDT values, and metadata needed to score and expand the tree.

    The algorithm is adapted from:

    - https://ai-boson.github.io/mcts/
    - https://github.com/patrickbryant1/MoLPC

    Parameters
    ----------
    chain : str
        Chain ID of the protein placed at this node.
    edge_chain : str
        Chain ID of the neighbour used to align this chain.
    chain_coords : np.ndarray
        Rotated atomic coordinates for ``chain``.
    chain_CA_inds : np.ndarray
        Cα indices within ``chain_coords``.
    chain_CB_inds : np.ndarray
        Cβ indices within ``chain_coords`` (Cα for glycine).
    chain_pdb : np.ndarray
        Raw ATOM lines for ``chain`` (used when writing the final PDB).
    chain_plddt : np.ndarray
        Per-atom pLDDT values for ``chain``.
    edges : np.ndarray, shape (N, 2)
        All pairwise interaction edges in the network.
    sources : np.ndarray, shape (N,)
        Source model identifier for each edge.
    pairdir : str
        Root directory containing per-pair dimer PDB sub-folders.
    plddt_dir : str
        Alias for ``pairdir`` (kept for API compatibility).
    chain_lens : dict[str, int]
        Sequence lengths indexed by chain ID.
    ucrosslinks : pd.DataFrame
        Crosslink restraint table.
    outdir : str
        Output directory (used when writing logs and structures).
    source : str or None, optional
        Source model from which this chain was taken.
    complex_scores : list[float], optional
        Initial score list; defaults to ``[0]``.
    parent : MonteCarloTreeSearchNode or None, optional
        Parent node in the tree.
    parent_path : list[str], optional
        Chain IDs of all ancestor nodes.
    total_chains : int, optional
        Total number of chains to assemble (termination criterion).
    crosslinker_length : int, optional
        Maximum Cα–Cα distance (Å) for a satisfied crosslink (default 35).
    """

    def __init__(
        self,
        chain, edge_chain, chain_coords, chain_CA_inds, chain_CB_inds,
        chain_pdb, chain_plddt,
        edges, sources, pairdir, plddt_dir, chain_lens,
        ucrosslinks, outdir,
        source=None,
        complex_scores=None,
        parent=None,
        parent_path=None,
        total_chains=0,
        crosslinker_length=35,
    ):
        # Chain-specific structural data
        self.chain        = chain
        self.edge_chain   = edge_chain
        self.chain_coords = chain_coords
        self.CA_inds      = chain_CA_inds
        self.CB_inds      = chain_CB_inds
        self.pdb          = chain_pdb
        self.plddt        = chain_plddt

        # Global assembly configuration
        self.edges              = edges
        self.sources            = sources
        self.ucrosslinks        = ucrosslinks
        self.pairdir            = pairdir
        self.plddt_dir          = plddt_dir
        self.chain_lens         = chain_lens
        self.outdir             = outdir
        self.crosslinker_length = crosslinker_length

        # Source dimer model
        self.source = source

        # MCTS statistics
        self.complex_scores    = complex_scores if complex_scores is not None else [0]
        self._number_of_visits = 0

        # Tree structure
        self.parent   = parent
        self.path     = copy.deepcopy(parent_path) if parent_path else []
        self.path.append(chain)
        self.children = []

        self.total_chains = total_chains
        self._untried_edges, self._untried_sources = self._get_possible_edges()

    # ── Private helpers ──────────────────────────────────────

    def _get_possible_edges(self) -> tuple:
        """
        Return all network edges that can extend the current assembly path.

        An edge is valid if exactly one of its two endpoint chains is *not*
        yet in ``self.path`` (i.e. it would introduce a new chain).

        Returns
        -------
        untried_edges : list[np.ndarray]
            Edges (length-2 arrays) that introduce a new chain.
        untried_sources : list[str]
            Corresponding source model identifiers.
        """
        untried_edges   = []
        untried_sources = []

        for chain in self.path:
            row_inds = np.argwhere(self.edges == chain)[:, 0]
            cedges   = self.edges[row_inds]
            csources = self.sources[row_inds]

            for edge, src in zip(cedges, csources):
                if len(np.setdiff1d(edge, self.path)) > 0:
                    untried_edges.append(edge)
                    untried_sources.append(src)

        return untried_edges, untried_sources

    def _load_dimer_pdb(self, source: str, edge: np.ndarray) -> tuple:
        """
        Load the PDB file for a dimer given its source name and edge.

        Tries both chain orderings (A-B and B-A) and returns the one
        that exists on disk.

        Parameters
        ----------
        source : str
            Source model identifier (subfolder name under ``pairdir``).
        edge : np.ndarray
            Two-element array with the chain IDs of the dimer.

        Returns
        -------
        tuple
            Output of ``read_pdb``.
        """
        folder = os.path.join(self.pairdir, source)
        path_ab = os.path.join(folder, f"{source}_{edge[0]}-{source}_{edge[1]}.pdb")
        path_ba = os.path.join(folder, f"{source}_{edge[1]}-{source}_{edge[0]}.pdb")
        pdb_file = path_ab if os.path.exists(path_ab) else path_ba
        return read_pdb(pdb_file)

    def _load_plddt(self, source: str, pdb_chains: dict, chain_id: str) -> np.ndarray:
        """
        Extract per-atom pLDDT values for a single chain from a confidence JSON.

        Parameters
        ----------
        source : str
            Source model identifier.
        pdb_chains : dict
            Parsed PDB chains (used to count atoms per chain).
        chain_id : str
            Chain whose pLDDT values are extracted.

        Returns
        -------
        np.ndarray
            Per-atom pLDDT array for ``chain_id``.
        """
        conf_path = os.path.join(self.pairdir, source, f"{source}_confidences.json")
        with open(conf_path) as fh:
            conf = json.load(fh)
        all_plddts = np.array(conf["atom_plddts"])

        offset = 0
        for ch in source.split("_")[-1]:
            n_atoms = len(pdb_chains[ch])
            if ch == chain_id:
                return all_plddts[offset: offset + n_atoms]
            offset += n_atoms
        raise ValueError(f"Chain '{chain_id}' not found in source '{source}'.")

    # ── MCTS operations ──────────────────────────────────────

    def expand(self):
        """
        Add one new chain to the assembly using an untried network edge.

        The new chain is superposed onto the existing assembly by aligning
        the shared (edge) chain, then the rotated coordinates are checked
        for steric clashes before creating a child node.

        Returns
        -------
        MonteCarloTreeSearchNode or None
            The new child node, or ``None`` if the expansion produced a
            steric clash.
        """
        new_edge   = self._untried_edges.pop()
        new_source = self._untried_sources.pop()

        new_chain  = np.setdiff1d(new_edge, self.path)[0]
        edge_chain = np.setdiff1d(new_edge, new_chain)[0]

        # Load dimer structure and pLDDT
        pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = \
            self._load_dimer_pdb(new_source, new_edge)
        new_chain_plddt = self._load_plddt(new_source, pdb_chains, new_chain)

        # Find the ancestor node for the edge chain
        edge_node = self
        while edge_node.chain != edge_chain:
            edge_node = edge_node.parent

        # Superpose the dimer onto the assembly via the shared edge chain
        sup = SVDSuperimposer()
        sup.set(edge_node.chain_coords, np.array(chain_coords[edge_chain]))
        sup.run()
        rot, tran = sup.get_rotran()
        rotated_coords = np.dot(np.array(chain_coords[new_chain]), rot) + tran

        # Gather assembly data from all ancestor nodes
        path_coords  = {}
        path_CA_inds = {}
        path_CB_inds = {}
        path_plddt   = {}
        node = self
        while node:
            path_coords[node.chain]  = node.chain_coords
            path_CA_inds[node.chain] = node.CA_inds
            path_CB_inds[node.chain] = node.CB_inds
            path_plddt[node.chain]   = node.plddt
            node = node.parent

        # Reject if steric clash detected
        if check_overlaps(path_coords, path_CA_inds, rotated_coords, chain_CA_inds[new_chain]):
            return None

        # Score the new assembly
        path_coords[new_chain]  = rotated_coords
        path_CA_inds[new_chain] = np.array(chain_CA_inds[new_chain])
        path_CB_inds[new_chain] = np.array(chain_CB_inds[new_chain])
        path_plddt[new_chain]   = new_chain_plddt

        complex_score = score_complex(
            path_coords, path_CA_inds, path_CB_inds, path_plddt,
            self.ucrosslinks, self.crosslinker_length,
        )

        child_node = MonteCarloTreeSearchNode(
            new_chain, edge_chain, rotated_coords,
            np.array(chain_CA_inds[new_chain]),
            np.array(chain_CB_inds[new_chain]),
            np.array(pdb_chains[new_chain]),
            new_chain_plddt,
            self.edges, self.sources, self.pairdir, self.plddt_dir,
            self.chain_lens, self.ucrosslinks, self.outdir,
            source=new_source,
            complex_scores=[complex_score],
            parent=self,
            parent_path=self.path,
            total_chains=self.total_chains,
            crosslinker_length=self.crosslinker_length,
        )
        self.children.append(child_node)
        return child_node

    def rollout(self) -> float:
        """
        Simulate a random assembly from the current node to completion.

        At each step a random valid edge is chosen, the new chain is
        superposed and clash-checked.  The simulation stops when all
        chains have been placed or a steric clash is encountered.

        Returns
        -------
        float
            Final assembly score of the simulated trajectory.
        """
        # Initialise rollout state from the current tree path
        rollout_path = []
        path_coords  = {}
        path_CA_inds = {}
        path_CB_inds = {}
        path_plddt   = {}

        node = self
        while node.parent:
            rollout_path.append(node.chain)
            path_coords[node.chain]  = node.chain_coords
            path_CA_inds[node.chain] = node.CA_inds
            path_CB_inds[node.chain] = node.CB_inds
            path_plddt[node.chain]   = node.plddt
            node = node.parent

        def _available_edges(path: list) -> tuple:
            """Return edges and sources that can extend ``path``."""
            edges_out   = []
            sources_out = []
            for chain in path:
                row_inds = np.argwhere(self.edges == chain)[:, 0]
                for edge, src in zip(self.edges[row_inds], self.sources[row_inds]):
                    if len(np.setdiff1d(edge, path)) > 0:
                        edges_out.append(edge)
                        sources_out.append(src)
            return edges_out, sources_out

        overlap = False
        while len(rollout_path) < self.total_chains and not overlap:
            avail_edges, avail_sources = _available_edges(rollout_path)
            if not avail_edges:
                break

            idx        = np.random.randint(len(avail_edges))
            new_edge   = avail_edges[idx]
            new_source = avail_sources[idx]
            new_chain  = np.setdiff1d(new_edge, rollout_path)[0]
            edge_chain = np.setdiff1d(new_edge, new_chain)[0]

            pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = \
                self._load_dimer_pdb(new_source, new_edge)
            new_chain_plddt = self._load_plddt(new_source, pdb_chains, new_chain)

            sup = SVDSuperimposer()
            sup.set(path_coords[edge_chain], np.array(chain_coords[edge_chain]))
            sup.run()
            rot, tran  = sup.get_rotran()
            rotated    = np.dot(np.array(chain_coords[new_chain]), rot) + tran

            overlap = check_overlaps(
                path_coords, path_CA_inds, rotated,
                np.array(chain_CA_inds[new_chain]),
            )

            if not overlap:
                rollout_path.append(new_chain)
                path_coords[new_chain]  = rotated
                path_CA_inds[new_chain] = np.array(chain_CA_inds[new_chain])
                path_CB_inds[new_chain] = np.array(chain_CB_inds[new_chain])
                path_plddt[new_chain]   = new_chain_plddt

        return score_complex(
            path_coords, path_CA_inds, path_CB_inds, path_plddt,
            self.ucrosslinks, self.crosslinker_length,
        )

    def back_prop(self, rollout_score: float) -> None:
        """
        Backpropagate a rollout score up the tree to the root.

        Parameters
        ----------
        rollout_score : float
            Score obtained from a completed rollout simulation.
        """
        self._number_of_visits += 1
        self.complex_scores.append(rollout_score)
        if self.parent:
            self.parent.back_prop(rollout_score)

    def best_child(self) -> "MonteCarloTreeSearchNode":
        """
        Select the child with the highest UCB1 score.

        UCB1 balances exploitation (high mean score) with exploration
        (low visit count):

        .. math::

            \\text{UCB}_i = \\bar{V}_i +
                2\\sqrt{\\frac{\\ln N}{n_i}}

        where :math:`N` is the parent visit count and :math:`n_i` is the
        child visit count.

        Returns
        -------
        MonteCarloTreeSearchNode
            Child node with the highest UCB1 value.
        """
        ucb_scores = [
            np.average(c.complex_scores)
            + 2.0 * np.sqrt(
                np.log(c.parent._number_of_visits + 1e-3)
                / (c._number_of_visits + 1e-12)
            )
            for c in self.children
        ]
        return self.children[int(np.argmax(ucb_scores))]

    def tree_policy(self) -> tuple:
        """
        Descend the tree to select a node for expansion or rollout.

        The policy prefers nodes with untried edges.  If all edges have
        been tried it follows the ``best_child`` path.

        Returns
        -------
        node : MonteCarloTreeSearchNode
            The selected node.
        dead_end : bool
            ``True`` if no further expansion is possible.
        """
        current = self
        while True:
            if current._untried_edges:
                return current.expand(), False
            if current.children:
                current = current.best_child()
            else:
                return current, True

    def best_path(self) -> "MonteCarloTreeSearchNode":
        """
        Run MCTS from this root node until the assembly is complete.

        Iterates the *selection → expansion → rollout → backpropagation*
        cycle until all chains in the network have been placed.

        Returns
        -------
        MonteCarloTreeSearchNode
            Terminal node representing the best assembly path found.
        """
        n_placed = 0
        while n_placed < self.total_chains:
            v, dead_end = self.tree_policy()
            if v is None:
                break
            if dead_end:
                break
            rollout_score = v.rollout()
            v.back_prop(rollout_score)
            n_placed = len(v.path)
            logger.info(" -> ".join(v.path))
        return v


# ─────────────────────────────────────────────────────────────
# Assembly utilities
# ─────────────────────────────────────────────────────────────

def build_path_dict_from_node(node: MonteCarloTreeSearchNode) -> tuple:
    """
    Extract coordinate and Cα index dictionaries from a terminal MCTS node.

    Traverses the ancestry chain up to the root, collecting structural data
    for every chain in the assembled complex.

    Parameters
    ----------
    node : MonteCarloTreeSearchNode
        Terminal (leaf) node of an MCTS run.

    Returns
    -------
    path_coords : dict[str, np.ndarray]
        Atomic coordinates indexed by chain ID.
    path_CA_inds : dict[str, np.ndarray]
        Cα atom indices indexed by chain ID.
    """
    path_coords  = {}
    path_CA_inds = {}
    while node:
        path_coords[node.chain]  = node.chain_coords
        path_CA_inds[node.chain] = node.CA_inds
        node = node.parent
    return path_coords, path_CA_inds


def find_paths(
    edges: np.ndarray,
    sources: np.ndarray,
    pairdir: str,
    chain_lens: dict,
    ucrosslinks: pd.DataFrame,
    outdir: str,
    crosslinker_length: int = 35,
) -> MonteCarloTreeSearchNode:
    """
    Run MCTS from every chain as root and return the globally best assembly.

    For each unique chain in the interaction network:

    1. A fresh MCTS tree is initialised with that chain as the root.
    2. MCTS is run to completion (``best_path``).
    3. The resulting assembly is scored.
    4. The globally best assembly (across all roots) is tracked.

    Parameters
    ----------
    edges : np.ndarray, shape (N, 2)
        All pairwise interaction edges.
    sources : np.ndarray, shape (N,)
        Source model identifiers for each edge.
    pairdir : str
        Directory containing per-pair dimer PDB sub-folders.
    chain_lens : dict[str, int]
        Sequence lengths indexed by chain ID.
    ucrosslinks : pd.DataFrame
        Crosslink restraint table.
    outdir : str
        Output directory (for logging).
    crosslinker_length : int, optional
        Maximum Cα–Cα distance (Å) for a satisfied crosslink (default 35).

    Returns
    -------
    MonteCarloTreeSearchNode
        Terminal node of the globally best assembly path.
    """
    unique_chains = np.unique(edges)
    num_chains    = len(unique_chains)

    best_global_root  = None
    best_global_path  = None
    best_global_score = -np.inf

    logger.info(f"Running MCTS with {num_chains} root candidates...")

    for root_chain in unique_chains:
        logger.info("")
        logger.info(f"=== MCTS root = {root_chain} ===")

        row_inds = np.argwhere(edges == root_chain)
        if len(row_inds) == 0:
            logger.info(f"No edges for chain {root_chain}, skipping.")
            continue

        row    = row_inds[0][0]
        sps    = edges[row]
        ssr    = sources[row]
        folder = os.path.join(pairdir, ssr)

        pdb_ab = os.path.join(folder, f"{ssr}_{sps[0]}-{ssr}_{sps[1]}.pdb")
        pdb_ba = os.path.join(folder, f"{ssr}_{sps[1]}-{ssr}_{sps[0]}.pdb")
        pdb_file = pdb_ab if os.path.exists(pdb_ab) else pdb_ba

        pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pdb_file)

        conf_path = os.path.join(folder, f"{ssr}_confidences.json")
        with open(conf_path) as fh:
            conf = json.load(fh)
        all_plddts = np.array(conf["atom_plddts"])

        offset = 0
        for ch in ssr.split("_")[-1]:
            n_atoms = len(pdb_chains[ch])
            if ch == root_chain:
                root_plddt = all_plddts[offset: offset + n_atoms]
            offset += n_atoms

        root = MonteCarloTreeSearchNode(
            root_chain, "",
            np.array(chain_coords[root_chain]),
            np.array(chain_CA_inds[root_chain]),
            np.array(chain_CB_inds[root_chain]),
            np.array(pdb_chains[root_chain]),
            root_plddt,
            edges, sources, pairdir, pairdir,
            chain_lens, ucrosslinks, outdir,
            source=None,
            complex_scores=[0],
            parent=None,
            parent_path=[],
            total_chains=num_chains,
            crosslinker_length=crosslinker_length,
        )

        best_path   = root.best_path()
        final_score = float(np.mean(best_path.complex_scores))

        final_coords, final_ca = build_path_dict_from_node(best_path)
        xl_total, xl_inter, xl_final = score_crosslinks(
            ucrosslinks, final_coords, final_ca, crosslinker_length
        )

        logger.info(
            f"Root {root_chain}: MCTS={final_score:.3f} | "
            f"XL_total={xl_total:.3f} | XL_inter={xl_inter:.3f} | "
            f"XL_final={xl_final:.3f}"
        )

        if final_score > best_global_score:
            best_global_score = final_score
            best_global_path  = best_path
            best_global_root  = root_chain

    logger.info("")
    logger.info(
        f"===== BEST ROOT = {best_global_root}, "
        f"SCORE = {best_global_score:.3f} ====="
    )

    final_coords, final_ca = build_path_dict_from_node(best_global_path)
    xl_total, xl_inter, xl_final = score_crosslinks(
        ucrosslinks, final_coords, final_ca, crosslinker_length
    )
    logger.info("===== FINAL CROSSLINK SCORES =====")
    logger.info(f"Total : {xl_total:.3f}")
    logger.info(f"Inter : {xl_inter:.3f}")
    logger.info(f"Final : {xl_final:.3f}")

    return best_global_path


def write_pdb(best_path: MonteCarloTreeSearchNode, outdir: str) -> None:
    """
    Write the assembled complex as a single PDB file.

    Coordinates in each ATOM line are replaced with the final rotated/
    translated positions produced during MCTS.

    Parameters
    ----------
    best_path : MonteCarloTreeSearchNode
        Terminal node of the best assembly.
    outdir : str
        Output directory.  Created if it does not exist.
    """
    os.makedirs(outdir, exist_ok=True)
    out_file = os.path.join(outdir, "best_complex.pdb")

    with open(out_file, "w") as fh:
        node = best_path
        while node:
            for atom_line, coord in zip(node.pdb, node.chain_coords):
                x, y, z = (format(v, ".3f") for v in coord)
                outline = (
                    atom_line[:30]
                    + f"{x:>8}{y:>8}{z:>8}"
                    + atom_line[54:]
                )
                fh.write(outline)
            node = node.parent if node.parent else None
            if node is None:
                break


def create_path_df(best_path: MonteCarloTreeSearchNode, outdir: str) -> None:
    """
    Save the optimal assembly pathway to a CSV file.

    Records the chain addition order, the bridging (edge) chain, and the
    source dimer model used at each step.

    Parameters
    ----------
    best_path : MonteCarloTreeSearchNode
        Terminal node of the best assembly.
    outdir : str
        Output directory.  Created if it does not exist.
    """
    os.makedirs(outdir, exist_ok=True)
    records = []
    node = best_path
    while node.parent:
        records.append(
            {"Chain": node.chain, "Edge_chain": node.edge_chain, "Source": node.source}
        )
        node = node.parent

    df = pd.DataFrame(records)
    out_csv = os.path.join(outdir, "optimal_path.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"Assembly path saved to: {out_csv}")
    logger.info(f"Total chains placed: {len(df) + 1}")


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def main(args) -> None:
    """
    Execute the MCTS-based complex assembly pipeline.

    Loads all required input files, runs MCTS from each chain as root,
    writes the best assembly as a PDB file, and saves the path metadata.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain the following attributes:

        - ``network``          – path to ``network.csv``
        - ``pairdir``          – path to the dimer pair directory
        - ``useqs``            – path to ``useqs.csv``
        - ``ucrosslinks``      – path to ``ucrosslinks.csv``
        - ``outdir``           – output directory
        - ``crosslinker_length`` – Cα–Cα cut-off in Å (default 35)
    """
    crosslinker_length = getattr(args, "crosslinker_length", 35)

    os.makedirs(args.outdir, exist_ok=True)
    log_file = os.path.join(args.outdir, "mcts_run.log")
    setup_logger(log_file)

    logger.info("Loading input files...")
    network     = pd.read_csv(args.network)
    useqs       = pd.read_csv(args.useqs)
    ucrosslinks = pd.read_csv(args.ucrosslinks)

    edges   = np.array(network[["Chain1", "Chain2"]])
    sources = np.array(network["Source"])

    useqs["Chain_length"] = useqs["Sequence"].apply(len)
    chain_lens = dict(zip(useqs["Chain"].values, useqs["Chain_length"].values))

    logger.info(
        f"Network: {len(network)} edges | "
        f"Chains: {len(chain_lens)} | "
        f"Crosslinks: {len(ucrosslinks)} | "
        f"Crosslinker length: {crosslinker_length} Å"
    )

    best_path = find_paths(
        edges, sources, args.pairdir, chain_lens, ucrosslinks,
        args.outdir, crosslinker_length,
    )

    write_pdb(best_path, args.outdir)
    create_path_df(best_path, args.outdir)

    logger.info("Assembly complete.")


# ─────────────────────────────────────────────────────────────
# Script entry point (debug / direct execution)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Edit the paths below to run this module directly without CLI arguments.
    _debug_args = argparse.Namespace(
        network            = r"PATH\TO\assembled_complex\network.csv",
        pairdir            = r"PATH\TO\assembled_complex\pairs",
        useqs              = r"PATH\TO\assembled_complex\useqs.csv",
        ucrosslinks        = r"PATH\TO\assembled_complex\ucrosslinks.csv",
        outdir             = r"PATH\TO\assembled_complex\output",
        crosslinker_length = 35,
    )
    main(_debug_args)
