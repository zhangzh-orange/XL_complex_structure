import argparse
import sys
import os
import numpy as np
import pandas as pd
import glob
import itertools
import math
import copy
from collections import Counter, defaultdict
from Bio.SVDSuperimposer import SVDSuperimposer
import pdb
import json
import logging

logger = logging.getLogger(__name__)

##############FUNCTIONS##############
def setup_logger(log_file):
    """
    Create and configure a logger that writes INFO-level logs
    both to a file and to the console.

    Parameters
    ----------
    log_file : str
        Path to the log file. Parent directories will be created if needed.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Prevent adding duplicate handlers (important for notebooks or repeated runs)
    if logger.handlers:
        return logger

    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def parse_atm_record(line):
    """
    Parse a single ATOM line from a PDB file into a structured record.

    Parameters
    ----------
    line : str
        A single line from a PDB file starting with 'ATOM'.

    Returns
    -------
    dict
        Dictionary containing parsed atom information such as
        atom name, residue name, chain ID, coordinates, occupancy,
        and B-factor.
    """
    record = defaultdict()
    record['name'] = line[0:6].strip()
    record['atm_no'] = int(line[6:11])
    record['atm_name'] = line[12:16].strip()
    record['atm_alt'] = line[17]
    record['res_name'] = line[17:20].strip()
    record['chain'] = line[21]
    record['res_no'] = int(line[22:26])
    record['insert'] = line[26].strip()
    record['resid'] = line[22:29]
    record['x'] = float(line[30:38])
    record['y'] = float(line[38:46])
    record['z'] = float(line[46:54])
    record['occ'] = float(line[54:60])
    record['B'] = float(line[60:66])

    return record


def read_pdb(pdbfile):
    """
    Read a PDB file and organize atom records by chain.

    For each chain, this function stores:
    - Raw ATOM lines
    - Atom coordinates
    - Indices of CA atoms
    - Indices of CB atoms (or CA for glycine)

    Parameters
    ----------
    pdbfile : str
        Path to the PDB file.

    Returns
    -------
    tuple
        pdb_chains : dict
            Mapping from chain ID to list of ATOM lines.
        chain_coords : dict
            Mapping from chain ID to list of atomic coordinates.
        chain_CA_inds : dict
            Mapping from chain ID to indices of CA atoms.
        chain_CB_inds : dict
            Mapping from chain ID to indices of CB atoms (or CA for GLY).
    """
    pdb_chains = {}
    chain_coords = {}
    chain_CA_inds = {}
    chain_CB_inds = {}

    with open(pdbfile) as file:
        for line in file:
            if not line.startswith("ATOM"):
                continue
            record = parse_atm_record(line)
            if record['chain'] in [*pdb_chains.keys()]:
                pdb_chains[record['chain']].append(line)
                chain_coords[record['chain']].append([record['x'],record['y'],record['z']])
                coord_ind+=1
                if record['atm_name']=='CA':
                    chain_CA_inds[record['chain']].append(coord_ind)
                if record['atm_name']=='CB' or (record['atm_name']=='CA' and record['res_name']=='GLY'):
                    chain_CB_inds[record['chain']].append(coord_ind)


            else:
                pdb_chains[record['chain']] = [line]
                chain_coords[record['chain']]= [[record['x'],record['y'],record['z']]]
                chain_CA_inds[record['chain']]= []
                chain_CB_inds[record['chain']]= []
                #Reset coord ind
                coord_ind = 0


    return pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds


def check_overlaps(path_coords: dict,
                   path_CA_inds: dict,
                   new_coords: np.ndarray,
                   new_CA_inds: np.ndarray):
    """
    Check whether a new chain has severe spatial overlap with any
    existing chain in the current path.

    The overlap is evaluated using Cα atoms only. If more than 50%
    of the Cα atoms of the shorter chain are within 5 Å of the other
    chain's Cα atoms, the chains are considered overlapping.

    Parameters
    ----------
    path_coords : dict
        Dictionary mapping chain IDs to arrays of atomic coordinates.
    path_CA_inds : dict
        Dictionary mapping chain IDs to indices of Cα atoms.
    new_coords : np.ndarray
        Coordinate array for the new chain.
    new_CA_inds : np.ndarray
        Indices of Cα atoms in the new chain.

    Returns
    -------
    bool
        True if a severe overlap is detected, False otherwise.
    """

    new_CAs = new_coords[new_CA_inds]
    l1 = len(new_CAs)

    for chain, coords in path_coords.items():
        p_CAs = coords[path_CA_inds[chain]]

        mat = np.vstack([new_CAs, p_CAs])
        d = mat[:, None, :] - mat[None, :, :]
        dists = np.sqrt((d ** 2).sum(axis=-1))

        overlap = dists[:l1, l1:] <= 5.0 # 5 Å threshold
        if overlap.sum() > 0.5 * min(l1, len(p_CAs)):
            return True

    return False

######## Crosslink-based Score ########
def cal_crosslink_distance(path_coords: dict,
                           path_CA_inds: dict,
                           a_chain: str,
                           a_res: int,
                           b_chain: str,
                           b_res: int,
                           crosslinker_length: int = 35):
    """
    Calculate the Cα–Cα distance between two residues and determine
    whether it satisfies a crosslinker distance constraint.

    The distance is computed between the Cα atoms of the specified
    residues from two chains. If either chain is missing, the
    distance is treated as infinite.

    Parameters
    ----------
    path_coords : dict
        Dictionary mapping chain IDs to arrays of atomic coordinates.
    path_CA_inds : dict
        Dictionary mapping chain IDs to indices of Cα atoms.
    a_chain : str
        Chain ID of the first residue.
    a_res : int
        Residue index (1-based) of the first residue.
    b_chain : str
        Chain ID of the second residue.
    b_res : int
        Residue index (1-based) of the second residue.
    crosslinker_length : int, optional
        Maximum allowed Cα–Cα distance for the crosslinker (in Å),
        by default 35.

    Returns
    -------
    tuple
        distance : float
            Cα–Cα distance rounded to two decimal places.
        satisfied : bool
            True if the distance is within the crosslinker length,
            False otherwise.
    """

    if a_chain not in path_coords or b_chain not in path_coords:
        return np.inf, False

    a_ca_atom = path_CA_inds[a_chain][a_res-1]
    b_ca_atom = path_CA_inds[b_chain][b_res-1]

    a_coord = path_coords[a_chain][a_ca_atom]
    b_coord = path_coords[b_chain][b_ca_atom]

    dist = np.linalg.norm(a_coord - b_coord)
    return round(dist, 2), dist <= crosslinker_length


def score_crosslinks(ucrosslinks: pd.DataFrame,
                     path_coords: dict,
                     path_CA_inds: dict,
                     crosslinker_length: int = 35,
                     inter_prop: float = 1):
    """
    Score how well a structural model satisfies a set of crosslinking restraints.

    Each crosslink is evaluated based on the Cα–Cα distance between the specified
    residues. A crosslink is considered satisfied if the distance is within the
    specified crosslinker length. Scores are computed for all crosslinks and
    separately for inter-chain crosslinks.

    Parameters
    ----------
    ucrosslinks : pd.DataFrame
        DataFrame containing crosslink information with columns:
        ['ChainA', 'ResidueA', 'ChainB', 'ResidueB'].
    path_coords : dict
        Dictionary mapping chain IDs to arrays of atomic coordinates.
    path_CA_inds : dict
        Dictionary mapping chain IDs to indices of Cα atoms.
    crosslinker_length : int, optional
        Maximum allowed Cα–Cα distance for a satisfied crosslink (in Å),
        by default 35.
    inter_prop : float, optional
        Weighting factor for inter-chain crosslink satisfaction in the
        final score (0–1), by default 1.

    Returns
    -------
    tuple
        score_total : float
            Fraction of all crosslinks that are satisfied.
        score_inter : float
            Fraction of inter-chain crosslinks that are satisfied.
        final_score : float
            Weighted combination of total and inter-chain scores.
    """

    if len(ucrosslinks) == 0:
        return 1.0, 1.0, 1.0

    consist = 0
    inter_consist = 0
    inter_total = 0

    for _, row in ucrosslinks.iterrows():
        dist, ok = cal_crosslink_distance(
            path_coords, path_CA_inds,
            row["ChainA"], row["ResidueA"],
            row["ChainB"], row["ResidueB"],
            crosslinker_length
        )

        consist += ok
        if row["ChainA"] != row["ChainB"]:
            inter_total += 1
            inter_consist += ok

    score_total = consist / len(ucrosslinks)
    score_inter = 1.0 if inter_total == 0 else inter_consist / inter_total

    final = (1 - inter_prop) * score_total + inter_prop * score_inter
    return score_total, score_inter, final

######## plDDT-based Score ########
def score_structure(path_coords: dict,
                    path_CB_inds: dict,
                    path_plddt: dict):

    score = 0.0
    chains = list(path_coords.keys())

    for i, c1 in enumerate(chains):
        coords1 = path_coords[c1][path_CB_inds[c1]]
        plddt1 = path_plddt[c1]

        for c2 in chains[i+1:]:
            coords2 = path_coords[c2][path_CB_inds[c2]]
            plddt2 = path_plddt[c2]

            mat = np.vstack([coords1, coords2])
            d = mat[:, None, :] - mat[None, :, :]
            dists = np.sqrt((d ** 2).sum(axis=-1))

            contacts = np.argwhere(dists[:len(coords1), len(coords1):] <= 8.0)
            if len(contacts) == 0:
                continue

            mean_plddt = np.mean(
                np.concatenate([plddt1[contacts[:, 0]],
                                plddt2[contacts[:, 1]]])
            )

            score += np.log10(len(contacts) + 1) * mean_plddt

    return score

def score_complex(path_coords, path_CA_inds, path_CB_inds, path_plddt, ucrosslinks):
    """
    Compute the overall score of a protein complex by combining
    structural quality and crosslink satisfaction.

    The final score is calculated as the product of:
    - a structure-based score (geometry and confidence)
    - a crosslink-based score (distance restraint satisfaction)

    Parameters
    ----------
    path_coords : dict
        Dictionary mapping chain IDs to arrays of atomic coordinates.
    path_CA_inds : dict
        Dictionary mapping chain IDs to indices of Cα atoms.
    path_CB_inds : dict
        Dictionary mapping chain IDs to indices of Cβ atoms
        (or Cα for glycine).
    path_plddt : dict
        Dictionary mapping chain IDs to per-residue pLDDT confidence scores.
    ucrosslinks : pd.DataFrame
        DataFrame containing crosslink information.

    Returns
    -------
    float
        Final complex score combining structural quality and
        crosslink satisfaction.
    """
    s_struct = score_structure(path_coords, path_CB_inds, path_plddt)
    _, _, s_xl = score_crosslinks(ucrosslinks, path_coords, path_CA_inds)
    return s_struct * s_xl

class MonteCarloTreeSearchNode():
    """
    Node representation for Monte Carlo Tree Search (MCTS)
    applied to protein complex assembly.

    Each node represents a decision step where a new chain
    is added to the current assembly path. Nodes track
    structural data, scoring history, and possible expansions.

    Based on https://ai-boson.github.io/mcts/ and 
    https://github.com/patrickbryant1/MoLPC 
    """
    def __init__(self, chain, edge_chain, chain_coords, chain_CA_inds, chain_CB_inds, chain_pdb, chain_plddt,
                edges, sources, pairdir, plddt_dir, chain_lens, ucrosslinks, outdir,
                source=None, complex_scores=[0], parent=None, parent_path=[], total_chains=0):
        # Chain-specific data
        self.chain = chain
        self.edge_chain = edge_chain
        self.chain_coords = chain_coords
        self.CA_inds = chain_CA_inds
        self.CB_inds = chain_CB_inds
        self.pdb = chain_pdb
        self.plddt = chain_plddt

        # Global configuration
        self.edges = edges
        self.sources = sources
        self.ucrosslinks = ucrosslinks
        self.pairdir = pairdir
        self.plddt_dir = plddt_dir
        self.chain_lens = chain_lens
        self.outdir = outdir

        # Source model from which this chain originates
        self.source = source

        # Scores from all rollouts passing through this node
        # Used to estimate the expected value of this node
        self.complex_scores = complex_scores

        # Tree structure
        self.parent = parent
        self.path = copy.deepcopy(parent_path) #All nodes up to (and including) the parent
        self.path.append(chain)
        self.children = [] #All nodes branching out from the current

        # Visit statistics
        self._number_of_visits = 0
        self._untried_edges, self._untried_sources = self.get_possible_edges()
        self.total_chains=total_chains

        return

    def get_possible_edges(self):
        """
        Determine all valid edges that can extend the current path.

        Returns
        -------
        tuple
            untried_edges : list
                Possible edges that introduce a new chain.
            untried_sources : list
                Corresponding source models for each edge.
        """

        untried_edges = []
        untried_sources = []
        for chain in self.path:
            #Get all edges to the current node
            cedges = self.edges[np.argwhere(self.edges==chain)[:,0]]
            csources = self.sources[np.argwhere(self.edges==chain)[:,0]]
            #Go through all edges and see that the new connection is not in path
            for i in range(len(cedges)):
                diff = np.setdiff1d(cedges[i],self.path)
                if len(diff)>0:
                    untried_edges.append(cedges[i])
                    untried_sources.append(csources[i])

        return untried_edges, untried_sources

    def expand(self):
        """
        Expand the tree by adding one new chain using an untried edge.

        Returns
        -------
        MonteCarloTreeSearchNode or None
            Newly created child node if expansion is valid,
            otherwise None (e.g., due to overlaps).
        """
        new_edge = self._untried_edges.pop()
        new_source = self._untried_sources.pop()

        new_chain = np.setdiff1d(new_edge, self.path)[0]
        edge_chain =  np.setdiff1d(new_edge, new_chain)[0]

        # Load the PDB file containing the interacting chain pair
        pdb_folder = os.path.join(self.pairdir,new_source)
        if os.path.exists(pdb_folder+"/"+new_source+'_'+new_edge[0]+'-'+new_source+'_'+new_edge[1]+'.pdb'):
            pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pdb_folder+"/"+new_source+'_'+new_edge[0]+'-'+new_source+'_'+new_edge[1]+'.pdb')
        else:
            pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pdb_folder+"/"+new_source+'_'+new_edge[1]+'-'+new_source+'_'+new_edge[0]+'.pdb')

        # Load per-atom pLDDT confidence values
        with open(pdb_folder +"/"+ new_source + '_confidences.json', 'r') as f:
            conf = json.load(f)
        source_plDDT = np.array(conf['atom_plddts'])

        # Extract pLDDT values for the new chain
        si = 0
        for p_chain in new_source.split('_')[-1]:
            n_atoms = len(pdb_chains[p_chain])
            if p_chain == new_chain:
                new_chain_plddt = source_plDDT[si : si + n_atoms]
            si += n_atoms

        # Superimpose the new structure onto the existing assembly
        edge_node = self
        while edge_node.chain!=edge_chain:
            edge_node = edge_node.parent


        # Superimpose the new structure onto the existing assembly
        sup = SVDSuperimposer()
        sup.set(edge_node.chain_coords,np.array(chain_coords[edge_chain]))
        sup.run()
        rot, tran = sup.get_rotran()

        #Rotate coords from new chain to its new relative position/orientation
        rotated_coords = np.dot(np.array(chain_coords[new_chain]), rot) + tran

        # Gather current path data
        path_coords = {}
        path_CA_inds = {}
        path_CB_inds = {}
        path_plddt = {}

        path_node = self
        ucrosslinks = self.ucrosslinks
        while path_node:
            path_coords[path_node.chain] = path_node.chain_coords
            path_CA_inds[path_node.chain] = path_node.CA_inds
            path_CB_inds[path_node.chain] = path_node.CB_inds
            path_plddt[path_node.chain] = path_node.plddt
            path_node = path_node.parent


        # Check for steric overlaps
        overlap = check_overlaps(path_coords,path_CA_inds,rotated_coords,chain_CA_inds[new_chain])

        #If no overlap
        if overlap==False:
            # Add the new chain and score the complex
            path_coords[new_chain] = rotated_coords
            path_CA_inds[new_chain] = np.array(chain_CA_inds[new_chain])
            path_CB_inds[new_chain] = np.array(chain_CB_inds[new_chain])
            path_plddt[new_chain] = new_chain_plddt

            complex_score = score_complex(path_coords, path_CA_inds, path_CB_inds, path_plddt, ucrosslinks)

            child_node = MonteCarloTreeSearchNode(new_chain, edge_chain, rotated_coords, np.array(chain_CA_inds[new_chain]),
                    np.array(chain_CB_inds[new_chain]), np.array(pdb_chains[new_chain]), new_chain_plddt,
                    self.edges, self.sources, self.pairdir, self.plddt_dir, self.chain_lens, self.ucrosslinks, self.outdir,
                    source=new_source, complex_scores=[complex_score], parent=self, parent_path=self.path, total_chains=self.total_chains)

            self.children.append(child_node)
            return child_node

        else:
            #Only consider nodes that do not result in overlaps
            return None


    def rollout(self):
        """
        Perform a random simulation from the current node until
        1. all chains are added or 2. an overlap occurs.
        
        Returns
        -------
        float
            Final complex score of the rollout.
        """
        overlap = False
        rollout_path = []
        
        path_coords = {}
        path_CA_inds = {}
        path_CB_inds = {}
        path_plddt = {}

        path_node = self
        ucrosslinks = self.ucrosslinks

        while path_node.parent:
            rollout_path.append(path_node.chain)
            path_coords[path_node.chain] = path_node.chain_coords
            path_CA_inds[path_node.chain] = path_node.CA_inds
            path_CB_inds[path_node.chain] = path_node.CB_inds
            path_plddt[path_node.chain] = path_node.plddt
            path_node = path_node.parent

        def get_possible_edges(path):
            """
            Get all valid edges that can extend a given path.
            Defined locally because self cannot be passed.
            """

            untried_edges = []
            untried_sources = []
            for chain in path:
                #Get all edges to the current node
                cedges = self.edges[np.argwhere(self.edges==chain)[:,0]]
                csources = self.sources[np.argwhere(self.edges==chain)[:,0]]
                # Go through all edges and see that the new connection is not in path
                for i in range(len(cedges)):
                    diff = np.setdiff1d(cedges[i],path)
                    if len(diff)>0:
                        untried_edges.append(cedges[i])
                        untried_sources.append(csources[i])

            return untried_edges, untried_sources

        while len(rollout_path)<self.total_chains and overlap==False:

            #Get untried edges and sources
            untried_edges, untried_sources = get_possible_edges(rollout_path)
            #Pick a random edge
            if len(untried_edges)>0:
                edge_ind = np.random.randint(len(untried_edges))
            else:
                overlap=True
                break
            new_edge = untried_edges[edge_ind]
            new_source = untried_sources[edge_ind]
            new_chain = np.setdiff1d(new_edge, rollout_path)[0]
            edge_chain = np.setdiff1d(new_edge, new_chain)[0]

            #Read the pdb file containing the new edge
            pdb_folder = os.path.join(self.pairdir,new_source)
            if os.path.exists(pdb_folder+"/"+new_source+'_'+new_edge[0]+'-'+new_source+'_'+new_edge[1]+'.pdb'):
                pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pdb_folder+"/"+new_source+'_'+new_edge[0]+'-'+new_source+'_'+new_edge[1]+'.pdb')
            else:
                pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pdb_folder+"/"+new_source+'_'+new_edge[1]+'-'+new_source+'_'+new_edge[0]+'.pdb')

            #plDDT - for scoring
            with open(pdb_folder+"/"+new_source + '_confidences.json', 'r') as f:
                conf = json.load(f)
            source_plDDT = np.array(conf['atom_plddts'])

            si = 0
            for p_chain in new_source.split('_')[-1]:
                n_atoms = len(pdb_chains[p_chain])  # ★ 新增：该链的原子数
                if p_chain == new_chain:
                    new_chain_plddt = source_plDDT[si : si + n_atoms]
                si += n_atoms

            # Align the overlapping chains
            # Get the coords for the other chain in the edge
            sup = SVDSuperimposer()
            sup.set(
                path_coords[edge_chain],
                np.array(chain_coords[edge_chain])
            )
            sup.run()
            rot, tran = sup.get_rotran()

            #Rotate coords from new chain to its new relative position/orientation
            rotated_coords = np.dot(np.array(chain_coords[new_chain]), rot) + tran

            #Check overlaps
            # overlap = check_overlaps(path_coords,path_CA_inds,rotated_coords,chain_CA_inds[new_chain])
            overlap = check_overlaps(
                path_coords,
                path_CA_inds,
                rotated_coords,
                np.array(chain_CA_inds[new_chain])
            )


            #If no overlap - score and create a child node
            if overlap==False:
                #Add the new chain
                rollout_path.append(new_chain)
                path_coords[new_chain] = rotated_coords
                path_CA_inds[new_chain] = np.array(chain_CA_inds[new_chain])
                path_CB_inds[new_chain] = np.array(chain_CB_inds[new_chain])
                path_plddt[new_chain] = new_chain_plddt

            else:
                break
        #Score rollout
        rollout_score = score_complex(path_coords, path_CA_inds, path_CB_inds, path_plddt, ucrosslinks)

        return rollout_score


    def back_prop(self, rollout_score):
        """
        Backpropagate a rollout score up the tree.
        """

        self._number_of_visits += 1
        self.complex_scores.append(rollout_score)
        #This is recursive and will back_prop to all parents
        if self.parent:
            self.parent.back_prop(rollout_score)

    def best_child(self):
        """
        Select the child node with the highest UCB score.
        """

        choices_weights = [(np.average(c.complex_scores) + 2 * np.sqrt(np.log(c.parent._number_of_visits+1e-3) / (c._number_of_visits+1e-12))) for c in self.children]

        return self.children[np.argmax(choices_weights)]

    def tree_policy(self):
        """
        Select a node for expansion or rollout using the tree policy.
        """
        current_node = self
        dead_end = False
        c=0
        while dead_end == False:
            if not len(current_node._untried_edges)==0: #Try possible moves
                return current_node.expand(), dead_end #Return the expansion
            else: #Select the best move - expand from there
                if len(current_node.children)>0:
                    current_node = current_node.best_child()
                else:
                    dead_end = True

        return current_node, dead_end

    def best_path(self):
        """
        Run MCTS until a full complex assembly is obtained.

        Returns
        -------
        MonteCarloTreeSearchNode
            Final node corresponding to the best assembly path.
        """
        nchains_in_path = 0
        n_total_chains = self.total_chains
        while nchains_in_path<n_total_chains: #add stop if there are no more options
            #Stop search if there are no more untried edges
            v, dead_end = self.tree_policy() #Returns expansion as long as there are actions
            if v:
                if dead_end==True:
                    nchains_in_path = n_total_chains
                else:
                    #Otherwise the best child is returned and the path is thus continued from there.
                    rollout_score = v.rollout() #Rollout
                    v.back_prop(rollout_score) #Backpropagate the score
                    nchains_in_path = len(v.path) #Check path len
                    logger.info(" -> ".join(v.path))

        return v


def build_path_dict_from_node(node):
    """
    Construct coordinate and Cα index dictionaries from a terminal MCTS node.

    The function traverses the node's ancestry up to the root and
    collects chain coordinates and Cα indices for the full assembly path.

    Parameters
    ----------
    node : MonteCarloTreeSearchNode
        Terminal node representing a complete or partial assembly path.

    Returns
    -------
    tuple
        path_coords : dict
            Mapping from chain ID to atomic coordinates.
        path_CA_inds : dict
            Mapping from chain ID to Cα atom indices.
    """
    path_coords = {}
    path_CA_inds = {}

    while node:
        path_coords[node.chain] = node.chain_coords
        path_CA_inds[node.chain] = node.CA_inds
        node = node.parent

    return path_coords, path_CA_inds


def find_paths(edges, sources, pairdir, chain_lens, ucrosslinks, outdir):
    """
    Run Monte Carlo Tree Search (MCTS) starting from each chain as a root
    and select the globally best assembly path.

    For each chain in the interaction network, the function:
    1. Initializes an MCTS tree with that chain as the root
    2. Runs MCTS to assemble a full complex
    3. Scores the resulting structure
    4. Keeps track of the best-scoring result globally

    Parameters
    ----------
    edges : np.ndarray
        Array of interacting chain pairs.
    sources : np.ndarray
        Array indicating the source model for each edge.
    pairdir : str
        Directory containing pairwise PDB models.
    chain_lens : dict
        Mapping from chain ID to sequence length.
    ucrosslinks : pd.DataFrame
        DataFrame containing unique crosslink restraints.
    outdir : str
        Output directory for logs and results.

    Returns
    -------
    MonteCarloTreeSearchNode
        Node corresponding to the best global assembly path.
    """

    nodes = np.unique(edges)
    num_nodes = len(nodes)

    best_global_root = None
    best_global_path = None
    best_global_score = -np.inf

    logger.info(f"Running MCTS {num_nodes} times (each node as root)...")

    # Iterate over each chain as root
    for root_chain in nodes:
        logger.info(" ")
        logger.info(f"=== Running MCTS with root = {root_chain} ===")

        # Find any edge connected to the root chain to initialize the structure
        idx = np.argwhere(edges == root_chain)
        if len(idx) == 0:
            logger.info(f"No edges found for chain {root_chain}, skipping.")
            continue

        row = idx[0][0]
        sps = edges[row]
        ssr = sources[row]
        start_pairdir = os.path.join(pairdir, ssr + "/")

        # Locate the correct PDB file
        pdb_file_1 = f"{start_pairdir}{ssr}_{sps[0]}-{ssr}_{sps[1]}.pdb"
        pdb_file_2 = f"{start_pairdir}{ssr}_{sps[1]}-{ssr}_{sps[0]}.pdb"

        if os.path.exists(pdb_file_1):
            pdb_file = pdb_file_1
        else:
            pdb_file = pdb_file_2

        pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pdb_file)

        # Load atom-level pLDDT scores
        with open(start_pairdir + ssr + '_confidences.json', 'r') as f:
            conf = json.load(f)
        source_plDDT = np.array(conf['atom_plddts'])

        # Slice pLDDT by atom counts (AF3-compatible)
        si = 0
        for ch in ssr.split('_')[-1]:
            n_atoms = len(pdb_chains[ch])
            if ch == root_chain:
                root_chain_plddt = source_plDDT[si:si + n_atoms]
            si += n_atoms

        # Initialize root node
        root = MonteCarloTreeSearchNode(
            root_chain, '', np.array(chain_coords[root_chain]),
            np.array(chain_CA_inds[root_chain]), np.array(chain_CB_inds[root_chain]),
            np.array(pdb_chains[root_chain]), root_chain_plddt,
            edges, sources, pairdir, pairdir,
            chain_lens, ucrosslinks, outdir,
            source=None, complex_scores=[0],
            parent=None, parent_path=[], total_chains=num_nodes
        )

        # Run MCTS
        best_path = root.best_path()

        # Score from MCTS rollouts
        final_score = np.mean(best_path.complex_scores)

        # Compute final crosslink score using full structure
        final_path_coords, final_path_CA_inds = build_path_dict_from_node(best_path)

        score_total, score_inter, score_final = score_crosslinks(
            ucrosslinks,
            final_path_coords,
            final_path_CA_inds
        )

        logger.info(
            f"Root {root_chain}: "
            f"MCTS = {final_score:.3f}, "
            f"XL_total = {score_total:.3f}, "
            f"XL_inter = {score_inter:.3f}, "
            f"XL_final = {score_final:.3f}"
        )

        # Track global best
        if final_score > best_global_score:
            best_global_score = final_score
            best_global_path = best_path
            best_global_root = root_chain

    logger.info(" ")
    logger.info(f"===== BEST GLOBAL ROOT = {best_global_root}, SCORE = {best_global_score:.3f} =====")

    # Final crosslink score for the best assembly
    final_path_coords, final_path_CA_inds = build_path_dict_from_node(best_global_path)

    score_total, score_inter, score_final = score_crosslinks(
        ucrosslinks,
        final_path_coords,
        final_path_CA_inds
    )

    logger.info(" ")
    logger.info("===== FINAL COMPLEX CROSSLINK SCORES =====")
    logger.info(f"Total crosslink score : {score_total:.3f}")
    logger.info(f"Inter-chain score     : {score_inter:.3f}")
    logger.info(f"Final weighted score  : {score_final:.3f}")

    return best_global_path


def write_pdb(best_path, outdir):
    """
    Write the assembled complex into a single PDB file.

    Coordinates are updated to reflect the final rotated/transformed
    positions obtained during MCTS.

    Parameters
    ----------
    best_path : MonteCarloTreeSearchNode
        Terminal node representing the best assembly.
    outdir : str
        Output directory for the PDB file.
    """

    current_node = best_path
    #Open a file to write to
    os.makedirs(outdir, exist_ok=True)
    with open(outdir+r'/best_complex.pdb', 'w') as file:
        while current_node.parent:
            chain_pdb = current_node.pdb
            chain_coords = current_node.chain_coords
            for i in range(len(chain_pdb)):
                line = chain_pdb[i]
                coord = chain_coords[i]
                #Get coords in str and calc blanks
                x,y,z =  format(coord[0],'.3f'), format(coord[1],'.3f'), format(coord[2],'.3f')
                x_blank = ' '*(8-len(x))
                y_blank = ' '*(8-len(y))
                z_blank = ' '*(8-len(z))
                outline = line[:30]+x_blank+x+y_blank+y+z_blank+z+line[54:]
                file.write(outline)
            current_node = current_node.parent
        #Write the last chain
        chain_pdb = current_node.pdb
        chain_coords = current_node.chain_coords
        for i in range(len(chain_pdb)):
            line = chain_pdb[i]
            coord = chain_coords[i]
            #Get coords in str and calc blanks
            x,y,z =  format(coord[0],'.3f'), format(coord[1],'.3f'), format(coord[2],'.3f')
            x_blank = ' '*(8-len(x))
            y_blank = ' '*(8-len(y))
            z_blank = ' '*(8-len(z))
            outline = line[:30]+x_blank+x+y_blank+y+z_blank+z+line[54:]
            file.write(outline)


def create_path_df(best_path, outdir):
    """
    Create and save a CSV file describing the optimal assembly path.

    The file records the order of chain additions, their connecting
    chains, and the source model for each step.

    Parameters
    ----------
    best_path : MonteCarloTreeSearchNode
        Terminal node of the optimal assembly.
    outdir : str
        Output directory for the CSV file.
    """
    #Create a df of all paths
    path_df = {'Chain':[], 'Edge_chain':[], 'Source':[]}

    current_node = best_path
    while current_node.parent:
        path_df['Chain'].append(current_node.chain)
        path_df['Edge_chain'].append(current_node.edge_chain)
        path_df['Source'].append(current_node.source)
        current_node = current_node.parent

    path_df = pd.DataFrame.from_dict(path_df)

    #Save
    path_df.to_csv(outdir+r'/optimal_path.csv', index=None)
    # logger.info('The best possible non-overlapping path has',len(path_df)+1,'chains')
    logger.info(f"The best possible non-overlapping path has {len(path_df)+1} chains")



#################MAIN####################

def main(args):
    """
    Main execution function for MCTS-based complex assembly.

    Loads input data, runs MCTS to find the optimal assembly path,
    writes the final PDB structure, and saves path metadata.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line or debug arguments.
    """

    #Data
    network = pd.read_csv(args.network)
    pairdir = args.pairdir
    useqs = pd.read_csv(args.useqs)
    outdir = args.outdir
    ucrosslinks = pd.read_csv(args.ucrosslinks)

    # Logs
    log_file = os.path.join(outdir, "mcts_run.log")
    logger = setup_logger(log_file)

    #Get all edges
    global edges, sources, chain_lens
    edges = np.array(network[['Chain1', 'Chain2']])
    sources = np.array(network['Source'])

    #Get all chain lengths
    useqs['Chain_length'] = [len(x) for x in useqs.Sequence]
    useqs = useqs[['Chain','Useq', 'Chain_length']]
    chain_lens = dict(zip(useqs.Chain.values,
                          useqs.Chain_length.values))

    #Find paths and assemble
    best_path = find_paths(edges, sources, pairdir, chain_lens, ucrosslinks, outdir)

    #Write PDB files of all complete paths
    write_pdb(best_path, outdir)

    #Create and save path df
    create_path_df(best_path, outdir)

if __name__ == "__main__":
    # ===== Optional: Debug mode (used when no command-line arguments are provided) =====
    # Modify the paths below if you want to run locally without CLI arguments
    debug_args = argparse.Namespace(
        network=r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\network.csv",
        pairdir=r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\pairs/",
        useqs=r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\useqs.csv",
        ucrosslinks = r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\ucrosslinks.csv",
        outdir=r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\output/",
    )

    # ===== If command-line arguments are provided, use them =====
    parser = argparse.ArgumentParser(
        description='Find optimal paths by Monte Carlo Tree Search.'
    )
    parser.add_argument('--network', type=str, help='Path to csv containing pairwise interactions.')
    parser.add_argument('--pairdir', type=str, help='Path to dir containing all connecting pairs')
    parser.add_argument('--useqs', type=str, help='CSV with unique seqs')
    parser.add_argument('--ucrosslinks', type=str, help='CSV with unique crosslinks')
    parser.add_argument('--outdir', type=str, help='Where to write all complexes')

    try:
        cmd_args = parser.parse_args()
        if all(v is None for v in vars(cmd_args).values()):
            logger.info("No command-line arguments detected — using debug arguments.")
            main(debug_args)
        else:
            main(cmd_args)
    except:
        logger.info("Argument parsing failed — using debug arguments.")
        main(debug_args)
    pass