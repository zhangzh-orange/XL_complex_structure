#! python3
# -*- encoding: utf-8 -*-
# ===================================
# File    :   mcts.py
# Time    :   2025/12/08 12:27:15
# Author  :   Zehong Zhang
# Contact :   zhang@fmp-berlin.de
# ===================================

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

# TODO: 1. Read inputs:
# parser = argparse.ArgumentParser(description = '''Find optimal paths by Monte Carlo Tree Search.''')
# parser.add_argument('--network', nargs=1, type= str, default=sys.stdin, help = 'Path to csv containing pairwise interactions.')
# parser.add_argument('--pairdir', nargs=1, type= str, default=sys.stdin, help = 'Path to dir containing all connecting pairs')
# parser.add_argument('--plddt_dir', nargs=1, type= str, default=sys.stdin, help = 'Path to dir containing plDDT scores for each complex')
# parser.add_argument('--useqs', nargs=1, type= str, default=sys.stdin, help = 'CSV with unique seqs')
# parser.add_argument('--chain_seqs', nargs=1, type= str, default=sys.stdin, help = 'CSV with mapping btw useqs and chains')
# parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Where to write all complexes')

##############FUNCTIONS##############

def parse_atm_record(line):
    """
    Parse a PDB ATOM/HETATM record line.

    Parameters
    ----------
    line : str
        A single line from a PDB file.

    Returns
    -------
    dict or None
        Parsed atom information as a dictionary with fields:
            - name (str)
            - atm_no (int)
            - atm_name (str)
            - atm_alt (str)
            - res_name (str)
            - chain (str)
            - res_no (int)
            - insert (str)
            - resid (str)
            - x, y, z (float)
            - occ (float)
            - B (float)
        Returns None for non-ATOM/HETATM lines or improperly formatted lines.
    """
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return None

    try:
        return {
            'name':     line[0:6].strip(),
            'atm_no':   int(line[6:11]),
            'atm_name': line[12:16].strip(),
            'atm_alt':  line[16].strip(),
            'res_name': line[17:20].strip(),
            'chain':    line[21].strip(),
            'res_no':   int(line[22:26]),
            'insert':   line[26].strip(),
            'resid':    line[22:27].strip(),
            'x':        float(line[30:38]),
            'y':        float(line[38:46]),
            'z':        float(line[46:54]),
            'occ':      float(line[54:60]),
            'B':        float(line[60:66])
        }
    except ValueError:
        return None  # 非法格式行自动跳过


def read_pdb(pdbfile):
    """
    Read a PDB file and extract atom records by chain.

    Parameters
    ----------
    pdbfile : str
        Path to a PDB file.

    Returns
    -------
    tuple(dict, dict, dict, dict)
        A 4-tuple:
        - pdb_chains : dict[str, list[str]]
            Original PDB lines grouped by chain.
        - chain_coords : dict[str, list[list[float]]]
            Atom coordinates [[x,y,z], ...] per chain.
        - chain_CA_inds : dict[str, list[int]]
            Indices of CA atoms per chain.
        - chain_CB_inds : dict[str, list[int]]
            Indices of CB atoms (or CA for GLY) per chain.
    """
    pdb_chains = defaultdict(list)
    chain_coords = defaultdict(list)
    chain_CA_inds = defaultdict(list)
    chain_CB_inds = defaultdict(list)
    coord_ind = defaultdict(int)

    with open(pdbfile) as file:
        for line in file:
            record = parse_atm_record(line)
            if record is None:
                continue

            chain = record['chain']

            # 保存原始行
            pdb_chains[chain].append(line)

            # 保存坐标
            chain_coords[chain].append([record['x'], record['y'], record['z']])

            # 当前索引
            idx = coord_ind[chain]
            coord_ind[chain] += 1

            # CA
            if record['atm_name'] == 'CA':
                chain_CA_inds[chain].append(idx)

            # CB or pseudo CB for GLY
            if record['atm_name'] == 'CB' or (record['atm_name'] == 'CA' and record['res_name'] == 'GLY'):
                chain_CB_inds[chain].append(idx)

    return pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds


def read_cif(cif_file):
    """
    Read a mmCIF file and extract atom-site information by chain.

    Parameters
    ----------
    cif_file : str
        Path to a mmCIF file.

    Returns
    -------
    tuple(dict, dict, dict, dict)
        Same structure as read_pdb():
        - pdb_chains : dict[str, list[str]]
        - chain_coords : dict[str, list[list[float]]]
        - chain_CA_inds : dict[str, list[int]]
        - chain_CB_inds : dict[str, list[int]]
    """

    pdb_chains = defaultdict(list)
    chain_coords = defaultdict(list)
    chain_CA_inds = defaultdict(list)
    chain_CB_inds = defaultdict(list)
    coord_ind = defaultdict(int)

    # mmCIF atom-site fields
    atom_fields = {
        "group_PDB": None,
        "id": None,
        "type_symbol": None,
        "label_atom_id": None,
        "label_alt_id": None,
        "label_comp_id": None,
        "label_asym_id": None,
        "label_seq_id": None,
        "Cartn_x": None,
        "Cartn_y": None,
        "Cartn_z": None,
        "occupancy": None,
        "B_iso_or_equiv": None,
    }

    field_order = []
    atom_data_mode = False

    with open(cif_file) as f:
        for line in f:
            line = line.strip()

            # 进入 loop 解析 atom_site
            if line.startswith("loop_"):
                atom_data_mode = False
                continue

            # 记录 atom_site 的列名
            if line.startswith("_atom_site."):
                atom_data_mode = True

                field = line.split()[0].replace("_atom_site.", "")
                if field in atom_fields:
                    field_order.append(field)
                else:
                    field_order.append(None)  # 未使用字段
                continue

            # 遇到数据行（不以 _ 开头），并且在 atom_site 区域
            if atom_data_mode and not line.startswith("_") and line:
                values = line.split()

                record = {k: None for k in atom_fields}

                for i, field in enumerate(field_order):
                    if field is None or i >= len(values):
                        continue
                    record[field] = values[i]

                # 转换类型
                try:
                    atm_name = record["label_atom_id"]
                    chain = record["label_asym_id"]
                    res_name = record["label_comp_id"]
                    res_no = int(record["label_seq_id"])
                    x = float(record["Cartn_x"])
                    y = float(record["Cartn_y"])
                    z = float(record["Cartn_z"])
                    occ = float(record["occupancy"])
                    B = float(record["B_iso_or_equiv"])
                except:
                    continue  # 格式异常跳过

                # 保存原子行（构造模拟 PDB 行）
                pdb_line = f"ATOM  {record['id']:>5} {atm_name:<4}{res_name:>3} {chain}{res_no:>4}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{B:6.2f}\n"
                pdb_chains[chain].append(pdb_line)

                # 保存坐标
                idx = coord_ind[chain]
                coord_ind[chain] += 1
                chain_coords[chain].append([x, y, z])

                # CA index
                if atm_name == "CA":
                    chain_CA_inds[chain].append(idx)

                # CB index (GLY 用 CA )
                if atm_name == "CB" or (atm_name == "CA" and res_name == "GLY"):
                    chain_CB_inds[chain].append(idx)

    return pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds


def check_overlaps(cpath_coords,cpath_CA_inds,n_coords,n_CA_inds):
    '''Check assembly overlaps of new chain

    Parameters
    ----------
    cpath_coords : list of np.ndarray
        Coordinates of all atoms in currently assembled chains.
        Each element corresponds to one chain: [chain1_coords, chain2_coords, ...]
        Shape per chain: (n_atoms_i, 3) where 3 represents (x, y, z) coordinates
    
    cpath_CA_inds : list of np.ndarray
        Indices of alpha-carbon (Cα) atoms for each assembled chain.
        Used to extract backbone atoms from full coordinate sets.
        Example: [[0, 1, 4, ...], [0, 1, 4, ...], ...] where indices point to Cα positions
    
    n_coords : np.ndarray
        Coordinates of all atoms in the new chain to be added.
        Shape: (n_atoms_new, 3) - all atom positions for the candidate chain
    
    n_CA_inds : np.ndarray
        Indices of alpha-carbon (Cα) atoms in the new chain.
        Used to extract backbone from the new chain's full atom set.
        Example: [0, 1, 4, 5, ...] indices corresponding to Cα atoms
    '''
    #Check CA overlap
    n_CAs = n_coords[n_CA_inds] #New chain CAs
    l1 = (len(n_CAs))
    #Go through all previous CAs and compare

    for i in range(len(cpath_coords)):
        p_CAs = cpath_coords[i][cpath_CA_inds[i]]
        #Calc 2-norm
        mat = np.append(n_CAs, p_CAs,axis=0)
        a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
        dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
        contact_dists = dists[l1:,:l1]
        overlap = np.argwhere(contact_dists<=5) #5 Å threshold
        if overlap.shape[0]>0.5*min(l1,len(p_CAs)): #If over 50% overlap
            return True

    return False


def score_interfaces(path_coords, path_CB_inds, path_plddt):
    '''Score all interfaces in the current complex
    '''

    interfaces_score = 0
    chain_inds = np.arange(len(path_coords))
    #Get interfaces per chain
    for i in chain_inds:
        chain_coords = path_coords[i]
        chain_CB_inds = path_CB_inds[i]
        l1 = len(chain_CB_inds)
        chain_CB_coords = chain_coords[chain_CB_inds]
        chain_plddt = path_plddt[i]
        
        # 获取除当前链i外的所有其他链索引
        other_chains = np.setdiff1d(chain_inds, i)
        for int_i in other_chains:
            int_chain_CB_coords = path_coords[int_i][path_CB_inds[int_i]]
            int_chain_plddt = path_plddt[int_i]
            #Calc 2-norm
            mat = np.append(chain_CB_coords,int_chain_CB_coords,axis=0)
            a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
            dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
            contact_dists = dists[:l1,l1:]
            contacts = np.argwhere(contact_dists<=8)
            #The first axis contains the contacts from chain 1
            #The second the contacts from chain 2
            if contacts.shape[0]>0:
                av_if_plDDT =  np.concatenate((chain_plddt[contacts[:,0]], int_chain_plddt[contacts[:,1]])).mean()
                interfaces_score +=  np.log10(contacts.shape[0]+1)*av_if_plDDT
    
    return interfaces_score


def read_crosslinks(csv_path):
    pass

def score_crosslinks(path_coords, path_CA_inds):
    # TODO: 计算输入complex中所有crosslink的满足度，crosslink满足数目越多，打分越高
    # 取CA算距离
    crosslinks = {
        "start_CA_chains":[],
        "start_CA_inds":[],
        "end_CA_chains":[],
        "end_CA_inds":[]
        }
    crosslinker_length = 35 # A 限制距离
    total_links = len(crosslinks["start_CA_inds"]) + 1e-6

    satisfied_links = 0
    for s,e in zip(crosslinks["start_CA_inds"],crosslinks["end_CA_inds"]):
        # 计算欧几里得距离
        # 定义两个三维点
        point1 = np.array([1, 2, 3])
        point2 = np.array([4, 6, 8])

        # 计算差值向量
        diff = point1 - point2

        # 计算欧几里得距离
        distance = np.sqrt(np.sum(diff**2))
        if distance <= crosslinker_length:
            satisfied_links += 1
    crosslinks_score = satisfied_links/total_links
    return crosslinks_score

def score_complex():
    # TODO: complex_score = interfaces_score * crosslinks_score
    # return complex_score
    pass


class MonteCarloTreeSearchNode():
    pass

if __name__ == "__main__":
    # pdb_file = r"N:\08_NK_structure_prediction\XL_complex_structure\data\6PBG.pdb"
    # pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pdb_file)
    pdb_file = r"N:\08_NK_structure_prediction\XL_complex_structure\data\copa_cope_arfgap3_2_model.cif"
    pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_cif(pdb_file)
    print(chain_coords["A"][:10])
    # print(chain_coords)
    # print(chain_CA_inds)
    # print(chain_CB_inds)