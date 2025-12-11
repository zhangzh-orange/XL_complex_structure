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
                pdb_line = f"ATOM  {record['id']:>5} {atm_name:<4} {res_name:>3} {chain} {res_no:>4} " \
           f"{x:8.3f} {y:8.3f} {z:8.3f} {occ:6.2f} {B:6.2f}\n"
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


def write_cif(output_file, pdb_chains):
    """
    将 `pdb_chains`（伪 PDB 格式）写入一个新的 mmCIF 文件的 atom_site 表中。
    """

    atom_site_fields = [
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv"
    ]

    with open(output_file, "w") as f:

        f.write("loop_\n")
        for col in atom_site_fields:
            f.write(col + "\n")

        atom_id = 1

        for chain_id, lines in pdb_chains.items():
            for line in lines:

                parts = line.split()

                atm_name = parts[2]
                res_name = parts[3]
                chain = parts[4]                 # e.g., 'A'
                res_no = int(parts[5])           # e.g., '10'
                x, y, z = map(float, parts[6:9])
                occ = float(parts[9])
                B = float(parts[10])
                element = atm_name[0]

                f.write(
                    f"ATOM {atom_id} {element} {atm_name} . {res_name} {chain} {res_no} "
                    f"{x:.3f} {y:.3f} {z:.3f} {occ:.2f} {B:.2f}\n"
                )
                atom_id += 1
