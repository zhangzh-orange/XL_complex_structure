from Bio.PDB import MMCIFParser
from Bio.PDB import PDBParser, PDBIO, Select, Superimposer
import os
import pandas as pd
import json
import numpy as np
import shutil
from pathlib import Path
from .mcts import score_crosslinks, read_pdb

class ProteinNucleicAcidSelect(Select):
    def accept_residue(self, residue):
        # 只保留标准氨基酸和核苷酸
        hetfield, resseq, icode = residue.id
        # 过滤掉 HETATM (小分子配体、水分子等)
        if hetfield == " ":
            return True
        return False
    
def rewrite_af_cif_structure(af_pred_folder, chains_df_path, output_folder):
    chains_df = pd.read_csv(chains_df_path)
    genes_chains_map = chains_df.set_index("Gene")["Chain"].to_dict()
    genes_chains_map = {k.lower(): v for k, v in genes_chains_map.items()}
    print("Gene -> Chain mapping:", genes_chains_map)

    os.makedirs(output_folder, exist_ok=True)
    parser = MMCIFParser(QUIET=True)
    io = PDBIO()

    for name in os.listdir(af_pred_folder):
        path = os.path.join(af_pred_folder, name)
        if os.path.isdir(path):
            cif_files = [f for f in os.listdir(path) if f.endswith(".cif")]
            if not cif_files:
                continue
            cif_path = os.path.join(path, cif_files[0])

            # 生成 chain 映射
            genes = str(name).split("_")[:-1]
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            mapping = {letters[i]: genes_chains_map[gene.lower()] for i, gene in enumerate(genes)}
            print(f"Folder: {name}, mapping: {mapping}")

            # 读取结构
            structure = parser.get_structure(name, cif_path)

            # 修改 Structure 中 chain.id 以匹配新的链名
            for model in structure:
                for chain in model:
                    if chain.id in mapping:
                        chain.id = mapping[chain.id]

            # 保存为 PDB
            folder_name = os.path.basename(path.rstrip("/\\"))
            output_path = os.path.join(
                output_folder,
                folder_name,
                f"{folder_name}.pdb"
            )

            # Ensure folder exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            out_pdb_path = os.path.join(output_path)
            io.set_structure(structure)
            io.save(out_pdb_path, ProteinNucleicAcidSelect())
            print(f"Saved rewritten PDB: {out_pdb_path}")
    print("Done!")


# TODO:对confidence文件进行修改，包括名字，对应链符号，去除所有非氨基酸打分
def rewrite_af_score_file(af_pred_folder, chains_df_path, output_folder):
    chains_df = pd.read_csv(chains_df_path)
    genes_chains_map = chains_df.set_index("Gene")["Chain"].to_dict()
    genes_chains_map = {k.lower(): v for k, v in genes_chains_map.items()}
    print("Gene -> Chain mapping:", genes_chains_map)

    for name in os.listdir(af_pred_folder):
        path = os.path.join(af_pred_folder, name)
        if os.path.isdir(path):
            score_files = [f for f in os.listdir(path) if (f.endswith("_confidences.json") and ("summary" not in f))]
            if not score_files:
                continue
            score_path = os.path.join(path, score_files[0])

            # 生成 chain 映射
            genes = str(name).split("_")[:-1]
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            mapping = {letters[i]: genes_chains_map[gene.lower()] for i, gene in enumerate(genes)}
            print(f"Folder: {name}, mapping: {mapping}")

            all_proteins_elements = mapping.keys()

            # 读取分数json
            with open(score_path, 'r') as f:
                conf = json.load(f)

            # print(conf.keys())
            # for key in conf.keys():
            #     print(np.array(conf[key]).shape)
            
            # Filter atom-level data
            atom_mask = [cid in all_proteins_elements for cid in conf["atom_chain_ids"]]
            conf["atom_chain_ids"] = [cid for cid, keep in zip(conf["atom_chain_ids"], atom_mask) if keep]
            conf["atom_chain_ids"] = [mapping[cid] for cid in conf["atom_chain_ids"]]
            conf["atom_plddts"] = conf["atom_plddts"][:len(conf["atom_chain_ids"])]

            # Filter token-level data
            token_mask = [cid in all_proteins_elements for cid in conf["token_chain_ids"]]
            conf["token_chain_ids"] = [cid for cid, keep in zip(conf["token_chain_ids"], token_mask) if keep]
            conf["token_chain_ids"] = [mapping[cid] for cid in conf["token_chain_ids"]]

            token_len = len(conf["token_chain_ids"])
            conf["contact_probs"] = np.array(conf["contact_probs"])[:token_len, :token_len].tolist()
            conf["pae"] = np.array(conf["pae"])[:token_len, :token_len].tolist()
            conf["token_res_ids"] = conf["token_res_ids"][:token_len]

            # for key in conf.keys():
            #     print(np.array(conf[key]).shape)

            # Build output path
            folder_name = os.path.basename(path.rstrip("/\\"))
            output_path = os.path.join(
                output_folder,
                folder_name,
                f"{folder_name}_AdjustedConfidences.json"
            )

            # Ensure folder exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save conf as JSON
            with open(output_path, "w") as f:
                f.write("{\n")
                for i, (k, v) in enumerate(conf.items()):
                    # Convert value to JSON
                    line = json.dumps(k) + ": " + json.dumps(v)
                    # Add comma except after last line
                    if i < len(conf) - 1:
                        line += ","
                    f.write("  " + line + "\n")
                f.write("}")
    print("Done!")



class ChainSelect(Select):
    def __init__(self, allowed_chains):
        self.allowed_chains = allowed_chains

    def accept_chain(self, chain):
        return chain.id in self.allowed_chains

def get_unique_filename(folder, base_name, ext=".pdb"):
    """如果文件存在，自动添加 -1, -2 ... 后缀"""
    path = os.path.join(folder, base_name + ext)
    if not os.path.exists(path):
        return path

    counter = 1
    while True:
        new_path = os.path.join(folder, f"{base_name}-{counter}{ext}")
        if not os.path.exists(new_path):
            return new_path
        counter += 1

# TODO:分割3mers到2mers,修改链名，相同的二聚体储存在同一个文件夹下，新建该二聚体名字的文件夹
# def split_trimer_to_dimer(input_pdb_path, output_folder):
#     # 检查是否为trimer，是的话进行下一步，不是直接输出原dimer


#     # 检查生成dimer是否再输出folder中已有，有的话保存为后缀-num的名称

#     # 保存名字和MoLPC相似
#     pass

# def split_trimer_to_dimers(rewrited_pdb_folder, output_folder):
#     for name in os.listdir(rewrited_pdb_folder):
#         path = os.path.join(rewrited_pdb_folder, name)
#         if os.path.isdir(path):
#             pdb_path = os.path.join(path, name+'.pdb')
#             score_path = os.path.join(path, name+'_AdjustedConfidences.json')

#             parser = PDBParser(QUIET=True)
#             structure = parser.get_structure("structure", pdb_path)

#             # Get list of chain IDs (expecting 3)
#             chains = [chain.id for chain in structure.get_chains()]
#             num_chains = len(chains)

#             if num_chains not in [2, 3]:
#                 raise ValueError(f"Expected 2 or 3 chains, found {num_chains}")

#             io = PDBIO()

#             if num_chains == 2:
#                 # 只有两条链，按字母排序
#                 dimer_chains = tuple(sorted(chains))
#                 folder_path = os.path.join(output_folder, f"{''.join(dimer_chains)}")
#                 os.makedirs(folder_path, exist_ok=True)

#                 base_name = f"{''.join(dimer_chains)}_{dimer_chains[0]}-{''.join(dimer_chains)}_{dimer_chains[1]}"
#                 out_pdb = get_unique_filename(folder_path, base_name, ext=".pdb")

#                 io.set_structure(structure)
#                 io.save(out_pdb, ChainSelect(dimer_chains))
#                 print(f"Saved 2-chain dimer: chains {dimer_chains} → {out_pdb}")

#                 # 复制score json文件
#                 out_conf = get_unique_filename(folder_path, f"{''.join(dimer_chains)}", ext="_confidences.json")
#                 with open(score_path, 'r') as f:
#                     conf = json.load(f)
#                 with open(out_conf, "w") as f:
#                     f.write("{\n")
#                     for i, (k, v) in enumerate(conf.items()):
#                         # Convert value to JSON
#                         line = json.dumps(k) + ": " + json.dumps(v)
#                         # Add comma except after last line
#                         if i < len(conf) - 1:
#                             line += ","
#                         f.write("  " + line + "\n")
#                     f.write("}")


#             else:
#                 # 三条链，生成三种二聚体组合，按字母排序
#                 dimer_sets = [
#                     tuple(sorted((chains[0], chains[1]))),
#                     tuple(sorted((chains[0], chains[2]))),
#                     tuple(sorted((chains[1], chains[2]))),
#                 ]

#                 with open(score_path, 'r') as f:
#                     conf = json.load(f)

#                 for i, dimer_chains in enumerate(dimer_sets, start=1):
#                     folder_path = os.path.join(output_folder, f"{''.join(dimer_chains)}")
#                     os.makedirs(folder_path, exist_ok=True)

#                     base_name = f"{''.join(dimer_chains)}_{dimer_chains[0]}-{''.join(dimer_chains)}_{dimer_chains[1]}"
#                     out_pdb = get_unique_filename(folder_path, base_name, ext=".pdb")

#                     io.set_structure(structure)
#                     io.save(out_pdb, ChainSelect(dimer_chains))
#                     print(f"Saved dimer {i}: chains {dimer_chains} → {out_pdb}")

#                     # 处理score json文件
#                     out_conf = get_unique_filename(folder_path, f"{''.join(dimer_chains)}", ext="_confidences.json")

#                     conf_dimer = {}

#                     # 转为 numpy 数组，快速掩码
#                     atom_chain_ids = np.array(conf["atom_chain_ids"])
#                     atom_plddts = np.array(conf["atom_plddts"])
#                     mask_atoms = np.isin(atom_chain_ids, dimer_chains)
#                     conf_dimer["atom_chain_ids"] = atom_chain_ids[mask_atoms].tolist()
#                     conf_dimer["atom_plddts"] = atom_plddts[mask_atoms].tolist()

#                     token_chain_ids = np.array(conf["token_chain_ids"])
#                     token_res_ids = np.array(conf["token_res_ids"])
#                     contact_probs = np.array(conf["contact_probs"])
#                     pae = np.array(conf["pae"])
#                     mask_tokens = np.isin(token_chain_ids, dimer_chains)

#                     conf_dimer["token_chain_ids"] = token_chain_ids[mask_tokens].tolist()
#                     conf_dimer["token_res_ids"] = token_res_ids[mask_tokens].tolist()
#                     conf_dimer["contact_probs"] = contact_probs[mask_tokens][:, mask_tokens].tolist()
#                     conf_dimer["pae"] = pae[mask_tokens][:, mask_tokens].tolist()

#                     # for key in conf_dimer.keys():
#                     #     print(np.array(conf_dimer[key]).shape)
#                     with open(out_conf, "w") as f:
#                         f.write("{\n")
#                         for i, (k, v) in enumerate(conf_dimer.items()):
#                             # Convert value to JSON
#                             line = json.dumps(k) + ": " + json.dumps(v)
#                             # Add comma except after last line
#                             if i < len(conf_dimer) - 1:
#                                 line += ","
#                             f.write("  " + line + "\n")
#                         f.write("}")
#     print("Done!")

def split_trimer_to_dimers(
    rewrited_pdb_folder,
    output_folder,
    progress_file="split_progress.json"
):
    progress_file = os.path.join(output_folder, progress_file)
    # ====== 1. 读取进度 ======
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            finished = set(json.load(f))
    else:
        finished = set()

    parser = PDBParser(QUIET=True)
    io = PDBIO()

    for name in os.listdir(rewrited_pdb_folder):
        # ====== 2. 跳过已完成 ======
        if name in finished:
            print(f"[SKIP] {name} already processed")
            continue

        path = os.path.join(rewrited_pdb_folder, name)
        if not os.path.isdir(path):
            continue

        try:
            pdb_path = os.path.join(path, name + ".pdb")
            score_path = os.path.join(path, name + "_AdjustedConfidences.json")

            structure = parser.get_structure("structure", pdb_path)

            chains = [chain.id for chain in structure.get_chains()]
            num_chains = len(chains)

            if num_chains not in [2, 3]:
                raise ValueError(f"Expected 2 or 3 chains, found {num_chains}")

            # ========= 2-chain =========
            if num_chains == 2:
                dimer_chains = tuple(sorted(chains))
                folder_path = os.path.join(output_folder, "".join(dimer_chains))
                os.makedirs(folder_path, exist_ok=True)

                base_name = f"{''.join(dimer_chains)}_{dimer_chains[0]}-{''.join(dimer_chains)}_{dimer_chains[1]}"
                out_pdb = get_unique_filename(folder_path, base_name, ext=".pdb")

                io.set_structure(structure)
                io.save(out_pdb, ChainSelect(dimer_chains))

                with open(score_path) as f:
                    conf = json.load(f)

                out_conf = get_unique_filename(
                    folder_path, "".join(dimer_chains), ext="_confidences.json"
                )
                with open(out_conf, "w") as f:
                    json.dump(conf, f)

            # ========= 3-chain =========
            else:
                dimer_sets = [
                    tuple(sorted((chains[0], chains[1]))),
                    tuple(sorted((chains[0], chains[2]))),
                    tuple(sorted((chains[1], chains[2]))),
                ]

                with open(score_path) as f:
                    conf = json.load(f)

                atom_chain_ids = np.array(conf["atom_chain_ids"])
                atom_plddts = np.array(conf["atom_plddts"])
                token_chain_ids = np.array(conf["token_chain_ids"])
                token_res_ids = np.array(conf["token_res_ids"])
                contact_probs = np.array(conf["contact_probs"])
                pae = np.array(conf["pae"])

                for dimer_chains in dimer_sets:
                    folder_path = os.path.join(output_folder, "".join(dimer_chains))
                    os.makedirs(folder_path, exist_ok=True)

                    base_name = f"{''.join(dimer_chains)}_{dimer_chains[0]}-{''.join(dimer_chains)}_{dimer_chains[1]}"
                    out_pdb = get_unique_filename(folder_path, base_name, ext=".pdb")

                    io.set_structure(structure)
                    io.save(out_pdb, ChainSelect(dimer_chains))

                    mask_atoms = np.isin(atom_chain_ids, dimer_chains)
                    mask_tokens = np.isin(token_chain_ids, dimer_chains)

                    conf_dimer = {
                        "atom_chain_ids": atom_chain_ids[mask_atoms].tolist(),
                        "atom_plddts": atom_plddts[mask_atoms].tolist(),
                        "token_chain_ids": token_chain_ids[mask_tokens].tolist(),
                        "token_res_ids": token_res_ids[mask_tokens].tolist(),
                        "contact_probs": contact_probs[mask_tokens][:, mask_tokens].tolist(),
                        "pae": pae[mask_tokens][:, mask_tokens].tolist(),
                    }

                    out_conf = get_unique_filename(
                        folder_path, "".join(dimer_chains), ext="_confidences.json"
                    )
                    with open(out_conf, "w") as f:
                        json.dump(conf_dimer, f)

            # ====== 3. 标记完成并立即写入 ======
            finished.add(name)
            with open(progress_file, "w") as f:
                json.dump(sorted(finished), f)

            print(f"[DONE] {name}")

        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            print("You can safely restart later.")
            continue

    print("All done!")
                
# TODO:初筛，对于每个二聚体只保留XL符合程度最高的，保存在另外的二聚体文件夹下
# ！！！Creative point
# 对于每个二聚体相互进行align，选取与其他所有二聚体差距最小的共识二聚体
# 保留相应pdb和confidence，并改名去掉数字
# 新建all文件夹，将所有的pdb和json move到all文件夹中
def select_most_central_pdb(pairs_folder: str):
    """
    遍历每个 dimer 文件夹，选择最接近其他二聚体的 PDB 文件，并复制到 dimer 文件夹中：
    - 最优 PDB 文件重命名为前两段名 + ".pdb"
    - 对应 confidence 文件也复制并重命名为 "<first_part>_confidences.json"
    """
    
    pairs_path = Path(pairs_folder)
    
    # 遍历每个子文件夹
    for dimer_folder in [d for d in pairs_path.iterdir() if d.is_dir()]:
        # ========= 1. 检查是否已经完成 =========
        pdb_files_root = list(dimer_folder.glob("*.pdb"))
        json_files_root = list(dimer_folder.glob("*_confidences.json"))

        if len(pdb_files_root) == 1 and len(json_files_root) == 1:
            print(f"[SKIP] {dimer_folder.name} already finalized")
            continue

        print(f"[PROCESS] {dimer_folder.name}")

        # ========= 2. 准备 all 文件夹 =========
        all_folder = dimer_folder / "all"
        all_folder.mkdir(exist_ok=True)
        
        # 移动文件到 all 文件夹
        for item in dimer_folder.iterdir():
            if item.is_file():
                target_path = all_folder / item.name
                # 如果目标文件存在，先删除
                if target_path.exists():
                    target_path.unlink()
                shutil.move(str(item), str(all_folder))
                print(f"Moved: {item} -> {all_folder}")
        
        pdb_files = [f for f in all_folder.iterdir() if f.suffix == ".pdb"]
        if not pdb_files:
            print(f"No PDB files in {all_folder}, skipping...")
            continue
        
        parser = PDBParser(QUIET=True)
        structures = [parser.get_structure(f.stem, str(f)) for f in pdb_files]
        
        # 提取 Cα 原子
        def get_ca_atoms(structure):
            ca_atoms = []
            for model in structure:
                for chain in model:
                    for res in chain:
                        if 'CA' in res:
                            ca_atoms.append(res['CA'])
            return ca_atoms
        
        # 计算 RMSD 矩阵
        n = len(structures)
        rmsd_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                atoms1 = get_ca_atoms(structures[i])
                atoms2 = get_ca_atoms(structures[j])
                min_len = min(len(atoms1), len(atoms2))
                atoms1 = atoms1[:min_len]
                atoms2 = atoms2[:min_len]
                sup = Superimposer()
                sup.set_atoms(atoms1, atoms2)
                rmsd_matrix[i, j] = sup.rms
                rmsd_matrix[j, i] = sup.rms
        
        avg_rmsd = rmsd_matrix.mean(axis=1)
        best_index = np.argmin(avg_rmsd)
        
        print("RMSD Matrix:")
        print(rmsd_matrix)
        print("Average RMSD:", avg_rmsd)
        print(pdb_files[best_index])
        
        # 复制最优 PDB 文件
        src_file = pdb_files[best_index]
        base_name = src_file.stem
        cleaned_name = "-".join(base_name.split("-")[:2]) + ".pdb"
        tgt_file = dimer_folder / cleaned_name
        shutil.copy(src_file, tgt_file)
        
        # 复制对应 confidence 文件
        parts = base_name.split("-")
        first_part = parts[0].split("_")[0]
        if parts[-1].isdigit():
            conf_name = f"{first_part}-{parts[-1]}_confidences.json"
        else:
            conf_name = f"{first_part}_confidences.json"
        src_conf_file = src_file.parent / conf_name
        tgt_conf_file = dimer_folder / f"{first_part}_confidences.json"
        
        if src_conf_file.exists():
            shutil.copy(src_conf_file, tgt_conf_file)
        else:
            print(f"Warning: {src_conf_file} not found, skipping confidence file copy.")
    print("Done")



# def select_best_crosslink_pdb(pairs_folder: str,
#                               ucrosslinks_path: str):
#     """
#     遍历每个 dimer 文件夹，选择 final_score 最大的 PDB 文件，并复制到 dimer 文件夹中：
#     - 最优 PDB 文件重命名为前两段名 + ".pdb"
#     - 对应 confidence 文件也复制并重命名为 "<first_part>_confidences.json"
    
#     参数：
#     - pairs_folder: str, 包含各二聚体的文件夹
#     - ucrosslinks_df: pd.DataFrame, 所有 crosslink 信息，列 ['ChainA','ResidueA','ChainB','ResidueB']
#     - get_path_coords_and_CA_inds: function, 输入 PDB 文件路径，返回 (path_coords, path_CA_inds)
#     """
    
#     pairs_path = Path(pairs_folder)
#     ucrosslinks_df = pd.read_csv(ucrosslinks_path)
    
#     for dimer_folder in [d for d in pairs_path.iterdir() if d.is_dir()]:
#         all_folder = dimer_folder / "all"
#         all_folder.mkdir(exist_ok=True)
        
#         # 移动文件到 all 文件夹（覆盖已存在）
#         for item in dimer_folder.iterdir():
#             if item.is_file():
#                 target_path = all_folder / item.name
#                 if target_path.exists():
#                     target_path.unlink()
#                 shutil.move(str(item), str(all_folder))
#                 print(f"Moved: {item} -> {all_folder}")
        
#         pdb_files = [f for f in all_folder.iterdir() if f.suffix == ".pdb"]
#         if not pdb_files:
#             print(f"No PDB files in {all_folder}, skipping...")
#             continue
        
#         best_score = -1
#         best_file = None
        
#         # 遍历 PDB 文件，计算 final_score
#         for pdb_file in pdb_files:
#             # 根据 PDB 文件生成 path_coords 和 path_CA_inds
#             _, path_coords, path_CA_inds, _ = read_pdb(str(pdb_file))
#             # pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds
            
#             # 获取 PDB 文件中存在的链
#             chains_in_pdb = list(path_coords.keys())
            
#             # 过滤 crosslinks，只保留当前二聚体中的链
#             ucrosslinks_filtered = ucrosslinks_df[
#                 ucrosslinks_df["ChainA"].isin(chains_in_pdb) &
#                 ucrosslinks_df["ChainB"].isin(chains_in_pdb)
#             ]
            
#             # 计算 final_score
#             _, inter_score, _ = score_crosslinks(
#                 ucrosslinks=ucrosslinks_filtered,
#                 path_coords=path_coords,
#                 path_CA_inds=path_CA_inds,
#                 crosslinker_length=45,
#                 inter_prop=0.8
#             )
            
#             if inter_score > best_score:
#                 best_score = inter_score
#                 best_file = pdb_file
        
#         if best_file is None:
#             print(f"No valid PDB found for {dimer_folder.name}")
#             continue
        
#         print(f"Best PDB for {dimer_folder.name}: {best_file.name} with score {best_score}")
        
#         # 复制最优 PDB 文件
#         base_name = best_file.stem
#         cleaned_name = "-".join(base_name.split("-")[:2]) + ".pdb"
#         tgt_file = dimer_folder / cleaned_name
#         shutil.copy(best_file, tgt_file)
        
#         # 复制对应 confidence 文件
#         parts = base_name.split("-")
#         first_part = parts[0].split("_")[0]
#         if parts[-1].isdigit():
#             conf_name = f"{first_part}-{parts[-1]}_confidences.json"
#         else:
#             conf_name = f"{first_part}_confidences.json"
#         src_conf_file = best_file.parent / conf_name
#         tgt_conf_file = dimer_folder / f"{first_part}_confidences.json"
        
#         if src_conf_file.exists():
#             shutil.copy(src_conf_file, tgt_conf_file)
#         else:
#             print(f"Warning: {src_conf_file} not found, skipping confidence file copy.")

if __name__ == "__main__":

#     rewrite_af_cif_structure(
#     af_pred_folder=r"N:\08_NK_structure_prediction\data\CORVET_complex\afx_pred\2mer",
#     chains_df_path=r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\chains.csv",
#     output_folder=r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\rewrited_pdbs"
# )

#     rewrite_af_score_file(
#     af_pred_folder=r"N:\08_NK_structure_prediction\data\CORVET_complex\afx_pred\2mer",
#     chains_df_path=r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\chains.csv",
#     output_folder=r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\rewrited_pdbs"
# )

    # split_trimer_to_dimers(
    #     r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\rewrited_pdbs", 
    #     r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\pairs")

    select_most_central_pdb(r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\pairs")
    
    pass