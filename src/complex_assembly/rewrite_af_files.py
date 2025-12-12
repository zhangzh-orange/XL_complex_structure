from Bio.PDB import MMCIFParser,  Select
from Bio.PDB import PDBParser, PDBIO, Select
import os
import pandas as pd
import json
import numpy as np


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

def split_trimer_to_dimers(rewrited_pdb_folder, output_folder):
    for name in os.listdir(rewrited_pdb_folder):
        path = os.path.join(rewrited_pdb_folder, name)
        if os.path.isdir(path):
            pdb_path = os.path.join(path, name+'.pdb')
            score_path = os.path.join(path, name+'_AdjustedConfidences.json')

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("structure", pdb_path)

            # Get list of chain IDs (expecting 3)
            chains = [chain.id for chain in structure.get_chains()]
            num_chains = len(chains)

            if num_chains not in [2, 3]:
                raise ValueError(f"Expected 2 or 3 chains, found {num_chains}")

            io = PDBIO()

            if num_chains == 2:
                # 只有两条链，按字母排序
                dimer_chains = tuple(sorted(chains))
                folder_path = os.path.join(output_folder, f"{''.join(dimer_chains)}")
                os.makedirs(folder_path, exist_ok=True)

                base_name = f"{''.join(dimer_chains)}_{dimer_chains[0]}-{''.join(dimer_chains)}_{dimer_chains[1]}"
                out_pdb = get_unique_filename(folder_path, base_name, ext=".pdb")

                io.set_structure(structure)
                io.save(out_pdb, ChainSelect(dimer_chains))
                print(f"Saved 2-chain dimer: chains {dimer_chains} → {out_pdb}")

                # TODO：复制score json文件
                out_conf = get_unique_filename(folder_path, f"{''.join(dimer_chains)}", ext="_confidence.json")
                with open(score_path, 'r') as f:
                    conf = json.load(f)
                with open(out_conf, "w") as f:
                    f.write("{\n")
                    for i, (k, v) in enumerate(conf.items()):
                        # Convert value to JSON
                        line = json.dumps(k) + ": " + json.dumps(v)
                        # Add comma except after last line
                        if i < len(conf) - 1:
                            line += ","
                        f.write("  " + line + "\n")
                    f.write("}")


            else:
                # 三条链，生成三种二聚体组合，按字母排序
                dimer_sets = [
                    tuple(sorted((chains[0], chains[1]))),
                    tuple(sorted((chains[0], chains[2]))),
                    tuple(sorted((chains[1], chains[2]))),
                ]

                with open(score_path, 'r') as f:
                    conf = json.load(f)

                for i, dimer_chains in enumerate(dimer_sets, start=1):
                    folder_path = os.path.join(output_folder, f"{''.join(dimer_chains)}")
                    os.makedirs(folder_path, exist_ok=True)

                    base_name = f"{''.join(dimer_chains)}_{dimer_chains[0]}-{''.join(dimer_chains)}_{dimer_chains[1]}"
                    out_pdb = get_unique_filename(folder_path, base_name, ext=".pdb")

                    io.set_structure(structure)
                    io.save(out_pdb, ChainSelect(dimer_chains))
                    print(f"Saved dimer {i}: chains {dimer_chains} → {out_pdb}")

                    # TODO:处理score json文件
                    out_conf = get_unique_filename(folder_path, f"{''.join(dimer_chains)}", ext="_confidence.json")

                    conf_dimer = {}

                    # 转为 numpy 数组，快速掩码
                    atom_chain_ids = np.array(conf["atom_chain_ids"])
                    atom_plddts = np.array(conf["atom_plddts"])
                    mask_atoms = np.isin(atom_chain_ids, dimer_chains)
                    conf_dimer["atom_chain_ids"] = atom_chain_ids[mask_atoms].tolist()
                    conf_dimer["atom_plddts"] = atom_plddts[mask_atoms].tolist()

                    token_chain_ids = np.array(conf["token_chain_ids"])
                    token_res_ids = np.array(conf["token_res_ids"])
                    contact_probs = np.array(conf["contact_probs"])
                    pae = np.array(conf["pae"])
                    mask_tokens = np.isin(token_chain_ids, dimer_chains)

                    conf_dimer["token_chain_ids"] = token_chain_ids[mask_tokens].tolist()
                    conf_dimer["token_res_ids"] = token_res_ids[mask_tokens].tolist()
                    conf_dimer["contact_probs"] = contact_probs[mask_tokens][:, mask_tokens].tolist()
                    conf_dimer["pae"] = pae[mask_tokens][:, mask_tokens].tolist()

                    # for key in conf_dimer.keys():
                    #     print(np.array(conf_dimer[key]).shape)
                    with open(out_conf, "w") as f:
                        f.write("{\n")
                        for i, (k, v) in enumerate(conf_dimer.items()):
                            # Convert value to JSON
                            line = json.dumps(k) + ": " + json.dumps(v)
                            # Add comma except after last line
                            if i < len(conf_dimer) - 1:
                                line += ","
                            f.write("  " + line + "\n")
                        f.write("}")

                    




# TODO:初筛，对于每个二聚体只保留XL符合程度最高的，保存在另外的二聚体文件夹下
# 或者不筛，直接随机挑选


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

    split_trimer_to_dimers(
        r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\rewrited_pdbs", 
        r"N:\08_NK_structure_prediction\data\CORVET_complex\assembled_complex\pairs")
    
    pass