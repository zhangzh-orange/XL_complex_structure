from Bio.PDB import MMCIFParser,  Select
from Bio.PDB import PDBIO
import os
import pandas as pd

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
            out_pdb_path = os.path.join(output_folder, f"{name}.pdb")
            io.set_structure(structure)
            io.save(out_pdb_path, ProteinNucleicAcidSelect())
            print(f"Saved rewritten PDB: {out_pdb_path}")


if __name__ == "__main__":
    # af_pred_folder = r"N:\08_NK_structure_prediction\data\LRBAandSNARE\afx_pred"
    # chains_df_path = r"N:\08_NK_structure_prediction\data\LRBAandSNARE\chains.csv"
    # rewrite_af_cif(af_pred_folder,chains_df_path)

    rewrite_af_cif_structure(
    af_pred_folder=r"N:\08_NK_structure_prediction\data\LRBAandSNARE\afx_pred",
    chains_df_path=r"N:\08_NK_structure_prediction\data\LRBAandSNARE\chains.csv",
    output_folder=r"N:\08_NK_structure_prediction\data\LRBAandSNARE\rewrited_cifs"
)
    
    pass