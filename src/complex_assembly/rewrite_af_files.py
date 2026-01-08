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
    """
    Biopython Select class that keeps only standard protein and
    nucleic-acid residues when writing a PDB file.

    All HETATM records (ligands, ions, water, etc.) are excluded.
    """
    def accept_residue(self, residue):
        """
        Accept only standard residues (ATOM records).

        Parameters
        ----------
        residue : Bio.PDB.Residue
            Residue object from a Biopython Structure.

        Returns
        -------
        bool
            True if the residue should be kept.
        """
        hetfield, resseq, icode = residue.id
        if hetfield == " ":
            return True
        return False
    
def rewrite_af_cif_structure(af_pred_folder, chains_df_path, output_folder):
    """
    Rewrite AlphaFold CIF structures into PDB format with renamed chains.

    This function:
    - Reads AlphaFold CIF files from subfolders
    - Renames chain IDs according to a Gene→Chain mapping table
    - Removes non-protein / non-nucleic-acid residues
    - Writes cleaned PDB files to the output directory

    Parameters
    ----------
    af_pred_folder : str
        Directory containing AlphaFold prediction subfolders.
    chains_df_path : str
        CSV file mapping gene names to desired chain IDs.
    output_folder : str
        Directory to store rewritten PDB files.
    """
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

            # Generate chains map
            genes = str(name).split("_")[:-1]
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            mapping = {letters[i]: genes_chains_map[gene.lower()] for i, gene in enumerate(genes)}
            print(f"Folder: {name}, mapping: {mapping}")

            # Read structure
            structure = parser.get_structure(name, cif_path)

            for model in structure:
                for chain in model:
                    if chain.id in mapping:
                        chain.id = mapping[chain.id]

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


def rewrite_af_score_file(af_pred_folder, chains_df_path, output_folder):
    """
    Rewrite AlphaFold confidence JSON files with updated chain IDs
    and filtered content.

    This function:
    - Renames chain IDs using a Gene→Chain mapping
    - Removes all non-protein atoms and tokens
    - Truncates PAE and contact probability matrices accordingly
    - Writes adjusted confidence JSON files to the output directory

    Parameters
    ----------
    af_pred_folder : str
        Directory containing AlphaFold prediction subfolders.
    chains_df_path : str
        CSV file mapping gene names to desired chain IDs.
    output_folder : str
        Directory to store adjusted confidence JSON files.
    """
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

            genes = str(name).split("_")[:-1]
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            mapping = {letters[i]: genes_chains_map[gene.lower()] for i, gene in enumerate(genes)}
            print(f"Folder: {name}, mapping: {mapping}")

            all_proteins_elements = mapping.keys()

            with open(score_path, 'r') as f:
                conf = json.load(f)
            
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
    """
    Biopython Select class that keeps only a specified subset of chains.
    """
    def __init__(self, allowed_chains):
        self.allowed_chains = allowed_chains

    def accept_chain(self, chain):
        return chain.id in self.allowed_chains

def get_unique_filename(folder, base_name, ext=".pdb"):
    """
    Generate a unique filename by appending numeric suffixes if needed.

    If the file already exists, the function will try:
    base_name-1.ext, base_name-2.ext, ...

    Parameters
    ----------
    folder : str
        Target directory.
    base_name : str
        Filename without extension.
    ext : str, optional
        File extension, by default ".pdb".

    Returns
    -------
    str
        A unique file path that does not yet exist.
    """
    path = os.path.join(folder, base_name + ext)
    if not os.path.exists(path):
        return path

    counter = 1
    while True:
        new_path = os.path.join(folder, f"{base_name}-{counter}{ext}")
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def split_trimer_to_dimers(
    rewrited_pdb_folder,
    output_folder,
    progress_file="split_progress.json"
):
    """
    Split 2-chain and 3-chain AlphaFold models into standardized dimers.

    Behavior:
    - 2-chain structures are copied directly
    - 3-chain structures are split into all possible dimers
    - PDB and confidence JSON files are filtered per dimer
    - Progress is saved to allow safe restart

    Parameters
    ----------
    rewrited_pdb_folder : str
        Folder containing rewritten PDB structures.
    output_folder : str
        Folder to store dimerized structures.
    progress_file : str, optional
        JSON file tracking completed entries.
    """
    progress_file = os.path.join(output_folder, progress_file)
    # Load progress
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            finished = set(json.load(f))
    else:
        finished = set()

    parser = PDBParser(QUIET=True)
    io = PDBIO()

    for name in os.listdir(rewrited_pdb_folder):
        # Skip completed entries
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

            # 2-chain case
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

            # 3-chain case
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

            # Mark as finished and save progress immediately
            finished.add(name)
            with open(progress_file, "w") as f:
                json.dump(sorted(finished), f)

            print(f"[DONE] {name}")

        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            print("You can safely restart later.")
            continue

    print("All done!")
                

def select_most_central_pdb(pairs_folder: str):
    """
    Select the most representative (central) dimer structure per pair.

    For each dimer folder:
    - All candidate PDBs are aligned using Cα atoms
    - Pairwise RMSDs are computed
    - The structure with the lowest mean RMSD is selected
    - The selected PDB and confidence file are copied to the root folder
    - All original files are moved to an `all/` subfolder

    Parameters
    ----------
    pairs_folder : str
        Directory containing dimer subfolders.
    """
    
    pairs_path = Path(pairs_folder)
    
    # Iterate over each subfolder
    for dimer_folder in [d for d in pairs_path.iterdir() if d.is_dir()]:
        # Check if already finalized
        pdb_files_root = list(dimer_folder.glob("*.pdb"))
        json_files_root = list(dimer_folder.glob("*_confidences.json"))

        if len(pdb_files_root) == 1 and len(json_files_root) == 1:
            print(f"[SKIP] {dimer_folder.name} already finalized")
            continue

        print(f"[PROCESS] {dimer_folder.name}")

        # Prepare the 'all' folder
        all_folder = dimer_folder / "all"
        all_folder.mkdir(exist_ok=True)
        
        # Move files into the 'all' folder
        for item in dimer_folder.iterdir():
            if item.is_file():
                target_path = all_folder / item.name
                # If the target file exists, delete it first
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
        
        # Extract Cα atoms
        def get_ca_atoms(structure):
            ca_atoms = []
            for model in structure:
                for chain in model:
                    for res in chain:
                        if 'CA' in res:
                            ca_atoms.append(res['CA'])
            return ca_atoms
        
        # Compute RMSD matrix
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
        
        # Copy the best PDB file
        src_file = pdb_files[best_index]
        base_name = src_file.stem
        cleaned_name = "-".join(base_name.split("-")[:2]) + ".pdb"
        tgt_file = dimer_folder / cleaned_name
        shutil.copy(src_file, tgt_file)
        
        # Copy the corresponding confidence file
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

if __name__ == "__main__":
    
    pass