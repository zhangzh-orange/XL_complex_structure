import re
import json
import random
from itertools import combinations
from pathlib import Path

import pandas as pd
from Bio import SeqIO


# ---------------------------------------------------------------
# Utility: peptide K-position lookup
# ---------------------------------------------------------------

def find_bracket_k_position(peptide: str, protein: str):
    """
    Locate a bracketed lysine ``[K]`` in *peptide* on the full *protein*
    sequence and return its 1-based position.

    Parameters
    ----------
    peptide : str
        Peptide string containing exactly one ``[K]`` marker.
    protein : str
        Full protein sequence.

    Returns
    -------
    int or None
        1-based residue position, or ``None`` if not found.
    """
    pep_no_space = peptide.replace(" ", "")
    m = re.search(r"\[([A-Z])\]", pep_no_space)
    if not m or m.group(1) != "K":
        return None

    aa_index_in_pep = m.start()
    clean_pep = re.sub(r"\[([A-Z])\]", r"\1", pep_no_space)

    start = protein.find(clean_pep)
    if start == -1:
        return None

    return start + aa_index_in_pep + 1


# ---------------------------------------------------------------
# Utility: unique crosslink selection
# ---------------------------------------------------------------

def select_unique_crosslinks(crosslinks):
    """
    Randomly select crosslinks such that each residue endpoint appears
    at most once.

    Parameters
    ----------
    crosslinks : list
        List of ``[[tag1, pos1], [tag2, pos2]]`` residue-pair entries.

    Returns
    -------
    list
        Filtered crosslink list.
    """
    shuffled = crosslinks.copy()
    random.shuffle(shuffled)

    chosen = []
    used = set()
    for cl in shuffled:
        p1 = tuple(cl[0])
        p2 = tuple(cl[1])
        if p1 in used or p2 in used:
            continue
        chosen.append(cl)
        used.add(p1)
        used.add(p2)
    return chosen


# ---------------------------------------------------------------
# JSON template helpers
# ---------------------------------------------------------------

def _init_json_template(key, proteins):
    """
    Build a bare AlphaFold3 input JSON template.

    Parameters
    ----------
    key : str
        Job name.
    proteins : dict
        ``{chain_tag: sequence}`` mapping.
    """
    seq_block = [{"protein": {"id": tag, "sequence": seq}} for tag, seq in proteins.items()]
    return {
        "name": key,
        "modelSeeds": [1],
        "sequences": seq_block,
        "dialect": "alphafold3",
        "version": 1,
        "crosslinks": [{"name": "azide-A-DSBSO", "residue_pairs": []}],
    }


def _make_crosslink(tag1, seq1, tag2, seq2, pepA, pepB):
    return [
        [tag1, find_bracket_k_position(pepA, seq1)],
        [tag2, find_bracket_k_position(pepB, seq2)],
    ]


def _handle_interaction(pA, pB, a_genes, b_genes, seq_map, pepA, pepB, tagA, tagB, crosslinks, seen):
    def safe_append(t1, s1, t2, s2, pX, pY):
        cl = _make_crosslink(t1, s1, t2, s2, pX, pY)
        if cl[0][1] is None or cl[1][1] is None:
            return
        key = (tuple(cl[0]), tuple(cl[1]))
        if key not in seen:
            seen.add(key)
            crosslinks.append(cl)

    if pA in a_genes and pB in b_genes:
        safe_append(tagA, seq_map[pA], tagB, seq_map[pB], pepA, pepB)
    elif pB in a_genes and pA in b_genes:
        safe_append(tagA, seq_map[pA], tagB, seq_map[pB], pepB, pepA)


# ---------------------------------------------------------------
# Main API
# ---------------------------------------------------------------

def prepare_multimer_jsons(
    raw_crosslink_csv,
    fasta_file,
    gene_list_excel,
    triplet_csv,
    output_dir,
    sample_times=3,
    trimer_max_len=2500,
):
    """
    Generate AlphaFold3 input JSON files for all dimer/trimer combinations.

    For each triplet in ``triplet_csv``:

    - If total sequence length ≤ ``trimer_max_len``, one trimer JSON is built.
    - Otherwise, the triplet is split into three dimer JSONs.

    ``sample_times`` independent random crosslink sets are produced per complex.

    Parameters
    ----------
    raw_crosslink_csv : str
        Raw XL-MS CSV with columns ``gene_a``, ``gene_b``, ``pepA``, ``pepB``.
    fasta_file : str
        UniProt FASTA file (headers must contain ``|<entry>|``).
    gene_list_excel : str
        Excel file mapping ``Gene`` → ``Entry`` (UniProt accession).
    triplet_csv : str
        CSV with columns ``p1``, ``p2``, ``p3`` listing triplets to predict.
    output_dir : str
        Directory to write output JSON files.
    sample_times : int, optional
        Number of random crosslink samplings per complex (default 3).
    trimer_max_len : int, optional
        Maximum combined sequence length for a trimer (default 2500).
    """
    print("=== XL_MOPLC: preparing multimer JSON inputs ===")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    xl_df = pd.read_csv(raw_crosslink_csv)
    gene_map = pd.read_excel(gene_list_excel).set_index("Gene")["Entry"].to_dict()

    seq_map = {}
    with open(fasta_file) as f:
        for rec in SeqIO.parse(f, "fasta"):
            prot_id = rec.id.split("|")[1]
            seq_map[prot_id] = str(rec.seq)

    prot_df = pd.read_csv(triplet_csv)
    triplets = [tuple(row) for row in prot_df[["p1", "p2", "p3"]].values]

    json_files = {}

    for sample_time in range(sample_times):
        for p1, p2, p3 in triplets:
            seq_local = {p: seq_map[gene_map[p]] for p in (p1, p2, p3)}
            total_len = sum(len(s) for s in seq_local.values())

            if total_len <= trimer_max_len:
                complexes_to_build = [("trimer", (p1, p2, p3))]
            else:
                print(f"Trimer {(p1, p2, p3)} length={total_len} > {trimer_max_len}, splitting to dimers")
                complexes_to_build = [("dimer", pair) for pair in combinations((p1, p2, p3), 2)]

            for complex_type, comp in complexes_to_build:
                tags = ["A", "B", "C"] if complex_type == "trimer" else ["A", "B"]
                key = "_".join(list(comp) + [str(sample_time)])
                print(f"  {key} (len={sum(len(seq_local[p]) for p in comp)})")

                prot_map = {t: seq_local[p] for t, p in zip(tags, comp)}
                json_files[key] = _init_json_template(key, prot_map)
                crosslinks = json_files[key]["crosslinks"][0]["residue_pairs"]
                seen = set()

                for _, row in xl_df.iterrows():
                    if any(
                        pd.isna(v) or str(v).strip() == ""
                        for v in [row.gene_a, row.gene_b, row.pepA, row.pepB]
                    ):
                        continue

                    a_genes = {g.strip() for g in row.gene_a.split(";")}
                    b_genes = {g.strip() for g in row.gene_b.split(";")}

                    if complex_type == "dimer":
                        pA, pB = comp
                        _handle_interaction(
                            pA, pB, a_genes, b_genes,
                            {p: seq_local[p] for p in comp},
                            row.pepA, row.pepB,
                            "A", "B", crosslinks, seen,
                        )
                    else:
                        pA, pB, pC = comp
                        for tA, tB, qA, qB in [("A", "B", pA, pB), ("A", "C", pA, pC), ("B", "C", pB, pC)]:
                            _handle_interaction(
                                qA, qB, a_genes, b_genes,
                                seq_local, row.pepA, row.pepB,
                                tA, tB, crosslinks, seen,
                            )

                json_files[key]["crosslinks"][0]["residue_pairs"] = select_unique_crosslinks(crosslinks)

    for key, data in json_files.items():
        out = Path(output_dir) / f"{key}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Done: {len(json_files)} JSON files saved to {output_dir}")
