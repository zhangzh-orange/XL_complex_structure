import pandas as pd


def crosslink_prepare(useq_df: pd.DataFrame, residue_pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw crosslink residue pairs onto chain/residue identifiers.

    Parameters
    ----------
    useq_df : pd.DataFrame
        Table with columns ``Useq`` (gene name) and ``Chain`` (single-letter chain ID).
    residue_pair_df : pd.DataFrame
        Raw XL-MS table; must contain columns ``gene_a``, ``gene_b``,
        ``Alpha protein(s) position(s)``, and ``Beta protein(s) position(s)``.

    Returns
    -------
    pd.DataFrame
        Deduplicated crosslink table with columns
        ``ChainA``, ``ResidueA``, ``ChainB``, ``ResidueB``.
    """
    gene_chain_map = useq_df.set_index("Useq")["Chain"].to_dict()

    crosslinks_set = set()

    for _, row in residue_pair_df.iterrows():
        try:
            a_CA = gene_chain_map.get(row["gene_a"])
            a_AA = int(row["Alpha protein(s) position(s)"])

            b_CA = gene_chain_map.get(row["gene_b"])
            b_AA = int(row["Beta protein(s) position(s)"])

            if not a_CA or not b_CA:
                continue
        except Exception:
            continue

        link = tuple(sorted([(a_CA, a_AA), (b_CA, b_AA)]))
        crosslinks_set.add(link)

    unique_crosslink_df = pd.DataFrame(
        [[a_CA, a_AA, b_CA, b_AA] for (a_CA, a_AA), (b_CA, b_AA) in crosslinks_set],
        columns=["ChainA", "ResidueA", "ChainB", "ResidueB"],
    )

    return unique_crosslink_df
