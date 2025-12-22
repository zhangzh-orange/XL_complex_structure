import math
import pandas as pd



def cal_crosslink_distance(a_chain_inds:str, 
                       a_AA_inds:int, 
                       b_chain_inds:str, 
                       b_AA_inds:int,
                       crosslinker_length:int=45):
    """Calculate distance of two crosslinked residues

    Parameters
    ----------
    a_chain_inds : str
        第一个位点的所在链单字母代号
    a_AA_inds : int
        第一个位点的氨基酸index
    b_chain_inds : str
        第二个位点的所在链的单字母代号
    b_AA_inds : int
        第二个位点的氨基酸index
    crosslinker_length : int, optional
        Restriction of max length of crosslinker, by default DSBSO with length 45 A

    Returns
    -------
    distance: float
        distance of two crosslinked residues
    if_consist: bool
        if align with special crosslinker length (smaller or equal)
    """
    
    
    a_CA_inds = chain_CA_inds[a_chain_inds][a_AA_inds]
    b_CA_inds = chain_CA_inds[b_chain_inds][b_AA_inds]

    a_CA_coords = chain_coords[a_chain_inds][a_CA_inds]
    b_CA_coords = chain_coords[b_chain_inds][b_CA_inds]

    def cal_euclidean_distance(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    distance = cal_euclidean_distance(a_CA_coords,b_CA_coords)

    consist_with_crosslinker_length = (distance<=crosslinker_length)
    return round(distance, 2), consist_with_crosslinker_length


def crosslink_prepare(useq_df:pd.DataFrame,
                      residue_pair_df:pd.DataFrame):
    """_summary_

    Parameters
    ----------
    useq_df : pd.DataFrame
        _description_
    residue_pair_df : pd.DataFrame
        _description_

    Returns
    -------
    _type_
        _description_
    """
    gene_chain_map = useq_df.set_index("Useq")["Chain"].to_dict()

    crosslinks_set = set()

    for _, row in residue_pair_df.iterrows():
        a_CA = gene_chain_map.get(row["gene_a"])
        a_AA = int(row["Alpha protein(s) position(s)"])
        
        b_CA = gene_chain_map.get(row["gene_b"])
        b_AA = int(row["Beta protein(s) position(s)"])

        # 如果任意一个 CA 为空，则跳过
        if not a_CA or not b_CA:
            continue
        
        # 用 tuple 表示残基对，无序排列，保证 a-b 和 b-a 被视为相同
        link = tuple(sorted([(a_CA, a_AA), (b_CA, b_AA)]))
        
        crosslinks_set.add(link)

    unique_crosslink_df = pd.DataFrame(
        [ [a_CA, a_AA, b_CA, b_AA] for (a_CA, a_AA), (b_CA, b_AA) in crosslinks_set ],
        columns=["ChainA", "ResidueA", "ChainB", "ResidueB"]
    )

    return unique_crosslink_df

if __name__ == "__main__":
    from src.complex_assembly.mcts_old import read_pdb
    pdb_file = r"N:\08_NK_structure_prediction\XL_complex_structure\data\lrba_stx7_0_AB_A-lrba_stx7_0_AB_B.pdb"
    pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pdb_file)
    print(chain_coords["A"][:10])
    print(chain_CA_inds["A"][:10])
    a_CA_inds = chain_CA_inds["A"][2]
    b_CA_inds = chain_CA_inds["A"][10]

    a_CA_coords = chain_coords["A"][a_CA_inds]
    b_CA_coords = chain_coords["A"][b_CA_inds]
    print(a_CA_coords)
    print(b_CA_coords)

    distance = cal_crosslink_distance("A",2,"A",10)
    print(distance)

    # useq_df = pd.read_csv(r"N:\08_NK_structure_prediction\data\LRBAandSNARE\assembled_complex\useqs.csv")
    # residue_pair_df = pd.read_csv(r"N:\08_NK_structure_prediction\data\LRBAandSNARE\heklopit_pl3017_frd1ppi_sc151_fdr1rp_LRBAandSNARE.csv")
    # ucrosslinks = crosslink_prepare(useq_df, residue_pair_df)
    # ucrosslinks.to_csv(r"N:\08_NK_structure_prediction\data\LRBAandSNARE\assembled_complex\ucrosslinks.csv",index=False)

