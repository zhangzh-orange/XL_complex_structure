#! python3
# -*- encoding: utf-8 -*-
# ===================================
# File    :   complex_assembly_main.py
# Time    :   2025/12/04 17:49:42
# Author  :   Zehong Zhang
# Contact :   zhang@fmp-berlin.de
# ===================================

# TODO: 1. 构建相互作用图 (先complex，再interactor)


# TODO: 2. 子组件预测（先组装已知的complex，再加入interactors）
# Alphafold3依次预测3mer，如报错，则预测2mer
# 检索3mer，如2mer在3mer中重合，则删去该2mer（含在上一文件中）

# TODO：3. 复合物组装
# 3.1 拆分长链？或许不需要rewrite_fd.py
# 3.2 复制结果分配工作目录（A-B-C to A-B, A-C, B-C）copy_preds.py 
# 3.3 重写PDB文件rewrite_af_pdb.py （替换链名为整个complex中的链名）
# 3.4 蒙特卡洛搜索组装mcts.py


# TODO：4. 评分，评估组装复合物的质量score_entire_complex.py

# TODO: 5. 优化空间结构，确保不发生冲突


# 1. 准备前置文件
from complex_assembly.rewrite_af_files import *
from complex_assembly.mcts import main
from preprocess.crosslink_prepare import *
import argparse
import logging

logger = logging.getLogger(__name__)

# 手动写入chains.csv文件
# 手动准备相应fasta文件

# 生成其他文件，network_prepare, 准备为function
# useq

# 准备ucrosslink文件
# useq_df = pd.read_csv(r"N:\08_NK_structure_prediction\data\WASH_complex\assembled_complex\useqs.csv")
# residue_pair_df = pd.read_csv(r"N:\08_NK_structure_prediction\data\WASH_complex\heklopit_pl3017_frd1ppi_sc151_fdr1rp_WASH_cleaned.csv")
# ucrosslinks = crosslink_prepare(useq_df, residue_pair_df)
# ucrosslinks.to_csv(r"N:\08_NK_structure_prediction\data\WASH_complex\assembled_complex\ucrosslinks.csv",index=False)


# 重写pdb
# rewrite_af_cif_structure(
#     af_pred_folder=r"N:\08_NK_structure_prediction\data\WASH_complex\afx_pred",
#     chains_df_path=r"N:\08_NK_structure_prediction\data\WASH_complex\assembled_complex\chains.csv",
#     output_folder=r"N:\08_NK_structure_prediction\data\WASH_complex\assembled_complex\rewrited_pdbs"
# )

# 重写confidence文件
# rewrite_af_score_file(
#     af_pred_folder=r"N:\08_NK_structure_prediction\data\WASH_complex\afx_pred",
#     chains_df_path=r"N:\08_NK_structure_prediction\data\WASH_complex\assembled_complex\chains.csv",
#     output_folder=r"N:\08_NK_structure_prediction\data\WASH_complex\assembled_complex\rewrited_pdbs"
# )

# 拆分trimer to dimer
# split_trimer_to_dimers(
#     r"N:\08_NK_structure_prediction\data\WASH_complex\assembled_complex\rewrited_pdbs", 
#     r"N:\08_NK_structure_prediction\data\WASH_complex\assembled_complex\pairs")

# 确定作为子部件装配的二聚体
# select_most_central_pdb(r"N:\08_NK_structure_prediction\data\WASH_complex\assembled_complex\pairs")

# 2. MCT运行
# ====== 可选：调试模式（无命令行时使用） ======
# 如果你希望在调试时手动指定参数，请修改以下路径：
debug_args = argparse.Namespace(
    network=r"N:\08_NK_structure_prediction\data\COPI_complex\assembled_complex\network.csv",
    pairdir=r"N:\08_NK_structure_prediction\data\COPI_complex\assembled_complex\pairs/",
    useqs=r"N:\08_NK_structure_prediction\data\COPI_complex\assembled_complex\useqs.csv",
    ucrosslinks = r"N:\08_NK_structure_prediction\data\COPI_complex\assembled_complex\ucrosslinks.csv",
    outdir=r"N:\08_NK_structure_prediction\data\COPI_complex\assembled_complex\output/",
)

# ====== 如果命令行有参数，则使用命令行参数 ======
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
    # 若命令行没有给参数，则 fallback 到 debug_args
    if all(v is None for v in vars(cmd_args).values()):
        logger.info("No command-line arguments detected — using debug arguments.")
        main(debug_args)
    else:
        # 使用命令行参数
        main(cmd_args)
except:
    # parse_args 出现异常则使用 debug 参数
    logger.info("Argument parsing failed — using debug arguments.")
    main(debug_args)
