#! python3
# -*- encoding: utf-8 -*-
# ===================================
# File    :   complex_assembly_main.py
# Time    :   2025/12/04 17:49:42
# Author  :   Zehong Zhang
# Contact :   zhang@fmp-berlin.de
# ===================================

# TODO: 1. 构建相互作用图

# TODO: 2. 子组件预测
# Alphafold3依次预测3mer，如报错，则预测2mer
# 检索3mer，如2mer在3mer中重合，则删去该2mer（含在上一文件中）

# TODO：3. 复合物组装
# 3.1 拆分长链？或许不需要rewrite_fd.py
# 3.2 复制结果分配工作目录（A-B-C to A-B, A-C, B-C）copy_preds.py 
# 3.3 重写PDB文件rewrite_af_pdb.py （替换链名为整个complex中的链名）
# 3.4 蒙特卡洛搜索组装mcts.py


# TODO：4. 评分，评估组装复合物的质量score_entire_complex.py

# TODO: 5. 优化空间结构，确保不发生冲突

