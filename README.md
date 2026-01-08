# XL_complex_structure

Crosslink-guided protein complex structure prediction and assembly

## Project Overview

This project implements a pipeline for predicting and assembling protein complex structures by integrating crosslinking mass spectrometry (XL–MS) data with AlphaFold3 multimer modeling.

The workflow consists of two main stages:

1. Subcomponent (dimer/trimer) structure prediction:
- Protein dimers and trimers are identified from XL–MS interaction networks.

- Each protein chain is represented as a node, and crosslinks as edges.

- Local trimeric units are extracted as connected subgraphs with at least two edges.

- Crosslink constraints are randomly filtered so each residue participates in at most one crosslink per model, repeated three times to generate multiple random inputs.

- If the combined sequence length of a trimer exceeds 2,700 residues, the triplet is decomposed into all possible dimeric subcomplexes to respect AlphaFold3 input size constraints.

- Predicted structures are post-processed: trimers are decomposed into dimers, confidence metrics (pLDDT, PAE, contact probabilities) are filtered, and redundant models are reduced by consensus RMSD-based selection.
2. Higher-order complex assembly using Monte Carlo Tree Search (MCTS):
- The interaction network is represented as a graph (nodes = chains, edges = experimentally supported interactions).

- MCTS explores assembly pathways from multiple root chains to avoid bias.

- At each step, candidate chains are aligned using predicted dimer structures; steric clashes are checked.

- A composite scoring function evaluates assemblies:
  
  - Structural interface score (Score_struct) based on interface size and pLDDT.
  
  - Crosslink satisfaction score (Score_XL) based on XL–MS restraints.
  
  - Final score (Score_final) = Score_struct * Score_XL.

- Rollouts are backpropagated to optimize assembly pathways.

-----

## Directory Structure

```
XL_complex_structure/
├── data/
│   ├── COPI_complex/
│   │   ├── COPI.fasta                # Protein sequences
│   │   ├── COPI_gene_list.xlsx       # Gene list with annotations
│   │   ├── heklopit_pl3017_frd1ppi_sc151_fdr1rp_COPI.csv  # XL–MS raw data
│   │   └── trimer.csv                 # Trimer/dimer combinations (from interact_map)
├── jsons/                             # Output JSONs for AF3X predictions
├── src/
│   ├── complex_assembly_pipeline.ipynb
│   └── af_json_prepare.py             # Functions for preparing AF3X input
├── README.md
└── requirements.txt                   # Python dependencies
```

-----

## Workflow

1. Prepare AlphaFold3X Input

```python
from src.af_json_prepare import prepare_multimer_jsons

prepare_multimer_jsons(
    raw_crosslink_csv=r"data/COPI_complex/heklopit_pl3017_frd1ppi_sc151_fdr1rp_COPI.csv",
    fasta_file=r"data/COPI_complex/COPI.fasta",
    gene_list_excel=r"data/COPI_complex/COPI_gene_list.xlsx",
    triplet_csv=r"data/COPI_complex/trimer.csv",
    output_dir=r"data/COPI_complex/jsons",
)
```

- Generates JSON input for AF3X predictions.

- For residues with multiple crosslinks, only one is randomly retained.

- Three random crosslink sets are generated per trimer/dimer.

File formats:

- `gene_list.xlsx`:

| Entry  | Gene  | Anno            |
| ------ | ----- | --------------- |
| Q9NV70 | EXOC1 | Exocyst complex |
| Q96KP1 | EXOC2 | Exocyst complex |
| ...    | ...   | ...             |

- `raw_crosslink_csv`:

| protein_a | protein_b | gene_a | gene_b | Alpha protein(s) position(s) | Beta protein(s) position(s) | pepA          | pepB          | Score    | Software |
| --------- | --------- | ------ | ------ | ---------------------------- | --------------------------- | ------------- | ------------- | -------- | -------- |
| O00471    | O00471    | EXOC5  | EXOC5  | 132                          | 132                         | QRAVEAQ[K]LMK | QRAVEAQ[K]LMK | 0.998358 | Plink3   |
| ...       | ...       | ...    | ...    | ...                          | ...                         | ...           | ...           | ...      | ...      |

- `trimer.csv`: Generated from interact_map (see complex_assembly_pipeline.ipynb).
2. Run AlphaFold3 / AF3X Predictions
- Installation and configuration instructions:
  
  - AlphaFold3: https://github.com/google-deepmind/alphafold3
  
  - AF3X: https://github.com/KosinskiLab/af3x

- Predicted multimer structures will be stored for downstream assembly.
3. Post-processing and Dimer Extraction
- Trimers are decomposed into all possible dimers.

- Confidence metrics are filtered to relevant chains.

- Redundant models are removed using RMSD-based consensus selection.
4. Complex Assembly (MCTS)
- Partial assemblies are encoded as nodes with chain order and 3D coordinates.

- MCTS explores possible assembly paths using the UCB score:

- UCBi=Vi+ln⁡Nni

- Expansion and rollouts incorporate predicted dimers and crosslink restraints.

- Steric clashes are filtered (50% of Cα atoms within 5Å → discard).

- Final assemblies are scored with:

- Scorefinal=Scorestruct×ScoreXL

- The highest-scoring complex across all root chains is selected as the final model.

## Required Dependencies

Python ≥3.10

PyTorch (GPU)

Biopython

NumPy, pandas, SciPy

Alphafold3 / AF3X environment

(Full requirements will be listed in requirements.txt)

## To-do / Sections for Future Completion

Detailed example workflow

Include step-by-step commands for COPI complex from raw CSV → final complex PDB.

Visualization scripts

Render final complexes, interfaces, and crosslink satisfaction.

Benchmarking / validation

Compare predicted structures with experimental data (if available).

Automated pipeline execution

Integrate all steps in a single run_pipeline.py or Jupyter notebook for reproducibility.

Expanded documentation

Explain scoring functions and MCTS algorithm with diagrams.

Include parameter tuning options for crosslink filtering and MCTS exploration.
