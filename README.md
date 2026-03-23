# XL_MOPLC

**XL-MS-guided Monte Carlo Tree Search for Protein Complex Assembly**

XL_MOPLC is a computational pipeline that integrates crosslinking mass spectrometry (XL-MS) restraints with AlphaFold3 structural predictions to assemble large multi-subunit protein complexes. A Monte Carlo Tree Search (MCTS) algorithm explores the assembly space guided by both structural compatibility and crosslink satisfaction, producing ranked atomic models of the full complex.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Input File Formats](#input-file-formats)
- [Directory Structure](#directory-structure)
- [Scoring Function](#scoring-function)
- [Command-Line Usage](#command-line-usage)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

Determining the architecture of large protein complexes remains a central challenge in structural biology. XL_MOPLC addresses this by combining two complementary data sources:

1. **XL-MS data** — crosslinks between lysine residues provide sparse but experimentally validated distance restraints (Cα–Cα ≤ crosslinker length).
2. **AlphaFold3 dimer/trimer predictions** — high-confidence local structural models serve as rigid building blocks.

The pipeline proceeds in two stages:

### Stage 1 — Sub-complex Structure Prediction
- A protein–protein interaction (PPI) network is built from XL-MS interaction data.
- Trimers (connected subgraphs with ≥ 2 edges) and dimers are enumerated.
- AlphaFold3 / AF3X is run on each sub-complex with crosslink constraints embedded in the input JSON.
- Redundant models are reduced by Cα RMSD-based consensus selection.

### Stage 2 — Higher-Order Assembly via MCTS
- The PPI network is encoded as a graph (nodes = chains, edges = experimentally supported interactions).
- MCTS explores all possible assembly orderings from multiple root chains to avoid starting-point bias.
- At each step, a new chain is docked onto the growing assembly by superposing the shared chain from the corresponding pre-computed dimer.
- Assemblies with steric clashes are discarded (> 50% of Cα atoms within 5 Å of existing atoms).
- The final model is selected by a composite score combining structural quality and XL-MS satisfaction.

---

## Pipeline

```
Raw XL-MS CSV + FASTA + chains.csv
         │
         ▼
[Step 1]  Build PPI network, clean identifiers
         │
         ▼
[Step 2]  Enumerate dimers & trimers → binary_pairs.csv, triplets.csv
         │
         ▼
[Step 3]  Prepare AF3 JSON inputs (with XL constraints)
         │
         ▼
      ⚙️  Run AlphaFold3 / AF3X  (external)
         │
         ▼
[Step 4]  Build network.csv and useqs.csv
         │
         ▼
[Step 5]  Map crosslinks → ucrosslinks.csv
         │
         ▼
[Step 6]  Rewrite CIF → PDB (chain renaming, HETATM removal)
         │
         ▼
[Step 7]  Rewrite confidence JSON (chain renaming, matrix trimming)
         │
         ▼
[Step 8]  Split trimer predictions → dimer pairs
         │
         ▼
[Step 9]  Select representative dimer per pair (RMSD consensus)
         │
         ▼
[Step 10] MCTS complex assembly
         │
         ▼
     output/*.pdb  (ranked assembled complexes)
```

The interactive notebook **`src/XL_MOPLC_pipeline.ipynb`** walks through every step with documentation and configurable parameters.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/XL_MOPLC.git
cd XL_MOPLC
```

### 2. Create a Python environment

Python ≥ 3.10 is required.

```bash
conda create -n xl_moplc python=3.11
conda activate xl_moplc
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install AlphaFold3 / AF3X (for sub-complex prediction)

- **AlphaFold3**: https://github.com/google-deepmind/alphafold3
- **AF3X** (recommended for batch runs): https://github.com/KosinskiLab/af3x

These are external tools and must be installed separately in their own environments.

---

## Quick Start

Open the pipeline notebook and follow the steps interactively:

```bash
cd src
jupyter notebook XL_MOPLC_pipeline.ipynb
```

Alternatively, run the MCTS assembly step from the command line after completing the pre-processing steps:

```bash
python src/complex_assembly_main.py \
    --network      data/assembled_complex/network.csv \
    --pairdir      data/assembled_complex/pairs/ \
    --useqs        data/assembled_complex/useqs.csv \
    --ucrosslinks  data/assembled_complex/ucrosslinks.csv \
    --outdir       data/assembled_complex/output/
```

---

## Input File Formats

### `chains.csv` — Gene to chain mapping

Manually prepared. Maps each protein subunit to a unique single-letter chain ID used throughout the pipeline.

| Entry  | Gene  | Chain |
|--------|-------|-------|
| Q9NV70 | EXOC1 | A     |
| Q96KP1 | EXOC2 | B     |
| O15143 | ARPC1B | C   |

- `Entry`: UniProt accession
- `Gene`: gene name (must match FASTA `GN=` fields and XL-MS table)
- `Chain`: unique uppercase letter (A–Z)

---

### FASTA file — Protein sequences

Standard UniProt format. Each header **must** contain a `GN=` field:

```
>sp|Q9NV70|EXOC1_HUMAN Exocyst complex component 1 OS=Homo sapiens OX=9606 GN=EXOC1 PE=1 SV=2
MMSIKAFTLVSAVERELLMGDKERVNIECVECCGRDLYVGTNDCFVYHFLLEERPVPAGP...
```

The gene name after `GN=` must exactly match the `Gene` column in `chains.csv`.

---

### XL-MS CSV — Raw crosslink data

Required columns:

| Column | Description |
|--------|-------------|
| `gene_a` | Gene name of protein A (may contain semicolons for ambiguous identifications) |
| `gene_b` | Gene name of protein B |
| `pepA` | Peptide sequence of protein A with crosslinked residue in brackets, e.g. `QRAVEAQ[K]LMK` |
| `pepB` | Peptide sequence of protein B with crosslinked residue in brackets |
| `Alpha protein(s) position(s)` | Sequence position of the crosslinked residue in protein A |
| `Beta protein(s) position(s)` | Sequence position of the crosslinked residue in protein B |

Example:

| gene_a | gene_b | pepA | pepB | Alpha protein(s) position(s) | Beta protein(s) position(s) |
|--------|--------|------|------|-------------------------------|------------------------------|
| EXOC1  | EXOC2  | `QRAVEAQ[K]LMK` | `VVSN[K]LEEK` | 132 | 47 |

---

### `gene_list.xlsx` — Gene to UniProt entry mapping

Required for AF3 JSON preparation. Maps gene names to UniProt accessions used to look up sequences in the FASTA file.

| Entry  | Gene  | Anno            |
|--------|-------|-----------------|
| Q9NV70 | EXOC1 | Exocyst complex |
| Q96KP1 | EXOC2 | Exocyst complex |

---

### AF3 JSON output format

The `prepare_multimer_jsons()` function produces AlphaFold3-compatible input files:

```json
{
    "name": "EXOC1_EXOC2_EXOC3_0",
    "modelSeeds": [1],
    "sequences": [
        {"protein": {"id": "A", "sequence": "MTAIKHAL..."}},
        {"protein": {"id": "B", "sequence": "MSRSRQPP..."}},
        {"protein": {"id": "C", "sequence": "MKETDREA..."}}
    ],
    "dialect": "alphafold3",
    "version": 1,
    "crosslinks": [
        {
            "name": "azide-A-DSBSO",
            "residue_pairs": [
                [["A", 60], ["C", 28]],
                [["A",  5], ["B", 310]]
            ]
        }
    ]
}
```

Three JSON files are generated per sub-complex (different random crosslink samplings), named `<proteins>_0.json`, `<proteins>_1.json`, `<proteins>_2.json`.

---

## Directory Structure

```
XL_MOPLC/
├── src/
│   ├── XL_MOPLC_pipeline.ipynb       # Main interactive pipeline notebook
│   ├── complex_assembly_main.py       # CLI entry point for MCTS assembly
│   ├── complex_assembly/
│   │   ├── mcts.py                    # MCTS algorithm and scoring
│   │   └── rewrite_af_files.py        # AF structure/score post-processing
│   ├── preprocess/
│   │   ├── af_json_prepare.py         # AF3 JSON input generation
│   │   ├── crosslink_prepare.py       # XL-MS → ucrosslinks.csv
│   │   └── network_prepare.py         # network.csv + useqs.csv
│   └── network/
│       └── interact_map.py            # PPI network construction and analysis
├── data/                              # User data (not tracked in git)
│   └── <complex_name>/
│       ├── chains.csv
│       ├── <complex>.fasta
│       ├── <xl_ms_data>.csv
│       ├── gene_list.xlsx
│       ├── jsons/                     # AF3 JSON inputs
│       ├── afx_pred/                  # AF3/AF3X prediction outputs
│       └── assembled_complex/
│           ├── network.csv
│           ├── useqs.csv
│           ├── ucrosslinks.csv
│           ├── rewrited_pdbs/
│           ├── pairs/
│           └── output/
├── requirements.txt
└── README.md
```

---

## Scoring Function

Each partial assembly during MCTS is evaluated by a composite score:

$$\text{Score}_\text{final} = \text{Score}_\text{struct} \times \text{Score}_\text{XL}$$

**Structural score** — rewards large, high-confidence interfaces:

$$\text{Score}_\text{struct} = \text{Interface area} \times \overline{\text{pLDDT}}_\text{interface}$$

Interface residues are defined as those with any heavy atom within 8 Å of the partner chain. The mean pLDDT is taken over all interface residues from the original AlphaFold prediction.

**Crosslink score** — fraction of XL-MS restraints satisfied:

$$\text{Score}_\text{XL} = \frac{| \{(i,j) : d_{ij}^{C\alpha} \leq L_\text{crosslinker}\} |}{N_\text{crosslinks}}$$

where $L_\text{crosslinker}$ is the maximum Cα–Cα distance allowed by the crosslinker (default 45 Å for DSBSO).

**MCTS node selection** uses the UCB1 formula:

$$\text{UCB}_i = \bar{V}_i + \sqrt{\frac{\ln N}{n_i}}$$

where $\bar{V}_i$ is the mean score of simulations through node $i$, $N$ is the total visit count, and $n_i$ is the visit count for node $i$.

---

## Command-Line Usage

The MCTS assembly step can be run independently after completing the pre-processing steps (Steps 4–9 in the notebook):

```bash
python src/complex_assembly_main.py \
    --network      PATH/TO/network.csv       \
    --pairdir      PATH/TO/pairs/            \
    --useqs        PATH/TO/useqs.csv         \
    --ucrosslinks  PATH/TO/ucrosslinks.csv   \
    --outdir       PATH/TO/output/
```

All arguments are required. For interactive use with step-by-step control, use the Jupyter notebook.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.10 | Core runtime |
| Biopython | ≥ 1.81 | Structure parsing, superposition |
| NumPy | ≥ 1.24 | Numerical operations |
| pandas | ≥ 2.0 | Data handling |
| networkx | ≥ 3.0 | PPI network construction |
| matplotlib | ≥ 3.7 | Network visualization |
| openpyxl | ≥ 3.1 | Reading gene list Excel files |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use XL_MOPLC in your research, please cite:

> Zhang, Z. et al. *XL_MOPLC: XL-MS-guided Monte Carlo Tree Search for Protein Complex Assembly.* (2025). Manuscript in preparation.

---

## Contact

**Zehong Zhang**
Leibniz-Forschungsinstitut für Molekulare Pharmakologie (FMP), Berlin
zhang@fmp-berlin.de

Please report bugs and feature requests via [GitHub Issues](../../issues).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
