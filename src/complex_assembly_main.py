#! python3
# -*- encoding: utf-8 -*-
# ===================================
# File    :   complex_assembly_main.py
# Author  :   Zehong Zhang
# Contact :   zhang@fmp-berlin.de
# ===================================
"""
XL_MOPLC — command-line entry point for MCTS-based complex assembly.

Usage
-----
    python complex_assembly_main.py \\
        --network          <path/to/network.csv>        \\
        --pairdir          <path/to/pairs/>              \\
        --useqs            <path/to/useqs.csv>           \\
        --ucrosslinks      <path/to/ucrosslinks.csv>     \\
        --outdir           <path/to/output/>             \\
        [--crosslinker_length 35]

All positional arguments are required.  For interactive use, run the
Jupyter notebook ``XL_MOPLC_pipeline.ipynb`` instead.
"""

import argparse
import logging

from complex_assembly.mcts import main

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the MCTS assembly pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed arguments ready to pass to ``mcts.main()``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "XL_MOPLC: MCTS-based protein complex assembly "
            "guided by XL-MS crosslink restraints."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--network",
        required=True,
        help="Path to network.csv (columns: Chain1, Chain2, Source).",
    )
    parser.add_argument(
        "--pairdir",
        required=True,
        help="Directory containing dimer pair sub-folders (one per chain pair).",
    )
    parser.add_argument(
        "--useqs",
        required=True,
        help="Path to useqs.csv (columns: Chain, Useq, Sequence).",
    )
    parser.add_argument(
        "--ucrosslinks",
        required=True,
        help="Path to ucrosslinks.csv (columns: ChainA, ResidueA, ChainB, ResidueB).",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for assembled PDB and log files.",
    )
    parser.add_argument(
        "--crosslinker_length",
        type=int,
        default=35,
        help=(
            "Maximum allowed Cα–Cα distance (Å) for a crosslink to be "
            "considered satisfied. Adjust to match the crosslinker used "
            "(e.g. 35 for DSSO/DSBSO, 45 for longer spacer arms)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
