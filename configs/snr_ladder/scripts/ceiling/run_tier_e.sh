#!/bin/bash
# Tier-E exact Bayes ceiling (factorized likelihoods), 2e5 frames per SNR.
# Run from the repository root; expects data/synthetic_awgn_amc_v1 to exist.
export OMP_NUM_THREADS=8
python configs/snr_ladder/scripts/ceiling/exact_alrt.py --run tier_e \
    --frames 200000 --out work_dirs/tier_e_ceiling.csv
