#!/bin/bash
# Unbiased SIS correction for the frame-normalization coupling (high-SNR tail).
# Run from the repository root; expects data/synthetic_awgn_amc_v1 to exist.
export OMP_NUM_THREADS=8
python configs/snr_ladder/scripts/ceiling/exact_alrt.py --run sis \
    --frames-per-snr 2100 --paths 512 --out work_dirs/sis_correction.csv
