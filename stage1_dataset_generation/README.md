# Stage 1 - Synthetic Multimessenger Dataset Generation

Run the generator to produce per-messenger CSVs and pairwise features used by Stage 2.

Requirements:
pip install numpy pandas matplotlib tqdm

Quick test run:
python generate_multimessenger_dataset_full.py --test_mode

Full run (ensure disk space):
python generate_multimessenger_dataset_full.py --n_events_per_messenger 300000 --n_true_sources 50000
