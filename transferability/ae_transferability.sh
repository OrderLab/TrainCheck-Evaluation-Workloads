source ~/miniconda3/etc/profile.d/conda.sh

conda activate traincheck-test
pip install -r requirements.txt

# trace collection
traincheck-collect --use-config --config gcn/md-config-var.yml --output-dir trace_gcn-torch222

conda activate traincheck-torch251
traincheck-collect --use-config --config gcn/md-config-var.yml --output-dir trace_gcn-torch251

# invariant inference
traincheck-infer -f trace_gcn-torch222 -o invariants-gcn-torch222.json

# invariant checking
traincheck-check -f trace_gcn-torch251 -i invariants-gcn-torch222.json -o checker_gcn-torch251

# compute fp rate
python3 compute_inv_applied_rate.py

