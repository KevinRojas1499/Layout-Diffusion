## This is a readme


## Installing open babel

In my case I needed to create a symlink with the openbabel installation.
```bash
sudo pacman -S swig openbabel
uv add openbabel


# Create a symlink so the build finds headers in the expected location
sudo mkdir -p /usr/local/include
sudo ln -s /usr/include/openbabel3 /usr/local/include/openbabel3

# Then install
uv add openbabel
```

For vast ai server using apt
```bash
sudo apt install openbabel openbabel-gui
sudo apt install libopenbabel-dev
sudo apt install python3-openbabel
sudo apt install swig
sudo mkdir -p /usr/local/include
sudo ln -s /usr/include/openbabel3 /usr/local/include/openbabel3
uv sync
```


## Preprocess the dataset

```{bash}
PYTHONPATH=. uv run python -m custom_datasets.preprocess_qm9 --output_dir data/qm9_preprocessed
```

### Math dataset

```{bash}
python custom_datasets/create_dataset.py -L 10 --decimals 2 \
  --min-terms 1 --max-terms 8 \
  --length-dist "1:0.05,2:0.1,3:0.2,4:0.25,5:0.2,6:0.1,7:0.07,8:0.03"
```

### Parenthesis Creation

python custom_datasets/create_parenthesis_dataset.py -L 10 --decimals 2 \
  --min-terms 1 --max-terms 6 \
  --length-dist "1:0.05,2:0.1,3:0.2,4:0.25,5:0.2,6:0.2"

### Parenthesis training

uv run torchrun --master-port 29502 toy_training.py --interpolant multimodal_both --dataset parenthesis --dir experiments/parenthesis  --data_path data/parenthesis/equations_l5.jsonl --model MMDiTBothVar --log_rate 2500

### Parenthesis sampling

uv run torchrun sampling_toy.py --num_samples 10000 --dataset parenthesis --data_path data/parenthesis/equations_l10.jsonl --interpolant multimodal_both --num_steps 1000 --dir samples-parenthesis/l10 --load_checkpoint experiments/parenthesis_l10/itr_100000/snapshot.pt --model MMDiTBothVar

### Samplers Grid Search Figure 3
uv run python run_equations_grid_search.py --steps-min 50 --steps-max 500 --steps-num 5 --nfe-min 50 --nfe-max 1500 --nfe-num 5 --num-samples 10000 --seed 1 --seed 2 --seed 3


## QM9 Distribution Evaluation

Evaluate generated molecules at different sample sizes (1k, 2.5k, 5k, 10k, etc.) to see how KS statistics change:

```bash
uv run python grid_search.py eval-sample-sizes --generated samples/muon-fused-residual-80k/samples.json
```

### Changing the output folder

Use `--comparison-output-template` to control where results are written. Use `{n_gen}` for the sample size:

```bash
uv run python grid_search.py eval-sample-sizes \
  --generated samples/muon-fused-residual-80k/samples.json \
  --comparison-output-template "results-diff-samples/muon-fused-80k-eval-{n_gen}"
```

This writes to `results-diff-samples/muon-fused-80k-eval-1000/`, `results-diff-samples/muon-fused-80k-eval-2500/`, etc. Default is `results/eval-n-gen-{n_gen}`.

### Stability analysis (multiple subsamples per size)

To assess metric variance, run multiple independent subsamples of the same size. Each repeat uses a different random subset:

```bash
uv run python eval/eval_ks_batch.py \
  --generated samples/muon-fused-residual-80k/samples.json \
  --n-gen-samples 2500 \
  --n-repeats 10 \
  --comparison-output-template "results/stability-2500"
```

This creates `results/stability-2500/repeat-0/`, `repeat-1/`, ... `repeat-9/`, each with KS stats from a different random 2500-sample subset. Compare the metrics across repeats to study statistical stability.

### Direct batch script

```bash
uv run python eval/eval_ks_batch.py \
  --generated samples/muon-fused-residual-80k/samples.json \
  --n-gen-samples 1000 2500 5000 10000 20000 \
  --comparison-output-template "my_results/eval-{n_gen}"
```

### Evaluate a folder

uv run eval/eval_ks_batch.py --folder qm9_sampling_grid/ 
