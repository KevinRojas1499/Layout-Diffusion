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