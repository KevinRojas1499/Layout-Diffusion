import torch, json, numpy as np
from parenthesis_training import (
    ConstraintHead, HeadWrappedModel,
    compute_constraint_coefficients_batch, evaluate_samples
)
from multimodal_interpolant import MultimodalInterpolant
from models.mmdit_qm9 import MMDiTQM9
from utils.tokenizer import VocabTokenizer
from custom_datasets.multimodal_math import ParenthesizedEquationsDataset

device = torch.device('cuda')
character_tokenizer = VocabTokenizer(vocab={'+', '-', '*', '=', '.', '(', ')'})
dataset = ParenthesizedEquationsDataset(character_tokenizer, data_path='data/parenthesis/equations_l5.jsonl')

interpolant = MultimodalInterpolant(
    max_length=dataset.max_length,
    vocab_size=character_tokenizer.vocab_size,
    mask_token=character_tokenizer.mask_token_id,
    pad_token=character_tokenizer.pad_token_id,
    bos_token=character_tokenizer.bos_token_id,
    euclidean_dim=1,
)

def make_model(ckpt_path):
    model = MMDiTQM9(
        branching_flows=False, autoregressive=False, euclidean_dim=1,
        vocab_size=character_tokenizer.vocab_size,
        symbols_depth=8, positions_depth=8, depth=8,
        dim_modalities=[256,256], dim_joint_attn=256, dim_conds=[256,256],
    ).to(device)
    constraint_head = ConstraintHead(hidden=64).to(device)
    snap = torch.load(ckpt_path, weights_only=True, map_location=device)
    model.load_state_dict(snap['ema'])
    if 'ema_head' in snap:
        constraint_head.load_state_dict(snap['ema_head'])
    model.eval()
    constraint_head.eval()
    return model, constraint_head

def eval_checkpoint(ckpt_path, n_steps, n_samples=500):
    model, constraint_head = make_model(ckpt_path)

    print(f'\n--- Backbone only ({n_steps} steps) ---')
    with torch.no_grad():
        s = interpolant.sampling(model, n_steps, n_samples, dataset.max_length, device, return_trace=False)
    m = evaluate_samples(s, character_tokenizer, euclidean_dim=1, delta=0.5)
    all_nums = [abs(s.xt[i,j,0].item()) for i in range(s.xt.shape[0]) for j in range(1, s.xt.shape[1])]
    max_per = [s.xt[i,1:,:].abs().max().item() for i in range(s.xt.shape[0])]
    print(f'  acc={m["accuracy"]:.3f}, MAE={m["mean_abs_error"]:.3f}, valid={m["valid"]}/{m["total"]}, scale avg_max={np.mean(max_per):.3f}')

    print(f'\n--- Per-step head ({n_steps} steps) ---')
    wrapped = HeadWrappedModel(
        model, constraint_head,
        character_tokenizer.idx_to_atom,
        character_tokenizer.pad_token_id,
        character_tokenizer.bos_token_id,
    )
    with torch.no_grad():
        s2 = interpolant.sampling(wrapped, n_steps, n_samples, dataset.max_length, device, return_trace=False)
    m2 = evaluate_samples(s2, character_tokenizer, euclidean_dim=1, delta=0.5)
    all_nums2 = [abs(s2.xt[i,j,0].item()) for i in range(s2.xt.shape[0]) for j in range(1, s2.xt.shape[1])]
    max_per2 = [s2.xt[i,1:,:].abs().max().item() for i in range(s2.xt.shape[0])]
    print(f'  acc={m2["accuracy"]:.3f}, MAE={m2["mean_abs_error"]:.3f}, valid={m2["valid"]}/{m2["total"]}, scale avg_max={np.mean(max_per2):.3f}')

checkpoints = [
    ('itr_4500 (best 50-step)', 'runs/parenthesis/depth8-constraint-head/itr_4500/snapshot.pt'),
    ('itr_5000 (best MAE)',     'runs/parenthesis/depth8-constraint-head/itr_5000/snapshot.pt'),
]

for label, ckpt_path in checkpoints:
    print(f'\n{"="*60}')
    print(f'Checkpoint: {label}')
    print(f'{"="*60}')
    for n_steps in [50, 400]:
        eval_checkpoint(ckpt_path, n_steps)
