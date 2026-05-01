"""Quick 400-step eval on a standard (no constraint head) checkpoint."""
import sys, torch, numpy as np
from parenthesis_training import evaluate_samples
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

def eval_ckpt(ckpt_path, n_steps=400, n_samples=500, label=''):
    model = MMDiTQM9(
        branching_flows=False, autoregressive=False, euclidean_dim=1,
        vocab_size=character_tokenizer.vocab_size,
        symbols_depth=8, positions_depth=8, depth=8,
        dim_modalities=[256,256], dim_joint_attn=256, dim_conds=[256,256],
    ).to(device)
    snap = torch.load(ckpt_path, weights_only=True, map_location=device)
    model.load_state_dict(snap['ema'])
    model.eval()

    with torch.no_grad():
        s = interpolant.sampling(model, n_steps, n_samples, dataset.max_length, device, return_trace=False)
    m = evaluate_samples(s, character_tokenizer, euclidean_dim=1, delta=0.5)
    max_per = [s.xt[i,1:,:].abs().max().item() for i in range(s.xt.shape[0])]
    print(f'{label}: acc={m["accuracy"]:.3f}, MAE={m["mean_abs_error"]:.3f}, valid={m["valid"]}/{m["total"]}, avg_max_abs={np.mean(max_per):.3f}')
    return m

# Eval requested checkpoint(s) from command line or defaults
ckpts = sys.argv[1:] if len(sys.argv) > 1 else [
    'runs/parenthesis/depth8-scale-hinge-sw005/itr_4500/snapshot.pt',
    'runs/parenthesis/depth8-scale-hinge-sw005/itr_5000/snapshot.pt',
    'runs/parenthesis/depth8-scale-hinge-sw005/itr_500/snapshot.pt',
]

for ckpt in ckpts:
    label = ckpt.split('/')[-2] if '/' in ckpt else ckpt
    eval_ckpt(ckpt, n_steps=400, label=label)
