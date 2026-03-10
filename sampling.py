import os
import json
import click
import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from multimodal_interpolant import MultimodalInterpolant
from custom_datasets.qm9 import QM9Dataset
from utils.misc import dotdict
from utils.tokenizer import VocabTokenizer
from models.mmdit_qm9 import MMDiTQM9
from visualize_dataset import plot_sample, plot_molecule, plot_molecule_with_mask, compute_axis_limits

import json
from json import JSONEncoder
import re

class MarkedList:
    _list = None
    def __init__(self, l):
        self._list = l

class CustomJSONEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, MarkedList):
            return "##<{}>##".format(o._list)

@click.command()
@click.option('--num_samples', type=int, default=50)
@click.option('--num_steps', type=int, default=100)
@click.option('--sampler', type=click.Choice(['euler', 'split', 'staggered']), default='staggered')
@click.option('--batch_size', type=int, default=50)
@click.option('--num_workers',type=int,default=2)
@click.option('--seed',type=int,default=42)
@click.option('--dir',type=str)
@click.option('--return_trace', type=bool, default=False)
@click.option('--load_checkpoint',type=str, help='Directory where we can find the desired checkpoints')
@click.option('--enable_plotting', is_flag=True, default=False)
@click.option('--use_ema', is_flag=True, default=False)
def sampling(**opts):
    opts = dotdict(opts)
    batch_size = opts.batch_size
    num_samples = opts.num_samples
    num_steps = opts.num_steps
    
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    assert batch_size % world_size == 0, "Batch size must be divisible by world size."
    seed = opts.seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    character_tokenizer = VocabTokenizer(vocab={'H', 'C', 'N', 'O', 'F'})
    print('Vocab')
    print('--------------------------------')
    for token, id in character_tokenizer.atom_to_idx.items():
        print(f'{token}: {id}')
    print('--------------------------------')
    dataset = QM9Dataset(character_tokenizer)
    
    model = MMDiTQM9(
        euclidean_dim=3,
        vocab_size=character_tokenizer.vocab_size,
        symbols_depth=6,
        positions_depth=6,
        depth=6,
        dim_modalities=[384, 384],
        dim_joint_attn=384,
        dim_conds=[384, 384]
    ).to(device)
    model = load_checkpoint(opts, rank, device, model, opts.use_ema)
    
    interpolant = MultimodalInterpolant(
        max_length=dataset.max_length,
        vocab_size=character_tokenizer.vocab_size,
        mask_token=character_tokenizer.mask_token_id,
        pad_token=character_tokenizer.pad_token_id, 
        bos_token=character_tokenizer.bos_token_id,
        euclidean_dim=3,
    )

    dist.barrier(device_ids=[device])
    
    model.eval()
    model = DDP(model)
    
    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)//1e6} M")
    
    if not os.path.exists(opts.dir) and rank == 0:
        os.makedirs(opts.dir)

    output_samples = {"molecules": []}
    for _ in tqdm(range(num_samples // batch_size + 1), desc="Sampling"):
        samples = interpolant.sampling(
            model,
            num_steps,
            batch_size,
            dataset.max_length,
            device,
            return_trace=opts.return_trace,
            sampler=opts.sampler,
        )
        for i, sample in enumerate(samples):
            symbols = character_tokenizer.decode(sample.yt.cpu())
            positions = sample.xt.cpu()[1:len(symbols)+1, :]
            output_samples["molecules"].append({
                'symbols': list(symbols),
                'positions': positions.tolist()
            })

        if opts.enable_plotting:
            for i, sample in enumerate(samples):
                path = os.path.join(opts.dir, f'samples/batch_{_}/')
                os.makedirs(path, exist_ok=True)
                try:
                    plot_sample(sample.xt.cpu(), sample.yt.cpu(), sample.mask_t.cpu(), os.path.join(path, f'sample_{i}.png'), character_tokenizer)
                    symbols = character_tokenizer.decode(sample.yt.cpu())
                    positions = sample.xt.cpu()[1:len(symbols)+1, :]
                    plot_molecule(symbols, positions, os.path.join(path, f'molecule_{i}.png'))

                    os.makedirs(os.path.join(path, f'trajectory_{i}'), exist_ok=True)
                    os.makedirs(os.path.join(path, f'trajectory_sample_{i}'), exist_ok=True)
                    
                    # Compute fixed axis limits from the final molecule for animation consistency
                    final_yt = sample.yt.cpu()
                    final_xt = sample.xt.cpu()
                    # Convert to numpy for axis limit computation
                    if isinstance(final_xt, torch.Tensor):
                        final_xt_np = final_xt.cpu().numpy()
                    else:
                        final_xt_np = np.array(final_xt)
                    axis_limits = compute_axis_limits(final_xt_np, padding=2.0)
                    
                    pbar = tqdm(enumerate(sample.trajectory), leave=False)
                    for j, trajectory in pbar:
                        # Get trajectory data - xt is [L+2, 3], yt is [L+2], mask_t is [L+2]
                        cur_yt = trajectory.yt.cpu()  # Token IDs
                        cur_xt = trajectory.xt.cpu()  # Positions [L+2, 3]
                        cur_mask_t = trajectory.mask_t.cpu()  # Mask [L+2]
                        cur_t = trajectory.t.cpu().item() if hasattr(trajectory.t, 'item') else trajectory.t
                        
                        # Determine which tokens are masked (mask_token_id)
                        mask_token_id = character_tokenizer.mask_token_id
                        is_masked = (cur_yt == mask_token_id).numpy()
                        
                        # Plot molecule with mask visualization and fixed axis limits
                        plot_molecule_with_mask(
                            cur_yt,  # Token IDs
                            cur_xt,  # All positions including BOS/EOS
                            is_masked,  # Which tokens are masked
                            os.path.join(path, f'trajectory_{i}', f'step_{j}.png'),
                            character_tokenizer,
                            t=cur_t,
                            axis_limits=axis_limits  # Fixed limits for animation
                        )
                        plot_sample(trajectory.xt.cpu(), trajectory.yt.cpu(), trajectory.mask_t.cpu(), os.path.join(path, f'trajectory_sample_{i}', f'step_{j}.png'), character_tokenizer)
                        pbar.set_description(f'Saving trajectory {i} step {j}')
                except Exception as e:
                    print(f'Error plotting sample {i}')
            
    b = json.dumps(output_samples, indent=2, separators=(',', ':'), cls=CustomJSONEncoder)
    b = b.replace('"##<', "").replace('>##"', "")
    with open(os.path.join(opts.dir, 'samples.json'), 'w') as f:
        f.write(b)

    dist.barrier(device_ids=[device])
    dist.destroy_process_group()

def load_checkpoint(opts, rank, device, model, use_ema):
    print(f'Loading checkpoint from {opts.load_checkpoint} in rank {rank}')
    snapshot = torch.load(os.path.join(opts.load_checkpoint), weights_only=True, map_location=f'cuda:{device}')
    model_key = 'ema' if use_ema else 'model'
    print(f'Loading {model_key} from checkpoint')
    model.load_state_dict(snapshot[model_key],strict=True)
    return model


if __name__ == '__main__':
    sampling()
