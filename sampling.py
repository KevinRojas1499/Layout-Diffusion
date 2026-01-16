import os
import json
import click
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from multimodal_interpolant import MultimodalInterpolant
from custom_datasets.qm9 import QM9Dataset
from utils.misc import dotdict
from utils.tokenizer import VocabTokenizer
from models.mmdit_qm9 import MMDiTQM9

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
@click.option('--batch_size', type=int, default=50)
@click.option('--num_workers',type=int,default=2)
@click.option('--seed',type=int,default=42)
@click.option('--dir',type=str)
@click.option('--return_trace', type=bool, default=False)
@click.option('--load_checkpoint',type=str, help='Directory where we can find the desired checkpoints')
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
        symbols_depth=4,
        positions_depth=4,
        depth=4,
        dim_modalities=[384, 384],
        dim_joint_attn=384,
        dim_conds=[384, 384]
    ).to(device)
    model = load_checkpoint(opts, rank, device, model)
    
    interpolant = MultimodalInterpolant(
        max_length=dataset.max_length,
        vocab_size=character_tokenizer.vocab_size,
        mask_token=character_tokenizer.mask_token_id,
        pad_token=character_tokenizer.pad_token_id, 
        bos_token=character_tokenizer.bos_token_id,
        eos_token=character_tokenizer.eos_token_id,
        euclidean_dim=3,
    )

    dist.barrier(device_ids=[device])
    
    model.train()
    model = DDP(model)
    
    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)//1e6} M")
    
    if not os.path.exists(opts.dir) and rank == 0:
        os.makedirs(opts.dir)

    output_samples = {"molecules": []}
    for _ in tqdm(range(num_samples // batch_size + 1), desc="Sampling"):
        samples = interpolant.euclidean_sampling(model, batch_size, num_steps, dataset.max_length, device, return_trace=opts.return_trace)
        for i, sample in enumerate(samples):
            symbols = character_tokenizer.decode(sample.yt.cpu())
            positions = sample.xt.cpu()[1:len(symbols)+1, :]
            output_samples["molecules"].append({
                'symbols': list(symbols),
                'positions': positions.tolist()
            })

            
    b = json.dumps(output_samples, indent=2, separators=(',', ':'), cls=CustomJSONEncoder)
    b = b.replace('"##<', "").replace('>##"', "")
    with open(os.path.join(opts.dir, 'samples.json'), 'w') as f:
        f.write(b)

    dist.barrier(device_ids=[device])
    dist.destroy_process_group()

def load_checkpoint(opts, rank, device, model):
    print(f'Loading checkpoint from {opts.load_checkpoint} in rank {rank}')
    snapshot = torch.load(os.path.join(opts.load_checkpoint), weights_only=True, map_location=f'cuda:{device}')
    model.load_state_dict(snapshot['model'],strict=False)
    return model


if __name__ == '__main__':
    sampling()
