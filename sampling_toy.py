import os
import json
import click
import torch
import torch.distributed as dist
from tqdm import tqdm
from multimodal_interpolant import MultimodalInterpolant
from branching_flows_interpolant import BranchingFlowsInterpolant
from utils.datasets import MultimodalVariableLengthToyDataset
from custom_datasets.multimodal_math import EquationsDataset
from custom_datasets.multimodal_math import ParenthesizedEquationsDataset
from utils.misc import dotdict
from utils.tokenizer import VocabTokenizer
from models.mmdit_qm9 import MMDiTQM9, MMDiTBothVar
from visualize_dataset import plot_sample, plot_sample_2
from multimodal_interpolant_both_var import MultimodalInterpolantBoth
import json
from json import JSONEncoder

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
@click.option('--dataset',type=click.Choice(['qm9', 'equations', 'parenthesis']), default='equations')
@click.option('--data_path',type=str, default=None)
@click.option('--model',type=click.Choice(['MMDiTBothVar', 'DiT']), default='DiT')
@click.option('--interpolant',type=click.Choice(['multimodal', 'branching', 'multimodal_both']), default='multimodal')
@click.option('--sampler',type=click.Choice(['euler', 'split', 'staggered']), default='split')
@click.option('--num_steps', type=int, default=100)
@click.option('--batch_size', type=int, default=100)
@click.option('--num_workers',type=int,default=2)
@click.option('--seed',type=int,default=42)
@click.option('--dir',type=str)
@click.option('--return_trace', type=bool, default=False)
@click.option('--load_checkpoint',type=str, help='Directory where we can find the desired checkpoints')
@click.option('--enable_plotting', is_flag=True, default=False)
def sampling(**opts):
    opts = dotdict(opts)
    batch_size = opts.batch_size
    num_samples = opts.num_samples
    num_steps = opts.num_steps

    # Distributed: init when RANK is set (torchrun); else single GPU
    use_distributed = "RANK" in os.environ
    if use_distributed:
        dist.init_process_group("nccl")

    # Distributed: use torchrun; else single GPU
    if use_distributed and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        seed = opts.seed * world_size + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        if rank == 0:
            print(f"Starting distributed: rank={rank}, world_size={world_size}, seed={seed}.")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seed = opts.seed
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting seed={seed}.")

    if world_size > 1:
        assert batch_size % world_size == 0, "Batch size must be divisible by world size."

    if opts.dataset == 'qm9':
        character_tokenizer = VocabTokenizer(vocab={'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'})
        euclidean_dim = 3
        hidden_dim = 66
    elif opts.dataset == 'equations':
        character_tokenizer = VocabTokenizer(vocab={'+','-','*', '=', '.', '(', ')'})
        euclidean_dim = 1
        hidden_dim = 256
    elif opts.dataset == 'parenthesis':
        character_tokenizer = VocabTokenizer(vocab={'+','-','*', '=', '.', '(', ')'})
        euclidean_dim = 1
        hidden_dim = 256
    print('Vocab')
    print('--------------------------------')
    for token, id in character_tokenizer.atom_to_idx.items():
        print(f'{token}: {id}')
    print('--------------------------------')
    if opts.dataset == 'qm9':
        dataset = MultimodalVariableLengthToyDataset(character_tokenizer)
    elif opts.dataset == 'equations':
        dataset = EquationsDataset(character_tokenizer, max_length=18)
    elif opts.dataset == 'parenthesis':
        dataset = ParenthesizedEquationsDataset(character_tokenizer, data_path=opts.data_path)
    if opts.model == 'MMDiTBothVar':
        model = MMDiTBothVar(
        euclidean_dim=euclidean_dim,
        vocab_size=character_tokenizer.vocab_size,
        symbols_depth=4,
        positions_depth=4,
        depth=4,
        dim_modalities=[hidden_dim, hidden_dim],
        dim_joint_attn=hidden_dim,
        dim_conds=[hidden_dim, hidden_dim]
    ).to(device)
    elif opts.model == 'DiT':
        model = MMDiTQM9(
            euclidean_dim=euclidean_dim,
            vocab_size=character_tokenizer.vocab_size,
            symbols_depth=4,
            positions_depth=4,
            depth=4,
            dim_modalities=[hidden_dim, hidden_dim],
            dim_joint_attn=hidden_dim,
            dim_conds=[hidden_dim, hidden_dim]
        ).to(device)
    model = load_checkpoint(opts, device, model)
    
    if opts.interpolant == 'multimodal':
        interpolant = MultimodalInterpolant(
        max_length=dataset.max_length,
        vocab_size=character_tokenizer.vocab_size,
        mask_token=character_tokenizer.mask_token_id,
        pad_token=character_tokenizer.pad_token_id, 
        bos_token=character_tokenizer.bos_token_id,
        euclidean_dim=euclidean_dim,
    )
    elif opts.interpolant == 'branching':
        interpolant = BranchingFlowsInterpolant(
            mask_token=character_tokenizer.mask_token_id,
        )
    elif opts.interpolant == 'multimodal_both':
        interpolant = MultimodalInterpolantBoth(
            max_length=dataset.max_length,
            vocab_size=character_tokenizer.vocab_size,
            mask_token=character_tokenizer.mask_token_id,
            pad_token=character_tokenizer.pad_token_id, 
            bos_token=character_tokenizer.bos_token_id,
            euclidean_dim=euclidean_dim,
        )
    
    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)//1e6} M")

    if rank == 0 and not os.path.exists(opts.dir):
        os.makedirs(opts.dir)
    if world_size > 1:
        dist.barrier()

    n_per_rank = (num_samples + world_size - 1) // world_size
    local_batch = batch_size // world_size if world_size > 1 else batch_size
    n_iters = (n_per_rank + local_batch - 1) // local_batch

    rank_output = os.path.join(opts.dir, f"_rank_{rank}.jsonl") if world_size > 1 else os.path.join(opts.dir, "samples.jsonl")
    f = open(rank_output, "w")

    for _ in tqdm(range(n_iters), desc="Sampling", disable=(rank != 0)):
        samples = interpolant.sampling(model, num_steps, local_batch, dataset.max_length, device, return_trace=opts.return_trace, sampler=opts.sampler)
        for i, sample in enumerate(samples):
            symbols = character_tokenizer.decode(sample.yt.cpu())
            if opts.interpolant == 'multimodal':
                positions = sample.xt.cpu()[1:len(symbols)+1, :]
                assert len(symbols) == len(positions), 'Symbols and positions have different lengths'
            else:
                x_length = sample.x_mask_t.cpu().sum(dim=-1).tolist()
                y_length = sample.y_mask_t.cpu().sum(dim=-1).tolist()
                positions = sample.xt.cpu()[1:x_length, :]
                if rank == 0:
                    print(f'x_length: {x_length}, y_length: {y_length}')
                    print(sample.xt.cpu())

            if opts.interpolant == 'multimodal':
                assert len(symbols) == len(positions), 'Symbols and positions have different lengths'
            if euclidean_dim == 1:
                numbers = positions.squeeze(-1).tolist()
            else:
                numbers = positions.tolist()
            f.write(json.dumps({
                'numbers': numbers,
                'symbols': list(symbols),
                'length': len(symbols)
            }, separators=(',', ':'), cls=CustomJSONEncoder))
            f.write('\n')
            if opts.enable_plotting and rank == 0:
                if opts.interpolant == 'multimodal':
                    plot_sample(sample.xt.cpu(), sample.yt.cpu(), sample.mask_t.cpu(), os.path.join(opts.dir, f'sample_{i}.png'), character_tokenizer)
                elif opts.interpolant == 'multimodal_both':
                    plot_sample_2(sample.xt.cpu(), sample.yt.cpu(), sample.x_mask_t.cpu(), sample.y_mask_t.cpu(), os.path.join(opts.dir, f'sample_{i}.png'), character_tokenizer)
        f.flush()

    f.close()

    if world_size > 1:
        dist.barrier()
        if rank == 0:
            with open(os.path.join(opts.dir, "samples.jsonl"), "w") as out:
                for r in range(world_size):
                    with open(os.path.join(opts.dir, f"_rank_{r}.jsonl"), "r") as inp:
                        out.write(inp.read())
                    os.remove(os.path.join(opts.dir, f"_rank_{r}.jsonl"))
        dist.barrier()
        dist.destroy_process_group()

def load_checkpoint(opts, device, model):
    print(f'Loading checkpoint from {opts.load_checkpoint}')
    snapshot = torch.load(os.path.join(opts.load_checkpoint), weights_only=True, map_location=device)
    model.load_state_dict(snapshot['model'],strict=False)
    return model



def main():
    sampling()

if __name__ == '__main__':
    main()