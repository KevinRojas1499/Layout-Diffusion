import os
import json
import click
import torch
from tqdm import tqdm
from multimodal_interpolant import MultimodalInterpolant
from branching_flows_interpolant import BranchingFlowsInterpolant
from utils.datasets import MultimodalVariableLengthToyDataset
from custom_datasets.multimodal_math import EquationsDataset
from utils.misc import dotdict
from utils.tokenizer import VocabTokenizer
from models.mmdit_qm9 import MMDiTQM9
from visualize_dataset import plot_sample

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
@click.option('--dataset',type=click.Choice(['qm9', 'equations']), default='equations')
@click.option('--interpolant',type=click.Choice(['multimodal', 'branching']), default='multimodal')
@click.option('--num_steps', type=int, default=100)
@click.option('--batch_size', type=int, default=50)
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
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = opts.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting seed={seed}.")

    if opts.dataset == 'qm9':
        character_tokenizer = VocabTokenizer(vocab={'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'})
        euclidean_dim = 3
        hidden_dim = 66
    elif opts.dataset == 'equations':
        character_tokenizer = VocabTokenizer(vocab={'+','-','*', '=', '.'})
        euclidean_dim = 1
        hidden_dim = 384
    print('Vocab')
    print('--------------------------------')
    for token, id in character_tokenizer.atom_to_idx.items():
        print(f'{token}: {id}')
    print('--------------------------------')
    if opts.dataset == 'qm9':
        dataset = MultimodalVariableLengthToyDataset(character_tokenizer)
    elif opts.dataset == 'equations':
        dataset = EquationsDataset(character_tokenizer)

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

    
    model.train()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)//1e6} M")
    
    if not os.path.exists(opts.dir):
        os.makedirs(opts.dir)

    output_samples = {"molecules": []}
    for _ in tqdm(range(num_samples // batch_size + 1), desc="Sampling"):
        samples = interpolant.sampling(model, num_steps, batch_size, dataset.max_length, device, return_trace=opts.return_trace)
        for i, sample in enumerate(samples):
            symbols = character_tokenizer.decode(sample.yt.cpu())
            positions = sample.xt.cpu()[1:len(symbols)+1, :]
            assert len(symbols) == len(positions), 'Symbols and positions have different lengths'
            output_samples["molecules"].append({
                'symbols': list(symbols),
                'positions': positions.tolist()
            })
            if opts.enable_plotting:
                plot_sample(sample.xt.cpu(), sample.yt.cpu(), sample.mask_t.cpu(), os.path.join(opts.dir, f'sample_{i}.png'), character_tokenizer)
            
    b = json.dumps(output_samples, indent=2, separators=(',', ':'), cls=CustomJSONEncoder)
    b = b.replace('"##<', "").replace('>##"', "")
    with open(os.path.join(opts.dir, 'samples.json'), 'w') as f:
        f.write(b)

def load_checkpoint(opts, device, model):
    print(f'Loading checkpoint from {opts.load_checkpoint}')
    snapshot = torch.load(os.path.join(opts.load_checkpoint), weights_only=True, map_location=device)
    model.load_state_dict(snapshot['model'],strict=False)
    return model



def main():
    sampling()

if __name__ == '__main__':
    main()