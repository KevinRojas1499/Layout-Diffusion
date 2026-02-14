import os
import click
import torch
import wandb
from collections import OrderedDict
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm
from multimodal_interpolant import MultimodalInterpolant
from branching_flows_interpolant import BranchingFlowsInterpolant
from custom_datasets.multimodal_math import EquationsDataset
from custom_datasets.multimodal_math import ParenthesizedEquationsDataset
from utils.datasets import MultimodalVariableLengthToyDataset
from utils.misc import dotdict
from utils.tokenizer import VocabTokenizer
from utils.optimizers import WarmUpScheduler
from models.mmdit_qm9 import MMDiTQM9, MMDiTBothVar
from visualize_dataset import plot_sample
from multimodal_interpolant_both_var import MultimodalInterpolantBoth


def init_wandb(opts):
    wandb.init(
        # set the wandb project where this run will be logged
        project='MMDiT-Toy',
        name= f'toy-{opts.run_name}',
        tags= ['training'],
        # # track hyperparameters and run metadata
        config=opts,
    )

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

@click.command()
@click.option('--model',type=click.Choice(['radd', 'DiT', 'MMDiTBothVar']), default='DiT')
@click.option('--dataset',type=click.Choice(['qm9', 'equations', 'parenthesis']), default='equations')
@click.option('--data_path',type=str, default=None)
@click.option('--interpolant',type=click.Choice(['multimodal', 'branching', 'multimodal_both']), default='multimodal')
@click.option('--optimizer',type=click.Choice(['adam','adamw']), default='adam')
@click.option('--ema_beta',type=float, default=.9999)
@click.option('--lr', type=float, default=1e-4)
@click.option('--batch_size', type=int, default=128)
@click.option('--log_rate',type=int,default=500)
@click.option('--num_iters',type=int,default=5000)
@click.option('--warmup_iters',type=int,default=100)
@click.option('--num_workers',type=int,default=2)
@click.option('--seed',type=int,default=42)
@click.option('--dir',type=str)
@click.option('--load_checkpoint',type=str, help='Directory where we can find the desired checkpoints')
@click.option('--train_only_dsm', is_flag=True, default=False)
@click.option('--enable_wandb', is_flag=True, default=False)
@click.option('--run_name', type=str, default='')
def training(**opts):
    opts = dotdict(opts)
    seed = opts.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    batch_size = opts.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if opts.dataset == 'qm9':
        character_tokenizer = VocabTokenizer(vocab={'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'})
        dataset = MultimodalVariableLengthToyDataset(character_tokenizer)
        euclidean_dim = 3
        hidden_dim = 66
    elif opts.dataset == 'equations':
        character_tokenizer = VocabTokenizer(vocab={'+','-','*', '=', '.', '(', ')'})
        dataset = EquationsDataset(character_tokenizer, data_path=opts.data_path)
        euclidean_dim = 1
        hidden_dim = 256
    elif opts.dataset == 'parenthesis':
        character_tokenizer = VocabTokenizer(vocab={'+','-','*', '=', '.', '(', ')'})
        dataset = ParenthesizedEquationsDataset(character_tokenizer, data_path=opts.data_path)
        euclidean_dim = 1
        hidden_dim = 256
    print('Vocab')
    print('--------------------------------')
    for token, id in character_tokenizer.atom_to_idx.items():
        print(f'{token}: {id}')
    print('--------------------------------')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True)
    
    wandb_enabled = opts.enable_wandb
    if wandb_enabled:
        init_wandb(opts)
    
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
            branching_flows=opts.interpolant == 'branching',
            euclidean_dim=euclidean_dim,
            vocab_size=character_tokenizer.vocab_size,
            symbols_depth=4,
            positions_depth=4,
            depth=4,
            dim_modalities=[hidden_dim, hidden_dim],
            dim_joint_attn=hidden_dim,
            dim_conds=[hidden_dim, hidden_dim]
        ).to(device)
    ema = deepcopy(model)
    # dim_2_params = [p for p in model.parameters() if p.ndim == 2] # Selects weights of Linear layers
    # other_params = [p for p in model.parameters() if p.ndim != 2]

    opt = torch.optim.AdamW(model.parameters(),lr=opts.lr)
    # muon  = torch.optim.Muon(dim_2_params,lr=opts.lr)
    scheduler = WarmUpScheduler(opt, opts.warmup_iters)
    scaler = torch.amp.GradScaler()
    
    if opts.interpolant == 'multimodal':
        interpolant = MultimodalInterpolant(
        max_length=dataset.max_length,
        vocab_size=character_tokenizer.vocab_size,
        mask_token=character_tokenizer.mask_token_id,
        pad_token=character_tokenizer.pad_token_id, 
        bos_token=character_tokenizer.bos_token_id,
        euclidean_dim=euclidean_dim,
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
    elif opts.interpolant == 'branching':
        interpolant = BranchingFlowsInterpolant(
            vocab_size=character_tokenizer.vocab_size,
            mask_token=character_tokenizer.mask_token_id,
            pad_token=character_tokenizer.pad_token_id,
            euclidean_dim=euclidean_dim,
        )
    start_iter = 0
    if opts.load_checkpoint is not None:
        start_iter = load_checkpoint(opts, device, model, ema, opt, scheduler)

    
    model.train()
    model = model.to(device)
    
    num_iters = opts.num_iters

    training_iter = start_iter
    log_rate = opts.log_rate
    while training_iter < num_iters:
        pbar = tqdm(dataloader,total=len(dataloader),leave=False)
        for data_ in pbar:
            if training_iter > num_iters:
                break
            for key, value in data_.items():
                data_[key] = value.to(device=device)
            
            opt.zero_grad()
            
            losses = interpolant.compute_loss(model, data_)
            if opts.interpolant == 'multimodal':
                loss = losses["dsm_loss"] + losses["discrete_unmasking_loss"] + losses["euclidean_unmasking_loss"] + losses["insertion_loss"]
            elif opts.interpolant == 'multimodal_both':
                loss = losses["dsm_loss"] + losses["discrete_unmasking_loss"] + losses["euc_insertion_loss"] + losses["disc_insertion_loss"]
            elif opts.interpolant == 'branching':
                loss = losses["dsm_loss"] + losses["discrete_unmasking_loss"] + losses["insertion_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            update_ema(ema, model, decay=opts.ema_beta)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            
            for param in model.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
            
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            
            training_iter += 1
            
            loss = loss.detach().item()
            
            
            if opts.interpolant == 'multimodal':
                pbar.set_description(f'Iter {training_iter} --- DSM Loss: {losses["dsm_loss"] :6.4f}, Discrete Unmasking Loss: {losses["discrete_unmasking_loss"] :6.4f}, Euclidean Unmasking Loss: {losses["euclidean_unmasking_loss"] :6.4f}, Insertion Loss: {losses["insertion_loss"] :6.4f}')
            elif opts.interpolant == 'multimodal_both':
                pbar.set_description(f'Iter {training_iter} --- DSM Loss: {losses["dsm_loss"] :6.4f}, Discrete Unmasking Loss: {losses["discrete_unmasking_loss"] :6.4f}, Euclidean Insertion Loss: {losses["euc_insertion_loss"] :6.4f}, Discrete Insertion Loss: {losses["disc_insertion_loss"] :6.4f}')
            elif opts.interpolant == 'branching':
                pbar.set_description(f'Iter {training_iter} --- DSM Loss: {losses["dsm_loss"] :6.4f}, Discrete Unmasking Loss: {losses["discrete_unmasking_loss"] :6.4f}, Insertion Loss: {losses["insertion_loss"] :6.4f}')
            if wandb_enabled:
                if opts.interpolant == 'multimodal':
                    wandb.log({
                        'loss': loss,
                        'dsm_loss': losses["dsm_loss"],
                        'discrete_unmasking_loss': losses["discrete_unmasking_loss"],
                        'insertion_loss': losses["insertion_loss"],
                        'step': training_iter
                    })
                elif opts.interpolant == 'multimodal_both':
                    wandb.log({
                        'loss': loss,
                        'dsm_loss': losses["dsm_loss"],
                        'discrete_unmasking_loss': losses["discrete_unmasking_loss"],
                        'euc_insertion_loss': losses["euc_insertion_loss"],
                        'disc_insertion_loss': losses["disc_insertion_loss"],
                        'step': training_iter
                    })
                elif opts.interpolant == 'branching':
                    wandb.log({
                        'loss': loss,
                        'dsm_loss': losses["dsm_loss"],
                        'discrete_unmasking_loss': losses["discrete_unmasking_loss"],
                        'insertion_loss': losses["insertion_loss"],
                        'step': training_iter
                    })
            # Evaluate sample accuracy
            if training_iter%log_rate == 0 or training_iter == num_iters:
                path = os.path.join(opts.dir, f'itr_{training_iter}/')
                os.makedirs(path, exist_ok=True)
                save_ckpt(model, ema, opt, scheduler, os.path.join(path, 'snapshot.pt'))
                model.eval()

                samples = interpolant.sampling(model, 50, 20, dataset.max_length+1, device, return_trace=True)
                for i, sample in enumerate(samples):
                    plot_sample(sample.xt.cpu(), sample.yt.cpu(), sample.mask_t.cpu(), os.path.join(path, f'sample_{i}.png'), character_tokenizer)

                    # os.makedirs(os.path.join(path, f'trajectory_{i}'), exist_ok=True)
                    # pbar = tqdm(enumerate(sample.trajectory), leave=False)
                    # for j, trajectory in pbar:
                    #     plot_sample(trajectory.xt.cpu(), trajectory.yt.cpu(), trajectory.mask_t.cpu(), os.path.join(path, f'trajectory_{i}', f'step_{j}.png'), character_tokenizer)
                    #     pbar.set_description(f'Saving trajectory {i} step {j}')
                model.train()

    save_ckpt(model, ema, opt, scheduler, os.path.join(opts.dir, 'final_checkpoint.pt'))

    if wandb_enabled:
        wandb.finish()

def load_checkpoint(opts, device, model, ema,opt, scheduler):
    snapshot = torch.load(os.path.join(opts.load_checkpoint), weights_only=True)
    model.load_state_dict(snapshot['model'],strict=False)
    ema.load_state_dict(snapshot['ema'], strict=False)
    opt.load_state_dict(snapshot['optimizer'])
    scheduler.load_state_dict(snapshot['scheduler'])
        
    start_iter = scheduler.last_epoch
    return start_iter

def save_ckpt(model, ema, opt, scheduler, path):
    snapshot = {
                    'model': model.state_dict(),
                    'ema': ema.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
    torch.save(snapshot,path)


if __name__ == '__main__':
    training()