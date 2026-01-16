import os
import click
import torch
import torch.distributed as dist
import wandb
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from multimodal_interpolant import MultimodalInterpolant
from custom_datasets.qm9 import QM9Dataset
from utils.misc import dotdict
from utils.tokenizer import VocabTokenizer
from utils.optimizers import WarmUpScheduler
from models.mmdit_qm9 import MMDiTQM9
from visualize_dataset import plot_sample, plot_molecule

# This makes training on A100s faster
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def init_wandb(opts):
    wandb.init(
        # set the wandb project where this run will be logged
        project='MMDiT-QM9',
        name= f'qm9-{opts.run_name}',
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
@click.option('--model',type=click.Choice(['radd', 'DiT']), default='DiT')
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
    batch_size = opts.batch_size
    
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True)
    
    wandb_enabled = opts.enable_wandb and rank == 0 # We only want to log once
    if wandb_enabled:
        init_wandb(opts)
    
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
    ema = deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(),lr=opts.lr)
    scheduler = WarmUpScheduler(opt, opts.warmup_iters)
    scaler = torch.amp.GradScaler(device)
    
    interpolant = MultimodalInterpolant(
        max_length=dataset.max_length,
        vocab_size=character_tokenizer.vocab_size,
        mask_token=character_tokenizer.mask_token_id,
        pad_token=character_tokenizer.pad_token_id, 
        bos_token=character_tokenizer.bos_token_id,
        eos_token=character_tokenizer.eos_token_id,
        euclidean_dim=3,
    )
    start_iter = 0
    if opts.load_checkpoint is not None:
        start_iter = load_checkpoint(opts, rank, device, model, ema, opt, scheduler)

    dist.barrier(device_ids=[device])
    
    model.train()
    model = DDP(model)
    
    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)//1e6} M")
    
    if not os.path.exists(opts.dir) and rank == 0:
        os.makedirs(opts.dir)

    num_iters = opts.num_iters

    training_iter = start_iter
    log_rate = opts.log_rate
    while training_iter < num_iters:
        pbar = tqdm(dataloader,total=len(dataloader),leave=False) if rank == 0 else dataloader
        for data_ in pbar:
            if training_iter > num_iters:
                break
            for key, value in data_.items():
                data_[key] = value.to(device=device)
            
            opt.zero_grad()
            
            losses = interpolant.compute_loss(model, data_)
            loss = losses["dsm_loss"] + losses["tokens_loss"] + losses["insertion_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            update_ema(ema, model.module, decay=opts.ema_beta)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            
            for param in model.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
            
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            
            training_iter += 1
            
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss.detach().item()/world_size
            
            
            if rank == 0:
                pbar.set_description(f'Iter {training_iter} --- DSM Loss: {losses["dsm_loss"] :6.4f}, Tokens Loss: {losses["tokens_loss"] :6.4f}, Insertion Loss: {losses["insertion_loss"] :6.4f}')
            if wandb_enabled:
                wandb.log({
                'loss': loss/world_size,
                'dsm_loss': losses["dsm_loss"]/world_size,
                'tokens_loss': losses["tokens_loss"]/world_size,
                'insertion_loss': losses["insertion_loss"]/world_size,
                'step': training_iter
            })
            dist.barrier(device_ids=[device])
            # Evaluate sample accuracy
            if training_iter%log_rate == 0 or training_iter == num_iters:
                path = os.path.join(opts.dir, f'itr_{training_iter}/')
                os.makedirs(path, exist_ok=True)
                if rank == 0:
                    save_ckpt(model, ema, opt, scheduler, os.path.join(path, 'snapshot.pt'))
                model.eval()
                dist.barrier(device_ids=[device])

                samples = interpolant.euclidean_sampling(model, 50, 5, dataset.max_length, device, return_trace=True)
                for i, sample in enumerate(samples):
                    plot_sample(sample.xt.cpu(), sample.yt.cpu(), sample.mask_t.cpu(), os.path.join(path, f'sample_{i}.png'), character_tokenizer)
                    symbols = character_tokenizer.decode(sample.yt.cpu())
                    positions = sample.xt.cpu()[1:len(symbols)+1, :]
                    plot_molecule(symbols, positions, os.path.join(path, f'molecule_{i}.png'))
                    
                    # os.makedirs(os.path.join(path, f'trajectory_{i}'), exist_ok=True)
                    # pbar = tqdm(enumerate(sample.trajectory), leave=False)

                    # for j, trajectory in pbar:
                    #     plot_sample(trajectory.xt.cpu(), trajectory.yt.cpu(), trajectory.mask_t.cpu(), os.path.join(path, f'trajectory_{i}', f'step_{j}.png'), character_tokenizer)
                    #     pbar.set_description(f'Saving trajectory {i} step {j}')

    if rank == 0:
        save_ckpt(model, ema, opt, scheduler, os.path.join(opts.dir, 'final_checkpoint.pt'))

    dist.barrier(device_ids=[device])
    if wandb_enabled:
        wandb.finish()
    dist.destroy_process_group()

def load_checkpoint(opts, rank, device, model, ema,opt, scheduler):
    print(f'Loading checkpoint from {opts.load_checkpoint} in rank {rank}')
    snapshot = torch.load(os.path.join(opts.load_checkpoint), weights_only=True)
    model.load_state_dict(snapshot['model'],strict=False)
    ema.load_state_dict(snapshot['ema'], strict=False)
    opt.load_state_dict(snapshot['optimizer'])
    scheduler.load_state_dict(snapshot['scheduler'])
        
    start_iter = scheduler.last_epoch
    return start_iter

def save_ckpt(model, ema, opt, scheduler, path):
    snapshot = {
                    'model': model.module.state_dict(),
                    'ema': ema.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
    torch.save(snapshot,path)


if __name__ == '__main__':
    training()
