import os
import click
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from euclidean_interpolant import EuclideanInterpolant
from utils.datasets import get_dataset 
from utils.misc import dotdict
from utils.optimizers import WarmUpScheduler
from model.transformer import EuclideanTransformer
# This makes training on A100s faster
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def init_wandb(opts):
    wandb.init(
        # set the wandb project where this run will be logged
        project='Euclidean Interpolant',
        name= f'{opts.model}-{opts.dataset}-{opts.max_length}',
        tags= ['training',opts.dataset],
        # # track hyperparameters and run metadata
        config=opts,
    )

@click.command()
@click.option('--dataset',type=click.Choice(['euclidean_variable_length_toy']), default='euclidean_variable_length_toy')
@click.option('--max_length',type=int, default=10)
@click.option('--model',type=click.Choice(['radd', 'DiT']), default='DiT')
@click.option('--optimizer',type=click.Choice(['adam','adamw']), default='adam')
@click.option('--lr', type=float, default=1e-5)
@click.option('--batch_size', type=int, default=32)
@click.option('--log_rate',type=int,default=5000)
@click.option('--num_iters',type=int,default=300000)
@click.option('--warmup_iters',type=int,default=500)
@click.option('--num_workers',type=int,default=2)
@click.option('--seed',type=int,default=42)
@click.option('--dir',type=str)
@click.option('--load_checkpoint',type=str, help='Directory where we can find the desired checkpoints')
@click.option('--enable_wandb', is_flag=True, default=False)
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

    dataset = get_dataset(opts.dataset, max_length=opts.max_length) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True)
    
    wandb_enabled = opts.enable_wandb and rank == 0 # We only want to log once
    if wandb_enabled:
        init_wandb(opts)
    
    model = EuclideanTransformer(
        hidden_size=384,
        cond_dim=384,
        n_heads=6,
        n_blocks=6,
        dropout=0.05,
        max_length=opts.max_length
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=opts.lr)
    scheduler = WarmUpScheduler(opt, opts.warmup_iters)
    scaler = torch.amp.GradScaler(device)
    
    interpolant = EuclideanInterpolant(
        max_length=opts.max_length,
    )
    start_iter = 0
    if opts.load_checkpoint is not None:
        start_iter = load_checkpoint(opts, rank, device, model, opt, scheduler)

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
            loss = losses["dsm_loss"] + losses["prediction_loss"] + losses["rate_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(opt)

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
                pbar.set_description(f'Iter {training_iter} --- DSM Loss: {losses["dsm_loss"] :6.4f}, Prediction Loss: {losses["prediction_loss"] :6.4f}, Rate Loss: {losses["rate_loss"] :6.4f}')
            if wandb_enabled:
                wandb.log({
                'loss': loss/world_size,
                'dsm_loss': losses["dsm_loss"]/world_size,
                'prediction_loss': losses["prediction_loss"]/world_size,
                'rate_loss': losses["rate_loss"]/world_size
            })
            dist.barrier(device_ids=[device])
            # Evaluate sample accuracy
            if training_iter%log_rate == 0 or training_iter == num_iters:
                path = os.path.join(opts.dir, f'itr_{training_iter}/')
                os.makedirs(path, exist_ok=True)
                if rank == 0:
                    save_ckpt(model, opt, scheduler, os.path.join(path, 'snapshot.pt'))
                model.eval()
                dist.barrier(device_ids=[device])
                

    if rank == 0:
        save_ckpt(model, opt, scheduler, os.path.join(opts.dir, 'final_checkpoint.pt'))

    dist.barrier(device_ids=[device])
    if wandb_enabled:
        wandb.finish()
    dist.destroy_process_group()

def load_checkpoint(opts, rank, device, model, opt, scheduler):
    print(f'Loading checkpoint from {opts.load_checkpoint} in rank {rank}')
    snapshot = torch.load(os.path.join(opts.load_checkpoint), weights_only=True)
    model.load_state_dict(snapshot['model'],strict=False)
    opt.load_state_dict(snapshot['optimizer'])
    scheduler.load_state_dict(snapshot['scheduler'])
        
    start_iter = scheduler.last_epoch
    return start_iter

def save_ckpt(model, opt, scheduler, path):
    snapshot = {
                    'model': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
    torch.save(snapshot,path)


if __name__ == '__main__':
    training()