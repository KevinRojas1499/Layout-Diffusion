'''
Qualitative samples from LayoutFlowVarLen checkpoints: real vs generated layouts, length and
class histograms, and snapshots of the insert -> unmask -> denoise trajectory, as one JSON
(draw it with whatever you like; src/visualize.draw_layout takes the same cx,cy,w,h boxes).
  python scripts/sample_layouts.py <out.json> <name>:<RICO|PubLayNet>:<t_max>:<ckpt> [...]
Generated layouts are the first N_GRID of a seeded batch, never selected. $LAB locates data + FID nets.
'''
import sys, json, shutil, os
import numpy as np
import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.environ.get('LAB', '/network/rit/lab/Yelab/kevin-back/kevin_rojas')
sys.path.insert(0, SRC)
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from src.backbone_varlen import gmm_sample, gmm_mean

N_GRID, N_STATS, N_TRAJ, SEED = 24, 2048, 5, 0
SNAP_T = [0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 1.00]
DATA_DIR = {'RICO': 'rico', 'PubLayNet': 'publaynet'}


@torch.no_grad()
def traced_inference(m, batch, snap_steps):
    '''Line-for-line copy of LayoutFlowVarLen.inference (same RNG call order) that also records states.'''
    Bn, S = batch['type'].shape
    dev = batch['bbox'].device
    exists = torch.zeros(Bn, S, dtype=torch.bool, device=dev) if m.insertion else batch['mask'].squeeze(-1).clone()
    x = torch.zeros(Bn, S, m.geom_dim, device=dev)
    y = torch.full((Bn, S), m.mask_id, dtype=torch.long, device=dev)
    N = m.inference_steps
    dt = 1.0 / N
    snaps = {}
    for i in range(N):
        t = torch.full((Bn,), i * dt, device=dev)
        t_next = (i + 1) * dt
        k, k_next = min(i * dt / m.t_max, 1.0), min(t_next / m.t_max, 1.0)
        p = 1.0 if (k_next >= 1.0 or i == N - 1) else (k_next - k) / (1 - k)
        v, logits, h, ins_rate, _ = m(x, y, exists, t)
        masked = exists & (y == m.mask_id)
        visible = exists & ~masked
        x = torch.where(visible.unsqueeze(-1), x + v * dt, x)
        reveal = masked & (torch.rand(Bn, S, device=dev) < p)
        if reveal.any():
            logits = logits[reveal]
            logits[:, 0] = float('-inf')
            cls = torch.distributions.Categorical(logits=logits).sample()
            log_pi, mu, sigma = m.model.unmask_geom(h[reveal], cls)
            g1 = gmm_sample(log_pi, mu, sigma) if m.unmask_geom == 'gmm' else gmm_mean(log_pi, mu)
            x[reveal] = t_next * g1 + (1 - t_next) * torch.randn_like(g1)
            y[reveal] = cls
        if m.insertion and p < 1.0:
            n_new = torch.poisson(ins_rate * p)
            free_rank = (~exists).cumsum(1)
            exists = exists | (~exists & (free_rank <= n_new.view(-1, 1)))
        if i + 1 in snap_steps:
            snaps[i + 1] = (m.sampler.preprocess(x, reverse=True).cpu(), y.cpu(), exists.cpu())
    order = exists.int().argsort(dim=1, descending=True, stable=True)
    x, y, exists = x.gather(1, order.unsqueeze(-1).expand_as(x)), y.gather(1, order), exists.gather(1, order)
    geom = exists.unsqueeze(-1) * m.sampler.preprocess(x, reverse=True)
    label = torch.where(exists, y, torch.zeros_like(y)).clamp(0, m.num_cat - 1)
    return geom, label, exists, snaps


def boxes(geom, label, keep):
    '''-> [[cls, cx, cy, w, h], ...] rounded, for the kept slots.'''
    return [[int(c)] + [round(float(v), 4) for v in g] for g, c, k in zip(geom, label, keep) if k]


def run(name, dataset, t_max, ckpt, dev):
    local = f'{os.path.dirname(os.path.abspath(sys.argv[1]))}/{name}.ckpt'
    shutil.copy(ckpt, local)          # training may rotate its top-k files while we read
    with initialize_config_dir(config_dir=f'{SRC}/conf', version_base=None):
        cfg = compose('train.yaml', overrides=[
            f'dataset={dataset}', f'dataset_name={dataset}', 'model=LayoutFlowVarLen', 'run_dir=/tmp/unused',
            f'dataset.dataset.data_path={B}/datasets/LayoutFlow-data/dataset/{DATA_DIR[dataset]}',
            f'model.pretrained_dir={B}/repos/LayoutFlow/pretrained', f'model.t_max={t_max}',
            'dataset.num_workers=0', 'dataset.persistent_workers=false', 'dataset.pin_memory=false'])
    model = instantiate(cfg.model, dataset=cfg.dataset_name, format=cfg.data.format, vis_dir=None)
    sd = torch.load(local, map_location='cpu', weights_only=False)
    model.load_state_dict(sd['state_dict'])
    model = model.to(dev).eval()
    assert model.insertion and abs(model.t_max - t_max) < 1e-9

    # real validation layouts
    val = instantiate(cfg.dataset, dataset={'split': 'validation'}, shuffle=False, batch_size=2048)
    rb, rl, rm = [], [], []
    for b in val:
        rb.append(b['bbox']); rl.append(b['type'].long()); rm.append(b['mask'].squeeze(-1))
    rb, rl, rm = torch.cat(rb), torch.cat(rl), torch.cat(rm)
    pick = torch.randperm(len(rb), generator=torch.Generator().manual_seed(SEED))[:N_GRID]

    # generated layouts: N_STATS samples; the grid shows the FIRST N_GRID of them, unselected
    S, nc = rb.shape[1], model.num_cat
    dummy = lambda n: {'type': torch.zeros(n, S, dtype=torch.long, device=dev), 'bbox': torch.zeros(n, S, 4, device=dev),
                       'mask': torch.zeros(n, S, 1, dtype=torch.bool, device=dev)}
    torch.manual_seed(SEED)
    gb, gl, gm = [], [], []
    for _ in range(N_STATS // 512):
        g, l, m_ = model.inference(dummy(512))
        gb.append(g.cpu()); gl.append(l.cpu()); gm.append(m_.cpu())
    gb, gl, gm = torch.cat(gb), torch.cat(gl), torch.cat(gm)

    # trajectory snapshots; must reproduce model.inference bit-for-bit under the same seed
    snap_steps = {round(t * model.inference_steps): t for t in SNAP_T}
    torch.manual_seed(SEED + 1); ref = model.inference(dummy(64))
    torch.manual_seed(SEED + 1); g, l, e, snaps = traced_inference(model, dummy(64), set(snap_steps))
    assert torch.equal(ref[0], g) and torch.equal(ref[1], l) and torch.equal(ref[2], e), 'traced loop diverged from model.inference'
    traj = []
    for j in range(N_TRAJ):
        frames = []
        for step in sorted(snaps):
            x, y, ex = snaps[step]
            vis = ex[j] & (y[j] != model.mask_id)
            frames.append({'t': snap_steps[step], 'boxes': boxes(x[j], y[j], vis), 'masked': int((ex[j] & ~vis).sum())})
        traj.append(frames)

    hist = lambda mask: np.bincount(mask.sum(1).numpy(), minlength=S + 1)[:S + 1].tolist()
    chist = lambda lab, mask: np.bincount(lab[mask].numpy(), minlength=nc)[:nc].tolist()
    # per-class mean box of the real data: lets a reader sanity-check the class-name mapping
    cls_box = {int(c): [round(float(v), 3) for v in rb[rm & (rl == c)].mean(0)] for c in range(1, nc) if (rm & (rl == c)).any()}
    out = {'name': name, 'dataset': dataset, 't_max': t_max, 'ckpt': os.path.basename(ckpt), 'epoch': int(sd['epoch']),
           'n_real': len(rb), 'n_gen': len(gb), 'empty_gen': int((~gm.any(1)).sum()),
           'real': [boxes(rb[i], rl[i], rm[i]) for i in pick], 'gen': [boxes(gb[i], gl[i], gm[i]) for i in range(N_GRID)],
           'len_real': hist(rm), 'len_gen': hist(gm), 'cls_real': chist(rl, rm), 'cls_gen': chist(gl, gm),
           'cls_mean_box': cls_box, 'traj': traj, 'snap_t': SNAP_T}
    print(f'{name}: epoch {out["epoch"]} | mean len real {rm.sum(1).float().mean():.2f} gen {gm.sum(1).float().mean():.2f} '
          f'| empty gen {out["empty_gen"]}/{len(gb)} | traced loop == model.inference: OK')
    return out


if __name__ == '__main__':
    dev = torch.device('cuda:0')
    res = [run(*(lambda n, d, t, c: (n, d, float(t), c))(*a.split(':', 3)), dev) for a in sys.argv[2:]]
    json.dump(res, open(sys.argv[1], 'w'), separators=(',', ':'))
    print('wrote', sys.argv[1], os.path.getsize(sys.argv[1]) // 1024, 'KB')
