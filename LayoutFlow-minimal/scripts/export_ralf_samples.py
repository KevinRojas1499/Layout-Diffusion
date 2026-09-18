'''
Generate layouts for RALF's CGL test split from a checkpoint and write them in the pickle format
RALF's eval.py consumes (results in parquet row order, RALF's alphabetical class ids), so their
validated metric suite (FID, occlusion, unreadability, underlay, overlay, ...) scores our samples:
  python scripts/export_ralf_samples.py <ckpt> <out_dir> <model> [hydra overrides...]
  cd $LAB/repos/RALF && .venv/bin/python eval.py --input-dir <out_dir> --fid-weight-dir cache/PRECOMPUTED_WEIGHT_DIR/fidnet/cgl \\
      --save-score-dir <scores> --dataset-path cache/dataset/cgl --run-on-local
The pickle also keeps our generated pad_mask/labels in our own ids for drawing (key 'ours').
'''
import glob, os, pickle, sys
import torch

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = os.environ.get('LAB', '/network/rit/lab/Yelab/kevin-back/kevin_rojas')
sys.path.insert(0, SRC)
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from src.data import CGL

RALF_NAMES = ['embellishment', 'logo', 'text', 'underlay']          # sorted(vocabulary) = RALF's class ids
TO_RALF = {i + 1: RALF_NAMES.index(n) for i, n in enumerate(CGL.LABELS)}
TEMPLATE = glob.glob(f'{B}/datasets/ralf/cache/training_logs/ralf_uncond_cgl/generated_samples_*/test_0.pkl')[0]


UNDERLAY = CGL.LABELS.index('underlay') + 1


def postprocess(geom, label, keep, steps):
    '''
    Explicitly labelled post-processing (env POST="snap128,contain"), reported as separate rows, never silently:
      snapN    round every box edge to the 1/N grid (RALF generates on a 128-bin grid)
      contain  an underlay that loosely contains a non-underlay element (>= 80% of its area inside) is enlarged
               minimally so that it contains it exactly (the strict underlay metric requires exact containment)
    geom (B,S,4) cx,cy,w,h in [0,1]; label (B,S) our ids; keep (B,S) bool.
    '''
    if not steps:
        return geom
    l, t_, r, b = geom[..., 0] - geom[..., 2] / 2, geom[..., 1] - geom[..., 3] / 2, geom[..., 0] + geom[..., 2] / 2, geom[..., 1] + geom[..., 3] / 2
    for step in steps.split(','):
        if step.startswith('snap'):
            n = int(step[4:])
            l, t_, r, b = [(v * n).round() / n for v in (l, t_, r, b)]
        elif step == 'contain':
            B, S = label.shape
            for i in range(B):
                u = torch.nonzero(keep[i] & (label[i] == UNDERLAY)).flatten().tolist()
                o = torch.nonzero(keep[i] & (label[i] != UNDERLAY)).flatten().tolist()
                for ui in u:
                    best, best_frac = None, 0.8
                    for oi in o:
                        iw = (torch.minimum(r[i, ui], r[i, oi]) - torch.maximum(l[i, ui], l[i, oi])).clamp(min=0)
                        ih = (torch.minimum(b[i, ui], b[i, oi]) - torch.maximum(t_[i, ui], t_[i, oi])).clamp(min=0)
                        frac = (iw * ih) / ((r[i, oi] - l[i, oi]) * (b[i, oi] - t_[i, oi])).clamp(min=1e-8)
                        if frac >= best_frac:
                            best, best_frac = oi, float(frac)
                    if best is not None:
                        l[i, ui], t_[i, ui] = torch.minimum(l[i, ui], l[i, best]), torch.minimum(t_[i, ui], t_[i, best])
                        r[i, ui], b[i, ui] = torch.maximum(r[i, ui], r[i, best]), torch.maximum(b[i, ui], b[i, best])
        else:
            raise ValueError(step)
    return torch.stack([(l + r) / 2, (t_ + b) / 2, r - l, b - t_], -1).clamp(0, 1)


@torch.no_grad()
def main(ckpt, out_dir, model_name, overrides, seed=0, split='test'):
    with initialize_config_dir(config_dir=f'{SRC}/conf', version_base=None):
        cfg = compose('train.yaml', overrides=[
            'dataset=CGL', 'dataset_name=CGL', f'model={model_name}', 'run_dir=/tmp/unused', 'data.max_len=10',
            f'dataset.dataset.data_path={B}/datasets/ralf/cache/dataset/cgl', f'dataset.dataset.feats_path={B}/datasets/ralf/canvas_feats/cgl',
            f'model.pretrained_dir={B}/repos/LayoutFlow/pretrained', 'model.fid_calc_every_n=0', 'model.scheduler=null',
            'dataset.num_workers=0', 'dataset.persistent_workers=false', 'dataset.pin_memory=false'] + overrides)
    model = instantiate(cfg.model, dataset='CGL', format=cfg.data.format, vis_dir=None)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(sd['state_dict']); model = model.cuda().eval()
    empty_id = cfg.dataset.dataset.num_cat - 1 if cfg.dataset.dataset.get('pad_empty', False) else None
    loader = instantiate(cfg.dataset, dataset={'split': split}, shuffle=False, batch_size=512)

    # optional sampler variants (env): SAMPLING="gmm_temp=0.5,solver=heun" ; STEPS=200 ; SELF_REFINE=0.97 (t_start of a
    # refinement pass of our own refinement mode on the generated layout, categories fixed, boxes re-integrated)
    for kv in filter(None, os.environ.get('SAMPLING', '').split(',')):
        k, v = kv.split('='); model.sampling[k] = type(model.sampling[k])(v) if not isinstance(model.sampling[k], bool) else v == 'true'
    if os.environ.get('STEPS'):
        model.inference_steps = int(os.environ['STEPS'])
    self_refine = float(os.environ.get('SELF_REFINE', 0) or 0)
    print('sampling:', model.sampling, '| steps', model.inference_steps, '| self_refine', self_refine)
    torch.manual_seed(seed)
    results, ours = [], []
    for batch in loader:
        ids = batch.pop('id')
        batch = {k: v.cuda() for k, v in batch.items()}
        geom, label, keep = model.inference(batch)
        if self_refine:
            b2 = dict(batch); b2['bbox'], b2['type'], b2['mask'], b2['length'] = geom, label, keep.unsqueeze(-1), keep.sum(1)
            geom, label, keep = model.inference(b2, task='refinement', t_start=self_refine)
        geom = postprocess(geom, label, keep, os.environ.get('POST', ''))
        if empty_id is not None:
            keep = keep & (label != empty_id)
        for i in range(len(ids)):
            k = keep[i]
            g, l = geom[i][k].cpu(), label[i][k].cpu()
            results.append({'label': [TO_RALF[int(c)] for c in l], 'center_x': g[:, 0].tolist(), 'center_y': g[:, 1].tolist(),
                            'width': g[:, 2].tolist(), 'height': g[:, 3].tolist(), 'id': ids[i]})
            ty = batch['type'][i]
            n_true = int(((ty != 0) & (ty != empty_id)).sum()) if empty_id is not None else int(batch['length'][i])
            ours.append({'id': ids[i], 'bbox': g, 'label': l, 'n_true': n_true})
    tpl = pickle.load(open(TEMPLATE, 'rb'))
    os.makedirs(out_dir, exist_ok=True)
    pickle.dump({'results': results, 'train_cfg': tpl['train_cfg'], 'test_cfg': tpl['test_cfg']}, open(f'{out_dir}/{split}_{seed}.pkl', 'wb'))
    torch.save({'ckpt': ckpt, 'epoch': int(sd['epoch']), 'samples': ours}, f'{out_dir}/ours_{split}_{seed}.pt')
    n_gen = torch.tensor([len(r['label']) for r in results], dtype=torch.float)
    n_true = torch.tensor([o['n_true'] for o in ours], dtype=torch.float)
    print(f'{out_dir}: epoch {sd["epoch"]} | {len(results)} layouts | mean N gen {n_gen.mean():.2f} true {n_true.mean():.2f} '
          f'| count MAE {(n_gen - n_true).abs().mean():.3f} | exact {(n_gen == n_true).float().mean():.3f} | empty {(n_gen == 0).sum().item()}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
