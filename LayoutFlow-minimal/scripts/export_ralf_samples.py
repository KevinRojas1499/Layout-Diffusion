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

    torch.manual_seed(seed)
    results, ours = [], []
    for batch in loader:
        ids = batch.pop('id')
        batch = {k: v.cuda() for k, v in batch.items()}
        geom, label, keep = model.inference(batch)
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
