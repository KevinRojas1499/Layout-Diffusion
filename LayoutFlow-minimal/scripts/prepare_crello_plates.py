'''
Canvas plates for content-aware Crello: in a Crello template the canvas is element 0 -- a full-canvas image,
colour or vector plate (80% of templates). This script writes that plate as a JPEG per template and re-emits the
layout with the plate removed, so a model conditions on the canvas and generates the remaining elements.
Templates without a canvas-covering first element get a flat plate from the ColoredBackground colour, else white.
  HF_HOME=... python scripts/prepare_crello_plates.py <layout_dir> <plate_dir> [size]
Writes <plate_dir>/{split}/{id}.jpg and <layout_dir>/{split}_canvas.pt (the layouts minus the plate element).
'''
import os, sys, torch, datasets
from PIL import Image

layout_dir, plate_dir = sys.argv[1], sys.argv[2]
size = int(sys.argv[3]) if len(sys.argv) > 3 else 384
ds = datasets.load_dataset('cyberagent/crello')
tn = ds['train'].features['type'].feature.names
for split, name in [('train', 'train'), ('validation', 'validation'), ('test', 'test')]:
    recs = torch.load(f'{layout_dir}/{name}.pt', weights_only=False)
    want = {r['id']: n for n, r in enumerate(recs)}
    os.makedirs(f'{plate_dir}/{name}', exist_ok=True)
    out, stats = [None] * len(recs), {'image': 0, 'flat': 0, 'missing': 0}
    d = ds[split]
    for k in range(len(d)):
        ex = d[k]
        if ex['id'] not in want:
            continue
        W, H = ex['canvas_width'], ex['canvas_height']
        idx = None
        for i in range(ex['length']):
            covers = (ex['left'][i] <= 0.02 * W and ex['top'][i] <= 0.02 * H
                      and ex['left'][i] + ex['width'][i] >= 0.98 * W and ex['top'][i] + ex['height'][i] >= 0.98 * H)
            if covers and tn[ex['type'][i]] != 'TextElement':
                idx = i
                break
        im = None
        if idx is not None and ex['image'][idx] is not None:
            im = ex['image'][idx].convert('RGB'); stats['image'] += 1
        else:
            col = (255, 255, 255)
            if idx is not None and ex['color'][idx]:
                try:
                    c = ex['color'][idx][0]
                    col = tuple(int(v) for v in c.replace('rgba(', '').replace(')', '').split(',')[:3])
                except Exception:
                    pass
            im = Image.new('RGB', (W, H), col); stats['flat'] += 1
        im = im.resize((size, size), Image.BICUBIC)
        im.save(f'{plate_dir}/{name}/{ex["id"]}.jpg', quality=88, optimize=True)
        r = dict(recs[want[ex['id']]])
        if idx is not None:                                   # drop the plate element from the layout
            keep = [j for j in range(len(r['type'])) if j != idx]
            r = {**r, 'type': r['type'][keep], 'bbox': r['bbox'][keep], 'text': [r['text'][j] for j in keep]}
        r['plate'] = f'{name}/{ex["id"]}.jpg'
        out[want[ex['id']]] = r
    miss = [n for n, r in enumerate(out) if r is None]
    out = [r for r in out if r is not None]
    torch.save(out, f'{layout_dir}/{name}_canvas.pt')
    L = [len(r['type']) for r in out]
    print(f'{name}: {len(out)} templates (plates: {stats["image"]} image, {stats["flat"]} flat; {len(miss)} unmatched); '
          f'elements/template {sum(L)/len(L):.1f}', flush=True)
