'''
Crello (HF cyberagent/crello) -> layout files for src.data.Crello.
Per template: id, canvas size, elements in z-order with type (1..5: SvgElement, TextElement, ImageElement,
ColoredBackground, SvgMaskElement), box (cx, cy, w, h) normalised by the canvas, text string, font id, font size
(relative to canvas height). Boxes are kept as they are (backgrounds may exceed the canvas). Templates with more
than max_len elements are dropped (95% have <= 20).
  usage: HF_HOME=... python scripts/prepare_crello.py <out_dir> [max_len]
'''
import sys, torch, datasets, numpy as np
out, max_len = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20
ds = datasets.load_dataset('cyberagent/crello')
tn = ds['train'].features['type'].feature.names
print('types', tn)
for split in ['train', 'validation', 'test']:
    d = ds[split].remove_columns(['preview', 'image']).with_format(None)
    recs, dropped = [], 0
    for e in d:
        n = e['length']
        if n > max_len or n == 0:
            dropped += 1; continue
        W, H = e['canvas_width'], e['canvas_height']
        box = torch.tensor([[(e['left'][i] + e['width'][i] / 2) / W, (e['top'][i] + e['height'][i] / 2) / H, e['width'][i] / W, e['height'][i] / H] for i in range(n)], dtype=torch.float32)
        recs.append({'id': e['id'], 'canvas': (W, H), 'type': torch.tensor([t + 1 for t in e['type']], dtype=torch.long), 'bbox': box,
                     'text': [e['text'][i] if tn[e['type'][i]] == 'TextElement' else '' for i in range(n)],
                     'font': torch.tensor(e['font'], dtype=torch.long), 'font_size': torch.tensor([s / H for s in e['font_size']], dtype=torch.float32),
                     'format': e['format'], 'category': e['category']})
    torch.save(recs, f'{out}/{split}.pt')
    L = np.array([len(r['type']) for r in recs]); B = torch.cat([r['bbox'] for r in recs])
    print(f'{split}: {len(recs)} templates (dropped {dropped} with > {max_len} elements); elements/template {L.mean():.1f}; '
          f'box cx in [{B[:,0].min():.2f},{B[:,0].max():.2f}], w in [{B[:,2].min():.2f},{B[:,2].max():.2f}]; outside-canvas elements {((B[:,0]-B[:,2]/2 < -0.01)|(B[:,0]+B[:,2]/2 > 1.01)|(B[:,1]-B[:,3]/2 < -0.01)|(B[:,1]+B[:,3]/2 > 1.01)).float().mean():.3f}')
