'''
Crello layout + text evaluation. Generates for the test split with a LayoutFlowText checkpoint (or reads a saved
sample file) and reports, for generated vs real layouts:
  layout : alignment, overlap (src.metrics, as test.py), element count MAE, class histogram
  text   : fraction of empty / whitespace-only strings, mean chars and words, distinct-1/2 over all strings,
           mean NLL per token under an external LM (Qwen2.5-0.5B), and the text-length / box-size relation:
           Pearson correlation of chars with box width, box area and with the number of lines vs box height.
  usage (text venv): python scripts/eval_crello.py <ckpt|samples.pt> <out.json> [overrides...]
'''
import json, sys, time
import numpy as np, torch, hydra
from hydra import compose, initialize_config_dir

sys.path.insert(0, '.')
from src.metrics import compute_alignment, compute_overlap

LAB = '/network/rit/lab/Yelab/kevin-back/kevin_rojas'


def generate(ckpt, overrides):
    import os
    with initialize_config_dir(config_dir=os.path.abspath('conf'), version_base=None):
        cfg = compose('train', overrides=['dataset=Crello', 'dataset_name=Crello', 'model=LayoutFlowText', f'dataset.dataset.data_path={LAB}/datasets/crello/layout',
                                          f'model.pretrained_dir={LAB}/repos/LayoutFlow/pretrained', 'model.fid_calc_every_n=0', 'dataset.num_workers=0',
                                          'dataset.persistent_workers=false', 'dataset.batch_size=128', 'dataset.shuffle=false', 'model.t_max=0.5'] + overrides)
    dl = hydra.utils.instantiate(cfg.dataset, dataset={'split': 'test'})
    model = hydra.utils.instantiate(cfg.model).cuda().eval()
    model.load_state_dict(torch.load(ckpt, map_location='cuda', weights_only=False)['state_dict'], strict=False)
    gen, real = [], []
    t0 = time.time()
    for batch in dl:
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            g, l, e = model.inference(batch)
        texts = model.last_texts
        for b in range(g.shape[0]):
            n = int(e[b].sum())
            gen.append({'bbox': g[b, :n].cpu().tolist(), 'type': l[b, :n].cpu().tolist(), 'text': texts[b][:n]})
            m = int(batch['length'][b]); bb = batch['bbox'][b, :m].cpu().tolist()
            real.append({'bbox': bb, 'type': batch['type'][b, :m].cpu().tolist(), 'text': list(batch['text'][b][:m])})
        print(f'{len(gen)} layouts, {time.time() - t0:.0f}s', flush=True)
    return gen, real


def layout_metrics(layouts, name):
    S = max(len(l['type']) for l in layouts)
    bbox = torch.zeros(len(layouts), S, 4); mask = torch.zeros(len(layouts), S, dtype=torch.bool)
    for i, l in enumerate(layouts):
        n = len(l['type'])
        if n:
            bbox[i, :n] = torch.tensor(l['bbox']).view(n, 4); mask[i, :n] = True
    keep = mask.sum(1) > 1
    align = compute_alignment(bbox[keep], mask[keep]) * 100
    overlap = compute_overlap(bbox[keep], mask[keep])
    counts = np.array([len(l['type']) for l in layouts])
    hist = np.bincount(np.concatenate([np.asarray(l['type'], dtype=int) for l in layouts if len(l['type'])]), minlength=6)[1:6]
    return {'n': len(layouts), 'empty_layouts': int((mask.sum(1) == 0).sum()), 'alignment_x100': align, 'overlap': overlap, 'count_mean': counts.mean(), 'class_hist': (hist / hist.sum()).round(3).tolist()}


def text_metrics(layouts, lm=None, text_id=2):
    items = [(s, b) for l in layouts for s, b, t in zip(l['text'], l['bbox'], l['type']) if t == text_id]
    strings = [s for s, _ in items]
    clean = [s.strip() for s in strings if s.strip()]
    out = {'n_text_elements': len(strings), 'empty_frac': 1 - len(clean) / max(len(strings), 1),
           'chars_mean': float(np.mean([len(s) for s in clean])) if clean else 0, 'words_mean': float(np.mean([len(s.split()) for s in clean])) if clean else 0}
    toks = [w for s in clean for w in s.lower().split()]
    big = [tuple(s.lower().split()[i:i + 2]) for s in clean for i in range(len(s.split()) - 1)]
    out['distinct1'] = len(set(toks)) / max(len(toks), 1); out['distinct2'] = len(set(big)) / max(len(big), 1)
    # text length vs box size (on non-empty strings)
    ch = np.array([len(s.strip()) for s, _ in items if s.strip()]); ln = np.array([s.strip().count('\n') + 1 for s, _ in items if s.strip()])
    bw = np.array([b[2] for s, b in items if s.strip()]); bh = np.array([b[3] for s, b in items if s.strip()]); ba = bw * bh
    if len(ch) > 2:
        out['corr_chars_width'] = float(np.corrcoef(ch, bw)[0, 1]); out['corr_chars_area'] = float(np.corrcoef(ch, ba)[0, 1]); out['corr_lines_height'] = float(np.corrcoef(ln, bh)[0, 1])
        out['chars_per_width'] = float(np.median(ch / np.clip(bw, 1e-3, None)))
    if lm is not None:
        out['lm_nll_per_token'] = lm(clean)
    return out


def make_lm(name='Qwen/Qwen2.5-0.5B'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name); m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()

    def nll(strings):
        tot, cnt = 0.0, 0
        for i in range(0, len(strings), 64):
            enc = tok(['Poster text: ' + s for s in strings[i:i + 64]], return_tensors='pt', padding=True, truncation=True, max_length=48).to('cuda')
            with torch.no_grad():
                logits = m(**enc).logits.float()
            pre = len(tok('Poster text: ')['input_ids'])
            lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, enc.input_ids[:, 1:, None]).squeeze(-1)
            msk = enc.attention_mask[:, 1:].bool(); msk[:, :pre - 1] = False               # score only the string's tokens
            tot += float(-(lp * msk).sum()); cnt += int(msk.sum())
        return tot / max(cnt, 1)
    return nll


if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]
    if src.endswith('.pt') and 'checkpoints' not in src:
        d = torch.load(src, weights_only=False); gen, real = d['gen'], d['real']
    else:
        gen, real = generate(src, sys.argv[3:])
        torch.save({'gen': gen, 'real': real}, out.replace('.json', '.samples.pt'))
    lm = make_lm()
    res = {'gen': {**layout_metrics(gen, 'gen'), **text_metrics(gen, lm)}, 'real': {**layout_metrics(real, 'real'), **text_metrics(real, lm)}}
    res['count_mae'] = float(np.mean([abs(len(g['type']) - len(r['type'])) for g, r in zip(gen, real)]))
    json.dump(res, open(out, 'w'), indent=1)
    for k in res['gen']:
        print(f'{k:22s} gen {res["gen"][k]}   real {res["real"][k]}')
    print('count MAE (paired by canvas index)', res['count_mae'])
