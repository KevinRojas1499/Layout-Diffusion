import numpy as np
import torch
import h5pickle as h5py
from torch.utils.data import Dataset, default_collate


def collate_fn(batch, max_len=None, format='xywh'):
    total_elems = [len(example["type"]) for example in batch]
    max_len = max(total_elems) if max_len is None else max_len
    B = len(batch)

    for key in ['mask', 'length', 'type', 'bbox']:
        if key == 'type':
            dtype, size = torch.int, (max_len,)
        elif key == 'bbox':
            dtype, size = torch.float32, (max_len, 4)
        elif key == 'mask':
            dtype, size = torch.bool, (max_len, 1)
        else:
            dtype, size = torch.int, (max_len,)

        for i in range(B):
            dummy_array = torch.full(fill_value=0, dtype=dtype, size=size)
            if key in ['length', 'mask']:
                L = batch[i]['length'].squeeze().item()
                if key == 'length':
                    dummy_array = min(L, max_len)
                else:
                    dummy_array[:L] = True
            elif key == 'bbox':
                dummy_array[:total_elems[i]] = batch[i][key][:max_len]
                if format == 'xywh':
                    dummy_array[:, 0] += dummy_array[:, 2] / 2
                    dummy_array[:, 1] += dummy_array[:, 3] / 2
                elif format == 'ltrb':
                    dummy_array[:, 2] += dummy_array[:, 0]
                    dummy_array[:, 3] += dummy_array[:, 1]
            else:
                dummy_array[:total_elems[i]] = batch[i][key][:max_len]
            batch[i][key] = dummy_array

    extra = {k: [b.pop(k) for b in batch] for k in ('ctx', 'id', 'ret', 'sal') if k in batch[0]}
    out = default_collate(batch)
    if 'ctx' in extra:
        out['ctx'] = torch.stack(extra['ctx'])
    if 'id' in extra:
        out['id'] = extra['id']
    if 'ret' in extra:
        out['ret'] = torch.stack(extra['ret'])
    if 'sal' in extra:
        out['sal'] = torch.stack(extra['sal'])
    return out


class H5LayoutDataset(Dataset):
    '''
    Shared dataset for RICO and PubLayNet: both are stored as one h5 file per
    split with the same per-sample keys ('type', 'bbox', 'length', ...); only
    the file names and category count differ.
    '''

    def __init__(self, files_by_split, split='train', data_path='.', num_cat=6, in_memory=False):
        super().__init__()
        self.num_cat = num_cat
        self.data = h5py.File(f'{data_path}/{files_by_split[split]}')
        self.keys = list(self.data.keys())
        # The h5 group reads dominate an epoch's CPU time; the whole split is a few MB.
        self.cache = [self._load(i) for i in range(len(self.keys))] if in_memory else None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.cache is not None:
            return dict(self.cache[index])     # shallow copy: collate_fn overwrites entries in place
        return self._load(index)

    def _load(self, index):
        key = self.keys[index]
        sample = self.data[key]
        sample_dict = {feature: torch.from_numpy(np.array(sample[feature])) for feature in sample.keys()}
        if 'categories' in sample_dict:
            sample_dict['type'] = sample_dict.pop('categories')
        return sample_dict


RICO_FILES = {'train': 'ldm_rico_train.h5', 'validation': 'ldm_rico_val.h5', 'test': 'ldm_rico_test.h5'}
PUBLAYNET_FILES = {'train': 'publaynet_train.h5', 'validation': 'publaynet_val.h5', 'test': 'publaynet_test.h5'}


class RICO(H5LayoutDataset):
    def __init__(self, split='train', data_path='./rico', num_cat=26, in_memory=False):
        super().__init__(RICO_FILES, split=split, data_path=data_path, num_cat=num_cat, in_memory=in_memory)


class PubLayNet(H5LayoutDataset):
    def __init__(self, split='train', data_path='./publaynet', num_cat=6, in_memory=False):
        super().__init__(PUBLAYNET_FILES, split=split, data_path=data_path, num_cat=num_cat, in_memory=in_memory)


class CGL(Dataset):
    '''
    RALF's preprocessed CGL release (parquet: id, image, saliency, label (strings), center_x/y,
    width/height in [0, 1]; max 10 elements) plus frozen canvas features from
    scripts/precompute_canvas_feats.py. Elements come out in the same (cx, cy, w, h) / 1-indexed
    category convention as RICO and PubLayNet; category 0 is the pad id.
    '''
    LABELS = ['logo', 'text', 'underlay', 'embellishment']
    SPLITS = {'train': 'train', 'validation': 'val', 'test': 'test', 'unannotated': 'with_no_annotations_test'}

    RETRIEVAL_NAMES = {'train': 'train', 'val': 'val', 'test': 'test', 'with_no_annotations_test': 'with_no_annotation'}

    def __init__(self, split='train', data_path='./cgl', feats_path='./canvas_feats/cgl', num_cat=5, in_memory=True,
                 pad_empty=False, max_len=10, hflip=False, grid=8, retrieval_k=0, salbox=False):
        import glob
        import pyarrow.parquet as pq
        super().__init__()
        self.num_cat = num_cat
        name = self.SPLITS[split]
        tab = pq.ParquetDataset(sorted(glob.glob(f'{data_path}/{name}-*.parquet'))).read(
            columns=['id', 'label', 'center_x', 'center_y', 'width', 'height'])
        cols = {c: tab.column(c).to_pylist() for c in tab.column_names}
        feats = torch.load(f'{feats_path}/{name}.pt')
        order = {i: n for n, i in enumerate(feats['id'])}
        self.ctx = feats['feats']
        # padding baseline: every layout has max_len elements, the missing ones of an 'empty' class (id num_cat-1)
        self.pad_empty, self.max_len = pad_empty, max_len
        self.hflip, self.grid = hflip and split == 'train', grid       # mirror canvas + layout w.p. 0.5 (training only)
        self.samples = []
        for n, i in enumerate(cols['id']):
            lab = torch.tensor([self.LABELS.index(l) + 1 for l in cols['label'][n]], dtype=torch.long)
            box = torch.tensor([cols[k][n] for k in ('center_x', 'center_y', 'width', 'height')], dtype=torch.float32).T.reshape(-1, 4)
            if pad_empty:
                k = max_len - len(lab)
                lab = torch.cat([lab, torch.full((k,), num_cat - 1, dtype=torch.long)])
                box = torch.cat([box, torch.zeros(k, 4)])
            self.samples.append({'id': i, 'type': lab, 'bbox': box, 'length': torch.tensor(len(lab)), 'ctx_idx': order[i]})
        # retrieval augmentation (RALF): the K DreamSim-nearest *training* layouts of each canvas, from RALF's
        # precomputed index tables (the train table excludes the query itself). ret (K, max_len, 5) = cx, cy, w, h, label.
        # saliency bounding box token (LayoutDiT / LayoutGD): scripts/precompute_salbox.py -> {split}_salbox.pt
        self.salbox = salbox
        if salbox:
            sb = torch.load(f'{feats_path}/{name}_salbox.pt')
            sb_order = {i: n for n, i in enumerate(sb['id'])}
            for smp in self.samples:
                smp['sal'] = sb['salbox'][sb_order[smp['id']]]
        self.retrieval_k = retrieval_k
        if retrieval_k:
            import os
            cache = os.path.dirname(os.path.dirname(os.path.normpath(data_path)))
            table = torch.load(f'{cache}/PRECOMPUTED_WEIGHT_DIR/retrieval_indexes/cgl_{self.RETRIEVAL_NAMES[name]}_dreamsim_wo_head_table_between_dataset_indexes_top_k32.pt', weights_only=False)
            db = pq.ParquetDataset(sorted(glob.glob(f'{data_path}/train-*.parquet'))).read(columns=['label', 'center_x', 'center_y', 'width', 'height'])
            dbc = {c: db.column(c).to_pylist() for c in db.column_names}
            self.db = torch.zeros(len(dbc['label']), max_len, 5)
            for n in range(len(dbc['label'])):
                k = len(dbc['label'][n])
                self.db[n, :k, :4] = torch.tensor([dbc[c][n] for c in ('center_x', 'center_y', 'width', 'height')], dtype=torch.float32).T.reshape(-1, 4)[:max_len]
                self.db[n, :k, 4] = torch.tensor([self.LABELS.index(l) + 1 for l in dbc['label'][n]], dtype=torch.float32)[:max_len]
            for smp in self.samples:
                smp['ret_idx'] = torch.tensor(table[smp['id']][:retrieval_k])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = dict(self.samples[index])
        s['ctx'] = self.ctx[s.pop('ctx_idx')].float()
        s['bbox'] = s['bbox'].clone()
        if self.retrieval_k:
            s['ret'] = self.db[s.pop('ret_idx')].clone()
        if self.hflip and torch.rand(()) < 0.5:
            s['bbox'][:, 0] = 1 - s['bbox'][:, 0]                                   # mirror centres
            s['ctx'] = s['ctx'].view(self.grid, self.grid, -1).flip(1).reshape(self.grid * self.grid, -1)   # mirror the feature grid
            if self.retrieval_k:
                s['ret'][..., 0] = torch.where(s['ret'][..., 4] > 0, 1 - s['ret'][..., 0], s['ret'][..., 0])
            if self.salbox:
                s['sal'] = s['sal'].clone(); s['sal'][0] = 1 - s['sal'][0]
        # collate_fn builds xywh from an (x, y, w, h) corner box; CGL boxes are already centred
        s['bbox'][:, :2] -= s['bbox'][:, 2:] / 2
        return s
