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

    return default_collate(batch)


class H5LayoutDataset(Dataset):
    '''
    Shared dataset for RICO and PubLayNet: both are stored as one h5 file per
    split with the same per-sample keys ('type', 'bbox', 'length', ...); only
    the file names and category count differ.
    '''

    def __init__(self, files_by_split, split='train', data_path='.', num_cat=6):
        super().__init__()
        self.num_cat = num_cat
        self.data = h5py.File(f'{data_path}/{files_by_split[split]}')
        self.keys = list(self.data.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        sample = self.data[key]
        sample_dict = {feature: torch.from_numpy(np.array(sample[feature])) for feature in sample.keys()}
        if 'categories' in sample_dict:
            sample_dict['type'] = sample_dict.pop('categories')
        return sample_dict


RICO_FILES = {'train': 'ldm_rico_train.h5', 'validation': 'ldm_rico_val.h5', 'test': 'ldm_rico_test.h5'}
PUBLAYNET_FILES = {'train': 'publaynet_train.h5', 'validation': 'publaynet_val.h5', 'test': 'publaynet_test.h5'}


class RICO(H5LayoutDataset):
    def __init__(self, split='train', data_path='./rico', num_cat=26):
        super().__init__(RICO_FILES, split=split, data_path=data_path, num_cat=num_cat)


class PubLayNet(H5LayoutDataset):
    def __init__(self, split='train', data_path='./publaynet', num_cat=6):
        super().__init__(PUBLAYNET_FILES, split=split, data_path=data_path, num_cat=num_cat)
