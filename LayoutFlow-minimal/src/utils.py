import torch


def convert_bbox(bbox, conv='xywh->ltrb'):
    '''bbox (*, 4); conv one of xywh<->ltwh<->ltrb (any pair, either direction).'''
    src, dst = conv.split('->')
    if src == dst:
        return bbox
    a, b, c, d = bbox.movedim(-1, 0)
    if conv == 'xywh->ltwh':
        out = [a - c / 2, b - d / 2, c, d]
    elif conv == 'ltwh->xywh':
        out = [a + c / 2, b + d / 2, c, d]
    elif conv == 'xywh->ltrb':
        out = [a - c / 2, b - d / 2, a + c / 2, b + d / 2]
    elif conv == 'ltrb->xywh':
        out = [(a + c) / 2, (b + d) / 2, c - a, d - b]
    elif conv == 'ltwh->ltrb':
        out = [a, b, a + c, b + d]
    elif conv == 'ltrb->ltwh':
        out = [a, b, c - a, d - b]
    else:
        raise ValueError(f'unsupported bbox conversion: {conv}')
    return torch.stack(out, dim=-1)
