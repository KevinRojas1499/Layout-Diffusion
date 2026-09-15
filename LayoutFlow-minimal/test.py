'''
Unconditional-generation eval: FID / alignment / overlap (+ optional mIoU), matching
upstream's `task=uncond` path. Trimmed from upstream test.py: dropped the other four
conditioning tasks (cat_cond/size_cond/elem_compl/refinement) and the
load-generated-bboxes-from-file shortcut -- this always generates fresh samples from
a checkpoint. Keeps the RICO continuous-bbox substitution (`rico_test.pt`) since
that's what upstream's own reported numbers were computed against.
'''
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from collections import OrderedDict as OD
import torch
import numpy as np
from pytorch_fid.fid_score import calculate_frechet_distance
from tqdm import tqdm
import rootutils

from src.fid import LayoutNet
from src.metrics import compute_alignment, compute_overlap, compute_overlap_ignore_bg, compute_maximum_iou
from src.utils import convert_bbox
from src.visualize import draw_layout

rootutils.setup_root(__file__, indicator='.git', pythonpath=True)


def get_data(dataloader):
    bbox, label, mask_bb, for_miou = [], [], [], []
    for batch in dataloader:
        bbox.append(batch['bbox'])
        label.append(batch['type'])
        mask_bb.append(batch['mask'].squeeze())
        for i, L in enumerate(batch['length']):
            for_miou.append([batch['bbox'][i, :L].cpu(), batch['type'][i][:L].cpu()])
    bbox, label, mask_bb = torch.cat(bbox), torch.cat(label), torch.cat(mask_bb)
    return bbox, convert_bbox(bbox, 'xywh->ltrb'), label, mask_bb, for_miou


@hydra.main(version_base=None, config_path='conf', config_name='test.yaml')
def main(cfg: DictConfig):
    with torch.no_grad():
        device = torch.device(cfg.device)
        val_loader = instantiate(cfg.dataset, dataset={'split': 'validation'}, shuffle=False, batch_size=1024)
        test_loader = instantiate(cfg.dataset, dataset={'split': 'test'}, shuffle=False, batch_size=1024)
        is_rico = cfg.dataset_name == 'RICO'

        num_classes = 25 if is_rico else 5
        fid_model = LayoutNet(num_classes, 20).to(device)
        state_dict = torch.load(f'{cfg.pretrained_dir}/fid_{cfg.dataset_name.lower()}.pth.tar', map_location='cpu')
        fid_model.load_state_dict(OD([(k.split('module.')[-1], v) for k, v in state_dict.items()]))
        fid_model.requires_grad_(False)
        fid_model.eval()

        # RICO's dataloader only has discretized bboxes; the paper's reported numbers
        # use the continuous ones from this separate file.
        if is_rico:
            data = torch.load(f'{cfg.pretrained_dir}/rico_test.pt')
            ltrb_bbox_test, label_test, mask_bb_test = data[..., :4], data[..., 4].long(), data[..., 5].bool()
            bbox_test = convert_bbox(ltrb_bbox_test, 'ltrb->xywh')
            gt_for_miou = [[bb[:m.sum()], lab[:m.sum()]] for bb, lab, m in zip(bbox_test, label_test, mask_bb_test)]
        else:
            bbox_test, ltrb_bbox_test, label_test, mask_bb_test, gt_for_miou = get_data(test_loader)
        _, ltrb_bbox_val, label_val, mask_bb_val, _ = get_data(val_loader)

        feats_real = fid_model.extract_features(ltrb_bbox_test.to(device), label_test.to(device), (~mask_bb_test).to(device))
        mu1, cov1 = feats_real.cpu().numpy().mean(0), np.cov(feats_real.cpu().numpy(), rowvar=False)
        feats_val = fid_model.extract_features(ltrb_bbox_val.to(device), label_val.to(device), (~mask_bb_val).to(device))
        mu_val, cov_val = feats_val.cpu().numpy().mean(0), np.cov(feats_val.cpu().numpy(), rowvar=False)
        print(f'[sanity check, val vs test] FID: {calculate_frechet_distance(mu1, cov1, mu_val, cov_val):.4f}')

        print('Loading model...')
        model = hydra.utils.get_class(cfg.model._target_).load_from_checkpoint(cfg.checkpoint, map_location=device)
        model.inference_steps = cfg.inference_steps
        model = model.to(device).eval()

        length_dist = torch.tensor(cfg.length_dist_by_dataset[cfg.dataset_name])
        fids, aligns, overlaps, mious = [], [], [], []
        for _ in range(cfg.multirun_n):
            bbox, label, pad_mask, for_miou = [], [], [], []
            for batch in tqdm(test_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                batch['length'] = torch.multinomial(length_dist, num_samples=len(batch['length']), replacement=True)
                geom_pred, cat_pred = model.inference(batch)
                bbox.append(geom_pred)
                label.append(cat_pred)
                m = torch.zeros(geom_pred.shape[:2], device=device, dtype=bool)
                for i, L in enumerate(batch['length']):
                    m[i, :L] = True
                pad_mask.append(m)
                if cfg.small:
                    break
            bbox = torch.cat(bbox)
            ltrb_bbox, label, pad_mask = convert_bbox(bbox, 'xywh->ltrb'), torch.cat(label), torch.cat(pad_mask)
            for bb, cat, mask in zip(bbox, label, pad_mask):
                L = mask.sum()
                for_miou.append([bb[:L].cpu(), cat[:L].cpu()])

            # 2000 samples is the convention other layout-generation papers use
            bbox, ltrb_bbox, label, pad_mask = bbox[:2000], ltrb_bbox[:2000], label[:2000], pad_mask[:2000]
            for_miou = for_miou[:2000]

            feats_fake = fid_model.extract_features(ltrb_bbox, label, ~pad_mask)
            mu2, cov2 = feats_fake.cpu().numpy().mean(0), np.cov(feats_fake.cpu().numpy(), rowvar=False)
            fid = calculate_frechet_distance(mu1, cov1, mu2, cov2)
            align = compute_alignment(bbox.cpu(), pad_mask.cpu())
            if is_rico:
                overlap = compute_overlap_ignore_bg(bbox.cpu(), label.cpu(), pad_mask.cpu())
            else:
                overlap = compute_overlap(bbox.cpu(), pad_mask.cpu())
            miou = compute_maximum_iou(gt_for_miou, for_miou) if cfg.calc_miou else -1
            miou_str = f' | mIoU {miou:.4f}' if cfg.calc_miou else ''
            print(f'FID {fid:.4f} | Alignment {100*align:.4f} | Overlap {overlap:.4f}{miou_str}')
            fids.append(fid); aligns.append(align); overlaps.append(overlap); mious.append(miou)

        if cfg.visualize:
            for i in range(min(20, len(pad_mask))):
                L = pad_mask[i].sum()
                draw_layout(bbox[i, :L].cpu(), label[i, :L].cpu(), num_colors=num_classes).save(f'./vis/{i}.png')

        if cfg.multirun_n > 1:
            fids, aligns, overlaps = np.array(fids), np.array(aligns), np.array(overlaps)
            print(f'Mean over {cfg.multirun_n} runs: FID {fids.mean():.4f} (+/- {fids.std():.4f}) | '
                  f'Alignment {100*aligns.mean():.4f} (+/- {100*aligns.std():.4f}) | '
                  f'Overlap {overlaps.mean():.4f} (+/- {overlaps.std():.4f})')


if __name__ == '__main__':
    main()
