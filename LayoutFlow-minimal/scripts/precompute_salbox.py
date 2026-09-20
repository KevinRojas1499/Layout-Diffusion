'''
Saliency bounding box per canvas (LayoutDiT / LayoutGD's `sal_box` token): threshold RALF's saliency map at 25/255,
take the bounding rectangle of the largest connected component, normalise to cx, cy, w, h in [0, 1].
Writes {feats_dir}/{split}_salbox.pt = {'id': [...], 'salbox': (N, 4) float}. Run in the RALF venv (needs cv2).
  usage: python scripts/precompute_salbox.py <cgl parquet dir> <feats dir>
'''
import glob, io, sys
import numpy as np, torch, cv2
import pyarrow.parquet as pq
from PIL import Image

data_path, out_dir = sys.argv[1], sys.argv[2]
for split in ['train', 'val', 'test', 'with_no_annotations_test']:
    files = sorted(glob.glob(f'{data_path}/{split}-*.parquet'))
    ids, boxes = [], []
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=256, columns=['id', 'saliency']):
            d = batch.to_pydict()
            for i, sal in zip(d['id'], d['saliency']):
                img = np.array(Image.open(io.BytesIO(sal['bytes'])).convert('L'))
                H, W = img.shape
                _, th = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    box = [(x + w / 2) / W, (y + h / 2) / H, w / W, h / H]
                else:
                    box = [0.0, 0.0, 0.0, 0.0]
                ids.append(i); boxes.append(box)
    torch.save({'id': ids, 'salbox': torch.tensor(boxes, dtype=torch.float32)}, f'{out_dir}/{split}_salbox.pt')
    b = torch.tensor(boxes); print(split, len(ids), 'empty', int((b[:, 2] == 0).sum()), 'mean cxcywh', b.mean(0).tolist())
