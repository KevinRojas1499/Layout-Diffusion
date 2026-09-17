import torch
import seaborn as sns
from PIL import Image, ImageDraw, ImageOps


def gen_colors(num_colors):
    palette = sns.color_palette('husl', num_colors)
    return [[int(c * 255) for c in rgb] for rgb in palette]


def draw_layout(layout, features, num_colors=6):
    '''layout (S, 4) in xywh format, values in [0, 1]; features (S,) category ids (1-indexed, 0=pad).'''
    colors = gen_colors(num_colors)
    img = Image.new('RGB', (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA')

    layout = torch.clip(layout, 0, 1)
    box = torch.stack([
        layout[:, 0] - layout[:, 2] / 2, layout[:, 1] - layout[:, 3] / 2,
        layout[:, 0] + layout[:, 2] / 2, layout[:, 1] + layout[:, 3] / 2,
    ], dim=1)
    box = 255 * torch.clamp(box, 0, 1)

    for i in range(len(layout)):
        cat = features[i] - 1
        if cat < 0:
            continue
        col = colors[cat] if cat < len(colors) else [0, 0, 0]
        draw.rectangle(box[i].tolist(), outline=tuple(col) + (200,), fill=tuple(col) + (64,), width=2)

    return ImageOps.expand(img, border=2)
