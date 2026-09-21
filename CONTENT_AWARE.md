# Content-Aware Design Generation with Variable-Length Elements and Text

This is the onboarding note for the current project in this repo. Read it first. Written 2026-09-14.

Legend: ✅ verified in the paper or code · ⚠️ from a summary, re-check before relying on it · ❓ open

---

## 0. Background

This repo implements **VL-DE (Variable-Length Diffuse Everything)**: generator matching with an auxiliary process. Variable length is handled by three operations:
- **insert:** add mask tokens,
- **unmask:** replace a mask with a noisy value,
- **denoise:** refine values,

and modalities are combined with decoupled clocks, Diffuse Everything style. The repo also has a higher-order sampler based on autonomized Strang operator splitting, which costs 2 network evaluations per step.

The paper (Kevin Rojas, Molei Tao) was withdrawn from NeurIPS 2026 after scores of 2/3/3. The reviewer complaints this project must avoid repeating:
- The motivation felt artificial. The only mixed-length experiment used a synthetic dataset we built ourselves.
- Overclaiming: "no prior method", "independent lengths", "essentially solved".
- Non-standard metrics and too few baselines.
- Missing details: noise/insertion schedules (α_t, γ_t), what training and inference use.
- The sampler's second-order claim was never verified. There's also a concern that insertion and denoising clocks may need to be decoupled *during training* for the split sampler to be valid.

---

## 1. Problem

Given a **canvas image** (a poster or banner background with the foreground removed), generate a design: a **variable number N of elements**, each with
- a **category** (discrete: text types, logo, underlay, ...),
- a **bounding box** (continuous: cx, cy, w, h),
- and, for text elements, a **text string** (variable length).

The output is a structured design. Text reaches the image through a deterministic **renderer**, not by generating pixels.

### Motivation (use this framing in the paper)
- **Length only matters when conditioning.**
  - Unconditional: p(x) = p(N) p(x | N), and p(N) is just the empirical histogram, so modeling length is trivial.
  - Conditional: p(x | c) = p(N | c) p(x | N, c). Each canvas appears once, so **p(N | c) must be learned**.
- **Nested variable length.** The element space is [C] × ℝ⁴ × (text of any length). The per-element state is itself variable-length, so each element carries a FlexMDM-style text generator. This is where the framework's "arbitrary state space" generality is genuinely used.
- **Joint generation beats pipelines.**
  - Layout first, then text: the text overflows or underfills boxes.
  - Text first, then layout: the text ignores the canvas.
  - Count, box size, text length and image content all constrain each other.

### Claims (each needs one experiment)
1. Jointly generates count + layout + text from the canvas, with no length oracle, padding or histogram sampling.
2. Competitive on standard content-aware layout metrics (PKU/CGL).
3. Generated counts match the ground truth better than padding and empty-class baselines (new count metric).
4. Joint generation improves text/box consistency over two-stage pipelines (new text-fit metric).
5. (Secondary) The higher-order sampler reduces network evaluations; verify its order with log-log error vs. step size.

**Do not claim:** "first/only method able to…", "independent lengths", "solved". Present the theorems as propositions that make the recipe work.

---

## 2. Datasets

| Dataset | Use | Facts |
|---|---|---|
| **LayoutDETR ad banners** — https://github.com/salesforce/LayoutDETR | **Main task** (English, canvas + text) | ~7.7K English banners ⚠️. Inpainted backgrounds: "1x" (foreground only, for evaluation) and "3x" (extra random regions, for training) ⚠️. JSON with boxes, **OCR text strings** (noisy), 8 categories: header, pre-header, post-header, body text, disclaimer/footnote, button, callout, logo ⚠️. 9:1 split. Google Drive, 14.7 GB. No font or color labels. Dataset license not stated ❓. |
| **Crello** — HF `cyberagent/crello` (v5.1.0) | Second text dataset (larger, with typography) | 19,479 train templates ✅ (val/test ~1.85K/1.97K ⚠️). "Almost all design templates use English" ✅. Element fields: type, left, top, width, height, color, opacity, image, text, font, font_size, text_align, angle, bold, italic, text_color, line_height, letter_spacing, ... ✅. Canvas fields include canvas_width/height, `length` (number of elements) ✅. CDLA-Permissive-2.0. Renderer: https://github.com/CyberAgentAILab/cr-renderer (text rendering "far from perfect"). |
| **PKU PosterLayout** | Layout-only comparison with literature | Chinese e-commerce posters, but annotations are only boxes + categories (logo, text, underlay) ✅. RALF split 7,735 / 1,000 / 1,000 ✅; max 10 elements ✅. Annotated split (inpainted real posters, has ground truth) plus an unannotated split (real blank canvases, no ground truth). |
| **CGL** | Layout-only comparison with literature | E-commerce posters; logo, text, underlay, embellishment ✅. RALF split 48,544 / 6,002 / 6,002 ✅; max 10 ✅. |

Use **RALF's splits and preprocessed data** (inpainted canvases + saliency maps) for PKU/CGL. Other papers use different splits (e.g., Scan-and-Print).

**Crello statistics (computed 2026-09-19, train split, HF `cyberagent/crello`, cached at `$LAB/datasets/hf_cache`):**
- elements per template: mean 10.7, median 10, p90 17, max 50; 57% have <= 10, 95% <= 20, 99% <= 30;
- element types: SvgElement 48%, TextElement 40%, ImageElement 9%, ColoredBackground 2%, SvgMaskElement 2%;
- text elements per template: mean 4.2, median 4, p90 8, max 41; 99% of templates have at least one;
- text length: mean 20.8 chars / 3.5 words, median 12 chars / 2 words, p90 44 / 7, p99 137 / 22 (short: mostly headlines, dates, CTAs);
- canvas: 80% of templates have a full-canvas first element (ColoredBackground or ImageElement) that can serve as the
  background layer; 25% contain a ColoredBackground; aspect: 56% landscape, 32% portrait, 12% square; top formats
  Instagram Story / Instagram / Facebook / Facebook cover / Twitter / Facebook AD;
- per-element typography available: font (~250 classes), font_size, text_color, text_align, line_height, letter_spacing,
  bold/italic per line, capitalize, angle, opacity.

**Compute these statistics first (❓):**
- element-count distribution per dataset;
- fraction of text elements; text length per element;
- for Crello, whether a clean background layer exists to serve as the canvas;
- canvas aspect ratios;
- rotated elements.

---

## 3. Tasks

- **Main:** canvas → {N, (category, box, text)}. On PKU/CGL there's no text: canvas → {N, (category, box)}.
- **Standard constrained tasks (RALF protocol)** ✅: C → S+P, C+S → P, completion, refinement, relationship. With decoupled clocks per field (category / box / text), hold the given fields clean and generate the rest.

---

## 4. Baselines

### How published content-aware methods choose N
| Method | Venue | N handling | Code |
|---|---|---|---|
| CGL-GAN | IJCAI 2022 | 10 slots + no-object class ✅ (RALF code) | RALF repo |
| DS-GAN | CVPR 2023 | pad to max ✅ | RALF repo |
| ICVT | ACM MM 2022 | autoregressive, 10 steps + background class ✅ | RALF repo |
| LayoutDM (image-conditioned) | CVPR 2023 | PAD token, max 10 ✅ | RALF repo |
| RALF | CVPR 2024 | autoregressive, EOS token ✅ | https://github.com/CyberAgentAILab/RALF (checkpoints **and generated layouts released** ✅) |
| LayoutDiT | arXiv 2024 | empty class c=0 ✅ | project page ⚠️ |
| PosterLlama / PosterO | ECCV 2024 / CVPR 2025 | LLM, free-running generation | PosterO: https://github.com/theKinsley/PosterO-CVPR2025 |
| Scan-and-Print | arXiv 2025 | autoregressive ⚠️ | project page |
| UniLayDiff | arXiv Dec 2025 | not stated | none |

**No paper reports count accuracy.** Reference numbers (UniLayDiff, PKU annotated ⚠️): Occ 0.115, Rea 0.0114, UndL 0.999, UndS 0.996, Ove 0.0005, FID 3.15. The graphic metrics are saturated.

### Text-related prior work
- **MarkupDM** (ACM MM 2025): autoregressive fill-in-the-middle over markup + image tokens; Crello text completion. https://github.com/CyberAgentAILab/MarkupDM
- **OpenCOLE:** LLM pipeline. https://github.com/CyberAgentAILab/OpenCOLE
- **FlexDM** (CVPR 2023): masked-field prediction on Crello.

There's no standard benchmark for joint layout + text generation.

### Baselines we build
1. **Padding:** same backbone, fixed max N + empty class. This isolates the value of insertion.
2. **N-predictor:** a network predicts N from the canvas, then a fixed-length model.
3. **Histogram-N:** N from the training histogram, ignoring the canvas (strawman).
4. **Two-stage:** layout model → text model given boxes + canvas.
5. **LLM in-context** (PosterO/LayoutPrompter style).
6. **MarkupDM** on Crello text completion.

---

## 5. Metrics

- **Layout:** copy RALF's `image2layout/train/helpers/metric.py` (~740 lines; needs numpy, torch, cv2, prdc, pytorch-fid, timm; Apache-2.0, keep attribution). Metrics: occlusion, readability, underlay effectiveness (loose/strict), overlay, alignment, validity. FID uses RALF's feature-extractor weights from their cache zip.
- **Count (new):**
  - per-image |N̂ − N| on annotated splits;
  - generated vs. ground-truth N distribution;
  - per-category count error.
- **Text (new):**
  - fluency (perplexity under a fixed pretrained language model);
  - diversity (distinct-n);
  - CLIP score of renders (secondary);
  - **text fit** (overflow rate / fill ratio from font metrics at an auto-fit size).

Keep the main metrics independent of the renderer.

---

## 6. Rendering and typography

- **Crello:** generate typography (font = discrete; size and color = continuous) as extra per-element attributes, or use heuristics. Render with cr-renderer.
- **LayoutDETR banners:** no typography labels, so use heuristics: fixed font, size auto-fit to the box, color with good contrast. Its repo mentions rendering code ⚠️.

---

## 7. Code plan

### What this repo already has
- Layout training: `layout_training.py`, Hydra configs in `conf/`.
- PubLayNet/RICO via LayoutFlow's h5 release: `custom_datasets/layoutflow_h5.py`, max 20 elements.
- LayoutFlow evaluation pipeline in `eval/layout/`.
- MMDiT/transformer models in `models/`.
- Interpolants: `multimodal_interpolant.py`, `multimodal_interpolant_both_var.py`, etc.

See `LAYOUT.md` and `RICO.md` for run history. RICO unconditional FID 2.30 was reproduced for LayoutFlow.

### Known issue
`LayoutFlow/` is a gitlink (mode 160000) with no `.gitmodules`, so it clones empty. The data and weights are expected under `/workspace/LayoutFlow-data` (see `LAYOUT.md`).

### To add
1. **Canvas conditioning:** image encoder + cross-attention in the model; saliency map input as in RALF.
2. **Data loaders** to a common schema: `canvas`, `saliency` (optional), `elements: [{category, box (cx, cy, w, h), text (str | None), typography (optional)}]`. Sources:
   - PKU/CGL from RALF's release;
   - LayoutDETR banners;
   - Crello.
3. **Metrics:** copy RALF's `metric.py` + FID weights; add count and text metrics.
4. **Nested text generation:** per-element text sequences with their own insert / unmask / denoise. Initialize text from a small pretrained masked-diffusion language model (MDLM checkpoints ❓), because the data is small.
5. **Baselines:** run the RALF repo **unchanged** in its own Docker environment. It's an old stack: torch 1.13.1 / cu117, Python <3.11, last commit July 2024 ✅.

---

## 8. First steps

- [ ] **Validate the evaluation pipeline.** In RALF's Docker environment, download their cache zip and recompute metrics on their released generated layouts. The numbers should match the RALF paper.
- [ ] **Go/no-go.** Compute count error on RALF's released outputs; train or evaluate a padding model (LayoutDM in the RALF harness, or LayoutDiT) and compare. If padding models already match ground-truth counts, rethink claim 3 before building more.
- [ ] **Dataset statistics** for LayoutDETR banners and Crello (section 2).
- [ ] **Stage 1:** layout only (category + box) with canvas conditioning on PKU/CGL + banners, with the count metric. *This is a fallback paper on its own.*
- [ ] **Stage 2:** per-element text; two-stage and LLM baselines; text-fit metric.
- [ ] **Sampler:** order verification and step-count curves.

## 9. Risks

- **Scrutiny of the new task:** reviewers will question how it's defined. Mitigate with public data, standard metrics per component, and strong simple baselines.
- **Erasure traces** in annotated splits may leak element positions and counts. Also report unannotated metrics.
- **Noisy OCR text** in the banners.
- **Weak language quality** from small data; use pretrained initialization.
- **Padding is cheap at max 10 elements**, so the count advantage must be shown empirically.

## 10. References

- RALF https://arxiv.org/abs/2311.13602
- PosterLayout https://arxiv.org/abs/2303.15937
- LayoutDiT https://arxiv.org/abs/2407.15233
- PosterO https://arxiv.org/abs/2505.07843
- Scan-and-Print https://arxiv.org/abs/2505.20649
- UniLayDiff https://arxiv.org/abs/2512.08897
- LayoutDETR https://arxiv.org/abs/2212.09877
- MarkupDM https://arxiv.org/abs/2409.19051
- FlexDM https://arxiv.org/abs/2303.18248
- COLE https://arxiv.org/abs/2311.16974
- LayoutFlow https://arxiv.org/abs/2403.18187
- Crello https://huggingface.co/datasets/cyberagent/crello
