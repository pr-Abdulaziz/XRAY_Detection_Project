# Prohibited Item Detection in X-ray Imagery: YOLO11s and RT-DETRv2 on PIDray

## Project Overview

This repository documents a comparative study of prohibited item detection in X-ray baggage imagery for security screening scenarios such as airports and public-entry checkpoints. The project uses the PIDray benchmark, a large-scale X-ray dataset for prohibited item analysis, and focuses on object detection rather than segmentation in the provided implementation materials.

The provided report compares two detector families under limited compute constraints: a lightweight one-stage CNN detector, YOLO11s, and a transformer-based detector, RT-DETRv2. Across the documented experiments, YOLO11s achieved the stronger overall detection results, while RT-DETRv2 provided a useful transformer-era comparison but underperformed on the reported setup.

## Motivation / Problem Statement

Automatic prohibited item detection in X-ray imagery is difficult because security scans are visually unlike ordinary natural images. Objects are frequently stacked, partially occluded, or deliberately concealed inside cluttered baggage. This creates several challenges:

- Cluttered scenes with multiple overlapping objects.
- Hidden or partially visible prohibited items.
- Visual ambiguity caused by translucent X-ray projections.
- Small-object localization difficulty in dense baggage layouts.
- The need for detectors that are accurate enough for screening while remaining computationally practical.

The report states that realistic X-ray inspection requires both reliable detection quality and reasonable processing efficiency, especially when considering real deployment constraints.

## Objectives

The project goals, as stated in the report and reflected in the notebooks, are:

- Study prohibited item detection in X-ray imagery using the PIDray dataset.
- Train and evaluate a lightweight `YOLO11s` detector.
- Train and evaluate an `RT-DETRv2` transformer-based detector.
- Compare both approaches against the PIDray benchmark context discussed in the report.
- Explore performance under limited computational resources rather than full-scale benchmark reproduction.

## Dataset

### PIDray

`PIDray` is an X-ray benchmark dataset for prohibited item analysis in realistic security inspection scenarios. According to the official PIDray repository and paper, it contains:

- `124,486` X-ray images
- `12` prohibited-item categories
- manually annotated instances
- benchmark tasks covering object detection, instance segmentation, and multi-label classification
- test subsets grouped by difficulty: `easy`, `hard`, and `hidden`

The report paraphrases the official scale as “more than 120,000 images,” which is consistent with the official benchmark description.

### Official PIDray Benchmark Statistics

From the official PIDray repository:

| Split / Difficulty | Image Count |
|---|---:|
| Train | 76,913 |
| Test Easy | 24,758 |
| Test Hard | 9,746 |
| Test Hidden | 13,069 |
| Total | 124,486 |

### What the Provided Project Materials Actually Use

The notebooks do **not** appear to operate on the full official PIDray benchmark. Instead, they point to local COCO-style files on Google Drive and use a smaller working setup:

- `train.json` with `18,261` training images
- `test.json` with `2,922` test images
- RT-DETRv2 additionally creates an internal validation split from the local training set:
  - `15,521` training images
  - `2,740` validation images
  - `2,922` test images

This is important to distinguish from the official PIDray benchmark scale. The report also states that “part of the dataset,” around `30,000` images, was used for experiments due to computational limits. The exact notebook counts sum to fewer than 30,000 images, so the safest interpretation is that the provided notebooks document a reduced local working split rather than the full benchmark.

### Annotation Type

The official PIDray benchmark includes both:

- bounding boxes
- segmentation masks

However, the provided report and notebooks mainly document **object detection** experiments. No segmentation training pipeline or segmentation results are explicitly implemented in the supplied notebooks.

### Class List

The RT-DETRv2 notebook explicitly shows the following 12 categories:

| Class ID | Class Name |
|---|---|
| 0 | Baton |
| 1 | Pliers |
| 2 | Hammer |
| 3 | Powerbank |
| 4 | Scissors |
| 5 | Wrench |
| 6 | Gun |
| 7 | Bullet |
| 8 | Sprayer |
| 9 | HandCuffs |
| 10 | Knife |
| 11 | Lighter |

Note: the report’s per-class table spells one category as `Handcuff`, while the notebook category list uses `HandCuffs`. This README preserves the notebook class names when referring to the implemented experiments.

### Official Links

- Official PIDray repository: <https://github.com/lutao2021/PIDray>
- Official paper DOI: <https://doi.org/10.1007/s11263-023-01855-1>
- Official dataset download page: <https://drive.google.com/drive/folders/1zvMIc1bqteRN9Z36hHYpoTGoZArsh4mE>
- Optional mirror/reference: <https://huggingface.co/datasets/Voxel51/PIDray>

### Usage / Licensing Note

The official PIDray materials are presented as a research benchmark and academic resource for prohibited item analysis. Readers should consult the official repository and paper directly for dataset access conditions and any usage restrictions rather than assuming broader licensing terms.

## Project Methodology

### 6.1 YOLO11s Pipeline

The report and `yolo_detection.ipynb` describe a conventional one-stage detection workflow built around Ultralytics YOLO.

Key steps documented in the notebook:

- PIDray annotations are read from COCO-style JSON files (`train.json` and `test.json`).
- A custom dataset-handling workflow links image IDs to annotation records.
- COCO bounding boxes in `[x_min, y_min, width, height]` format are converted into YOLO labels:
  - class ID
  - normalized `x_center`
  - normalized `y_center`
  - normalized width
  - normalized height
- Category IDs are remapped to contiguous class IDs from `0` to `11`.
- Label files are written per image in YOLO text format.
- A `data.yaml` file is created for training.
- The notebook indicates that images are symlinked when possible, with copy-based fallback if symlinks fail.

Model choice and detector structure:

- The selected detector is `YOLO11s` (`yolo11s.pt`), chosen because the report cites limited computational resources.
- The report summarizes YOLO11s using the standard components:
  - backbone for feature extraction
  - neck for multi-scale feature aggregation
  - detection head for box and class prediction
- The report explicitly mentions feature fusion via FPN/PAN-style processing and final prediction filtering through non-maximum suppression (NMS).

Implementation details visible in the notebook include:

- Ultralytics training API
- pretrained initialization from `yolo11s.pt`
- validation using the saved `best.pt` checkpoint
- post-training inspection through `results.csv`
- manual computation of F1-score from precision and recall columns

### 6.2 RT-DETRv2 Pipeline

The `RT-DETRv2.ipynb` notebook documents a transformer-based object detection workflow implemented with Hugging Face Transformers.

Documented pipeline components:

- Starting checkpoint: `PekingU/rtdetr_v2_r50vd`
- Image processor: `RTDetrImageProcessor.from_pretrained(checkpoint)`
- Model class: `RTDetrV2ForObjectDetection`
- Labels are remapped from original COCO category IDs to contiguous IDs for 12 PIDray classes.
- A custom `PIDrayRTDetrDataset` class reads COCO-style annotations and prepares examples for the image processor.

Preprocessing and augmentation shown in the notebook:

- `IMAGE_SIZE = 640`
- training transform uses Albumentations with:
  - `LongestMaxSize(640)`
  - padding to `640 x 640`
  - horizontal flip
- validation/test transform uses resizing and padding without augmentation

Bounding-box handling:

- COCO boxes are converted to VOC-style boxes for Albumentations.
- After transformation, boxes are converted back to COCO-style annotations for the Hugging Face processor.
- Invalid or degenerate boxes are filtered out.

Training and evaluation are run through `Trainer`:

- Hugging Face `Trainer`
- custom `collate_fn`
- custom `compute_metrics` using `torchmetrics.detection.MeanAveragePrecision`
- checkpointing by epoch
- best-model selection on validation `mAP`

The report describes RT-DETRv2 cautiously as a transformer-based detector rather than a classical contrastive VLM. That distinction is consistent with the notebook implementation.

## Experimental Setup

### YOLO11s

The report gives the following experiment settings for YOLO11s:

| Setting | Value |
|---|---|
| Framework | Ultralytics YOLO11 API |
| Model | YOLO11s |
| Image size | 640 x 640 |
| Optimizer | AdamW |
| Batch size | 64 |
| Epochs | 50 |
| Reported hardware | Google Colab, NVIDIA A100 |

Notebook evidence supports most of these settings directly:

- `epochs=50`
- `imgsz=640`
- `batch=64`
- `optimizer="AdamW"`
- training launched from Colab with CUDA enabled

One hardware detail differs between sources:

- the report states `A100`
- the YOLO notebook output shown in the saved execution log reports `NVIDIA L4`

This README therefore treats the YOLO hardware description conservatively: the experiment was run in Google Colab with GPU acceleration, but the exact GPU shown in the saved notebook output is not identical to the report summary.

The notebook also indicates a local working split of:

- `18,261` train images
- `2,922` validation/test images

### RT-DETRv2

The report and notebook together support the following setup:

| Setting | Value |
|---|---|
| Model family | RT-DETRv2 |
| Backbone / checkpoint | `PekingU/rtdetr_v2_r50vd` (ResNet-50 variant) |
| Image size | 640 x 640 |
| Optimizer | AdamW (`adamw_torch`) |
| Learning rate | `5e-5` |
| Weight decay | `1e-4` |
| Scheduler | cosine decay with warmup |
| Warmup ratio | `0.05` |
| Batch size | 16 per device in one training configuration |
| Epochs | report states 50; notebook shows 25 in resumed configurations |
| Mixed precision | FP16 enabled |
| Validation selection | 15% split from local training images |
| Hardware shown in notebook | Google Colab, NVIDIA A100-SXM4-80GB |

Important caveat: the report presents RT-DETRv2 as a 50-epoch experiment, but the saved notebook cells show resumed training configurations with `num_train_epochs=25`. This README therefore distinguishes between:

- the report-level experimental description (`50` epochs), and
- the notebook-level saved configuration that explicitly shows `25` epochs for resumed runs

The notebook also includes two resumed configurations with different batch sizes:

- `16` per device
- `64` per device

The report’s `16 per device` setting is therefore the clearest published configuration, while the notebook suggests later iteration or reconfiguration during experimentation.

## Results

### YOLO11s Results

The report and the YOLO notebook agree on the following peak results:

| Metric | Value |
|---|---:|
| mAP@50 | 73.779% |
| mAP@50-95 | 60.101% |
| Precision | 78.361% |
| Recall | 67.214% |
| F1-score | 72.361% |

The YOLO notebook states that the best result was achieved at `Epoch 50`.

### RT-DETRv2 Results

The report’s RT-DETRv2 table and the notebook evaluation output are aligned:

| Metric | Value |
|---|---:|
| mAP (overall) | 0.4766 |
| mAP@50 | 0.5801 |
| mAP@75 | 0.5143 |

### RT-DETRv2 Per-Class Results

| Class | mAP |
|---|---:|
| Baton | 0.5926 |
| Pliers | 0.8066 |
| Hammer | 0.5429 |
| Powerbank | 0.2260 |
| Scissors | 0.6346 |
| Wrench | 0.7206 |
| Gun | 0.3163 |
| Bullet | 0.3072 |
| Sprayer | 0.3658 |
| HandCuffs | 0.8813 |
| Knife | 0.1689 |
| Lighter | 0.1563 |

### Comparative Interpretation

Across the documented experiments, `YOLO11s` produced the stronger overall detection performance in this project. Using the report’s main localization metric, YOLO11s reached `60.101%` mAP@50-95, while RT-DETRv2 reached `0.4766` overall mAP on its reported test evaluation. The report further argues that YOLO11s handled cluttered X-ray scenes more effectively in this constrained setting, whereas RT-DETRv2 offered useful transformer-based behavior analysis but struggled more on difficult localization cases, especially for small or concealed items.

## Baseline Comparison

The report places both project models in the context of the official PIDray benchmark baselines from Zhang et al.

The benchmark context cited in the report includes:

| Method / Context | Reported Detection AP |
|---|---:|
| Cascade Mask R-CNN + Ours | 66.6% |
| DDOD | 63.6% |
| TOOD | 62.2% |
| YOLO11s in this project | 60.1% mAP@50-95 |
| RT-DETRv2 in this project | 47.7% overall mAP |

A careful reading of the report supports the following interpretation:

- Stronger benchmark methods, especially heavier two-stage detectors, outperform the project experiments.
- `YOLO11s` remains comparatively competitive given its lighter architecture and the project’s stated compute limits.
- `RT-DETRv2` provides a meaningful transformer-based comparison but does not match YOLO11s or the stronger PIDray baselines in the provided setup.
- The report attributes part of the performance gap to limited computational resources and reduced-scale experimentation rather than claiming strict like-for-like superiority or inferiority.

Because the notebooks appear to use a smaller local working split rather than the full official benchmark, these comparisons should be treated as informative rather than as strict benchmark reproductions.

## Key Findings

- `YOLO11s` was the stronger detector in the documented project setting.
- The lightweight one-stage YOLO pipeline produced competitive results relative to the benchmark context discussed in the report.
- `RT-DETRv2` offered a useful transformer-based comparison but underperformed YOLO11s on the reported experiments.
- The report suggests transformer-based detection was less effective here on small, cluttered, or hidden prohibited items.
- The implementation materials consistently reflect a reduced local PIDray working split rather than the full official benchmark.
- Compute constraints appear to have shaped both model choice and final performance.

## Limitations

- The provided materials document **detection-focused** experiments even though the report title also mentions segmentation.
- The notebooks appear to use a reduced local subset/split rather than the full official PIDray benchmark.
- The report’s “around 30,000 images” statement does not exactly match the image counts visible in the notebooks.
- Some training details differ between report and notebook outputs, especially for RT-DETRv2 epoch and batch-size settings.
- YOLO hardware reporting is not fully consistent: the report mentions A100, while the saved notebook output shows an L4 GPU.
- The current materials do not provide a clean, repository-ready training pipeline outside Colab notebooks.
- Reproducibility is partial because paths, checkpoints, and intermediate artifacts are tied to Google Drive / Colab-style execution.

## Reproducibility / Environment

The supplied notebooks indicate a Google Colab-centered workflow with Google Drive-backed storage. The paths currently point to directories such as:

- `/content/drive/My Drive/Deep_Learning_Project/Training_and_annotations`
- `/content/drive/MyDrive/Deep_Learning_Project/Training_and_annotations`

Explicit dependencies visible in the notebooks include:

- `torch`
- `torchvision`
- `ultralytics`
- `transformers`
- `datasets`
- `albumentations`
- `torchmetrics`
- `scikit-learn`
- `pycocotools`
- `timm`
- `tensorboard`
- `matplotlib`
- `PIL`

Framework-specific tooling shown in the notebooks:

- Ultralytics YOLO API for `YOLO11s`
- Hugging Face `Trainer` and `RTDetrImageProcessor` for `RT-DETRv2`
- Albumentations for preprocessing and augmentation
- TorchMetrics for mAP computation

Exact end-to-end reproducibility is currently incomplete from the provided materials alone. A clean public repository would need path normalization, explicit environment setup, and export of any missing utility code from the notebooks into standalone scripts or modules.

## Suggested Repository Scope

Based on the provided materials, this repository is best understood as a research project record for prohibited item detection in X-ray imagery. A conservative scope would include:

- the project report
- the two experiment notebooks
- documented results and figures exported from notebook runs
- any cleaned data-preparation utilities derived directly from the notebook logic
- trained checkpoints or evaluation artifacts if they are later added

The current evidence supports a notebook-first experimental workflow rather than a fully modular training codebase.

## References

1. Zhang, L., Jiang, L., Ji, R., & Fan, H. *PIDray: A large-scale X-ray benchmark for real-world prohibited item detection*. International Journal of Computer Vision. DOI: <https://doi.org/10.1007/s11263-023-01855-1>
2. Official PIDray repository: <https://github.com/lutao2021/PIDray>
3. Ultralytics YOLO documentation and pretrained YOLO11s model family: <https://docs.ultralytics.com/>
4. Hugging Face Transformers RT-DETR documentation: <https://huggingface.co/docs/transformers/>

## Citation / Acknowledgment

If you use PIDray in research derived from this repository, cite the original PIDray paper and acknowledge the dataset authors. This project builds on PIDray as the underlying benchmark and should not replace citation of the official dataset source.