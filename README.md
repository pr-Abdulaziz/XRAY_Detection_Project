# Prohibited Item Detection in X-ray Imagery: YOLO11s and RT-DETRv2 on PIDray

## Project Overview

This repository documents a comparative study of prohibited item detection in X-ray baggage imagery for security screening scenarios such as airports and public-entry checkpoints. The primary project sources in this repository are the report (`Report.pdf`) and two Colab-style notebooks (`yolo_detection.ipynb` and `RT-DETRv2.ipynb`).

The available materials mainly document **object detection** experiments on PIDray. Although the report title refers to both detection and segmentation, the notebooks and reported results in this repository are detection-focused and do not present a repository-ready segmentation training pipeline or segmentation results.

Across the documented experiments, the report presents two detector families:

- `YOLO11s`, used here as a lightweight one-stage CNN detector.
- `RT-DETRv2`, used here as a transformer-based object detector for comparison.

In the reported setup, `YOLO11s` achieved the stronger overall results.

## Motivation / Problem Statement

Automatic prohibited item detection in X-ray imagery is difficult because security scans differ substantially from natural images. As the report explains, X-ray baggage images often contain overlapping objects, cluttered contents, partial occlusion, and intentionally concealed items.

These conditions create several practical challenges:

- cluttered scenes with multiple overlapping objects
- hidden or partially visible prohibited items
- visual ambiguity caused by translucent X-ray projections
- small-object localization difficulty in dense baggage layouts
- the need for models that balance detection quality with practical compute requirements

The report frames the task as requiring both reliable detection accuracy and computational efficiency for realistic screening workflows.

## Objectives

Based on the report and notebooks, the project objectives are:

- study prohibited item detection in X-ray imagery using the PIDray dataset
- train and evaluate a lightweight `YOLO11s` detector
- train and evaluate an `RT-DETRv2` transformer-based object detector
- compare both models against the PIDray benchmark context discussed in the report
- explore performance under limited computational resources rather than full-scale benchmark reproduction

## Dataset

### Official PIDray Benchmark

According to the official PIDray paper and repository, `PIDray` is a large-scale X-ray benchmark for prohibited item analysis with:

- `124,486` X-ray images
- `12` prohibited-item categories
- manually annotated instances
- benchmark tasks covering object detection, instance segmentation, and multi-label classification
- test subsets grouped by difficulty: `easy`, `hard`, and `hidden`

The report paraphrases this scale as “more than 120,000 images,” which is consistent with the official benchmark sources.

### Official Dataset Summary

| Split / Difficulty | Image Count |
| --- | ---: |
| Train | 76,913 |
| Test Easy | 24,758 |
| Test Hard | 9,746 |
| Test Hidden | 13,069 |
| Total | 124,486 |

### Working Split Used in the Project Materials

The notebooks do **not** appear to use the full official PIDray benchmark directly. Instead, the notebook paths point to local Google Drive files (`train.json`, `test.json`, and image folders), and the saved notebook outputs indicate a smaller working setup.

| Source | Split | Image Count | Notes |
| --- | --- | ---: | --- |
| `yolo_detection.ipynb` | Train | 18,261 | Notebook output shows YOLO cache built from `18,261` training images |
| `yolo_detection.ipynb` | Validation/Test | 2,922 | Notebook output shows YOLO validation cache built from `2,922` images |
| `RT-DETRv2.ipynb` | Train | 15,521 | Notebook indicates a `15%` validation split from local training data |
| `RT-DETRv2.ipynb` | Validation | 2,740 | Created from the local training set via `train_test_split(test_size=0.15)` |
| `RT-DETRv2.ipynb` | Test | 2,922 | Loaded from local `test.json` |

The report states that the experiments used only part of the dataset, around `30,000` images, due to computational limits. The notebook-visible counts do not exactly total `30,000`, so the safest interpretation is that the repository documents a reduced local experimental workflow rather than a full official PIDray benchmark reproduction.

### Annotation Scope

The official PIDray benchmark supports object detection, instance segmentation, and multi-label classification. In contrast, the provided report and notebooks mainly document **object detection**. No explicit segmentation training pipeline or segmentation result table is implemented in the supplied notebooks.

### Class List Shown in the RT-DETRv2 Notebook

| Class ID | Class Name |
| --- | --- |
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

The report’s per-class result table spells one category as `Handcuff`, while the RT-DETRv2 notebook uses `HandCuffs`. This README preserves the notebook spelling when referring to notebook class metadata and the report spelling when referring to the report table.

## Methodology

### YOLO11s Pipeline

The report and `yolo_detection.ipynb` describe a one-stage detection workflow built around Ultralytics YOLO.

The notebook indicates the following data-preparation steps:

- read PIDray annotations from local COCO-style JSON files (`train.json` and `test.json`)
- map image IDs to annotation records
- remap category IDs to contiguous class IDs from `0` to `11`
- convert COCO bounding boxes from `[x_min, y_min, width, height]` into YOLO label format with normalized center coordinates and box size
- write one YOLO label file per image
- create a `data.yaml` file for Ultralytics training
- symlink images when possible, with copy fallback if symlinks fail

The report describes `YOLO11s` as a lightweight CNN-based detector selected under limited compute constraints. It summarizes the model in standard terms as a backbone, neck, and detection head, with multi-scale feature aggregation and non-maximum suppression for final predictions.

The notebook also shows:

- pretrained initialization from `yolo11s.pt`
- training through the Ultralytics API
- post-training validation with the saved `best.pt` checkpoint
- post-training analysis through `results.csv`
- manual F1-score calculation from precision and recall columns

### RT-DETRv2 Pipeline

The `RT-DETRv2.ipynb` notebook documents a transformer-based object detection pipeline implemented with Hugging Face Transformers.

The notebook indicates the following core components:

- checkpoint: `PekingU/rtdetr_v2_r50vd`
- image processor: `RTDetrImageProcessor.from_pretrained(checkpoint)`
- model class: `RTDetrV2ForObjectDetection`
- custom dataset class: `PIDrayRTDetrDataset`
- label remapping from original COCO category IDs to contiguous IDs for 12 PIDray classes

The notebook preprocessing pipeline uses `IMAGE_SIZE = 640` and Albumentations-based transforms. The saved cells show:

- `LongestMaxSize(640)`
- padding to `640 x 640`
- horizontal flip
- additional geometric and photometric augmentation in training cells

For box handling, the notebook converts COCO boxes to VOC-style boxes for augmentation, filters invalid or degenerate boxes, and converts boxes back to COCO-style annotations before passing them to the Hugging Face image processor.

Training and evaluation are implemented with Hugging Face `Trainer`, including:

- custom `collate_fn`
- custom `compute_metrics`
- `torchmetrics.detection.MeanAveragePrecision`
- checkpointing by epoch
- best-model selection based on validation `mAP`

The report refers to RT-DETRv2 using VLM-style wording in places, but the actual implemented model in the notebook is more accurately described as a transformer-based object detector used here for comparison rather than a classical vision-language model.

## Experimental Setup

### YOLO11s

The report states the following main YOLO11s configuration:

| Setting | Value |
| --- | --- |
| Framework | Ultralytics YOLO API |
| Model | `YOLO11s` |
| Initialization | `yolo11s.pt` |
| Image size | `640 x 640` |
| Optimizer | `AdamW` |
| Batch size | `64` |
| Epochs | `50` |
| Reported hardware | Google Colab, `NVIDIA A100` |

The notebook directly supports most of these settings with `epochs=50`, `imgsz=640`, `batch=64`, and `optimizer="AdamW"`.

One hardware detail is inconsistent across sources:

- the report states `A100`
- the saved YOLO notebook output shows `NVIDIA L4`

The conservative conclusion is that the experiment was run in Google Colab with GPU acceleration, but the exact GPU reported in the saved materials is not fully consistent.

### RT-DETRv2

The report and notebook together support the following RT-DETRv2 setup:

| Setting | Value |
| --- | --- |
| Model family | `RT-DETRv2` |
| Checkpoint | [`PekingU/rtdetr_v2_r50vd`](https://huggingface.co/PekingU/rtdetr_v2_r50vd) |
| Backbone context | ResNet-50 variant |
| Image size | `640 x 640` |
| Optimizer | `adamw_torch` / AdamW |
| Learning rate | `5e-5` |
| Weight decay | `1e-4` |
| Scheduler | cosine decay with warmup |
| Warmup ratio | `0.05` |
| Mixed precision | FP16 enabled |
| Validation split | `15%` from the local training images |
| Hardware shown in notebook | Google Colab, `NVIDIA A100-SXM4-80GB` |

The batch-size and epoch reporting requires careful qualification:

- the report states `batch size = 16 per device` and `epochs = 50`
- the saved notebook includes resumed configurations with `num_train_epochs=25`
- the saved notebook also shows resumed runs with both `per_device_train_batch_size=16` and `per_device_train_batch_size=64`

Accordingly, the README distinguishes between the report-level experiment description and the specific resumed training configurations visible in the notebook.

## Results

### YOLO11s Results

The report’s YOLO11 performance table matches the saved notebook post-training summary.

| Metric | Value |
| --- | ---: |
| mAP@50 | 73.779% |
| mAP@50-95 | 60.101% |
| Precision | 78.361% |
| Recall | 67.214% |
| F1-Score | 72.361% |

The YOLO notebook indicates that the peak result was achieved at `Epoch 50`.

### RT-DETRv2 Results

The report’s RT-DETRv2 result table aligns with the notebook evaluation output.

| Metric | Value |
| --- | ---: |
| mAP (overall) | 0.4766 |
| mAP@50 | 0.5801 |
| mAP@75 | 0.5143 |

### RT-DETRv2 Per-Class Results

| Class | mAP |
| --- | ---: |
| Baton | 0.5926 |
| Pliers | 0.8066 |
| Hammer | 0.5429 |
| Powerbank | 0.2260 |
| Scissors | 0.6346 |
| Wrench | 0.7206 |
| Gun | 0.3163 |
| Bullet | 0.3072 |
| Sprayer | 0.3658 |
| Handcuff / HandCuffs | 0.8813 |
| Knife | 0.1689 |
| Lighter | 0.1563 |

The category name above is written as `Handcuff` in the report table and `HandCuffs` in the notebook label list; the score is the same across both sources.

### Comparative Interpretation

Across the documented experiments, `YOLO11s` produced the stronger overall detection performance in this project. Using the report’s main localization metric, YOLO11s reached `60.101%` mAP@50-95, while RT-DETRv2 reached `0.4766` overall mAP on its reported test evaluation.

The report further argues that YOLO11s handled cluttered X-ray scenes more effectively in this constrained setting, while RT-DETRv2 provided a useful transformer-based comparison but struggled more on difficult localization cases, especially for small or concealed items.

## Baseline Comparison

The report places both project models in the context of the PIDray benchmark baselines from Zhang et al.

| Method / Context | Reported Detection AP |
| --- | ---: |
| Cascade Mask R-CNN + Ours | 66.6% |
| DDOD | 63.6% |
| TOOD | 62.2% |
| YOLO11s in this project | 60.1% mAP@50-95 |
| RT-DETRv2 in this project | 47.7% overall mAP |

These comparisons should be interpreted cautiously. The project notebooks appear to use a reduced local working split rather than the full official benchmark, so this repository documents an informative comparison rather than a strict benchmark reproduction.

## Key Findings

- `YOLO11s` was the stronger detector in the documented project setting.
- The report presents `YOLO11s` as comparatively competitive given its lighter architecture and the project’s stated compute limits.
- `RT-DETRv2` provided a useful transformer-based comparison but underperformed YOLO11s in the reported setup.
- The available repository materials are detection-focused even though the broader PIDray benchmark also supports segmentation and multi-label classification.
- The notebooks reflect a reduced local PIDray workflow rather than a clean reproduction of the full official benchmark.

## Limitations

- The report title mentions segmentation, but the repository materials mainly document object detection experiments.
- The notebooks appear to use a reduced local subset or split rather than the full official PIDray benchmark.
- The report’s “around 30,000 images” description does not exactly match the image counts visible in the saved notebook outputs.
- Some configuration details differ between the report and notebook outputs, especially RT-DETRv2 epochs and batch size.
- YOLO hardware reporting is not fully consistent: the report states `A100`, while the saved notebook output shows `NVIDIA L4`.
- The repository currently reflects a notebook-first Colab workflow rather than a polished modular training codebase.
- Reproducibility is partial because paths, checkpoints, and artifacts are tied to Google Drive and Colab-style execution.

## Reproducibility / Environment

The notebooks indicate a Google Colab-centered workflow with Google Drive-backed storage. Example paths visible in the saved notebooks include:

- `/content/drive/My Drive/Deep_Learning_Project/Training_and_annotations`
- `/content/drive/MyDrive/Deep_Learning_Project/Training_and_annotations`

Dependencies explicitly visible in the notebooks include:

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

Framework-specific tooling shown in the notebooks includes:

- Ultralytics YOLO for `YOLO11s`
- Hugging Face `Trainer` and `RTDetrImageProcessor` for `RT-DETRv2`
- Albumentations for preprocessing and augmentation
- TorchMetrics for mAP computation

Exact end-to-end reproducibility is incomplete from the repository contents alone. A cleaner public release would require path normalization, explicit environment setup, and extraction of notebook logic into standalone scripts or modules.

## Suggested Repository Scope

Based on the provided materials, this repository is best understood as a research project record for prohibited item detection in X-ray imagery. A conservative public scope would include:

- the project report
- the two experiment notebooks
- documented result tables and exported figures from notebook runs
- cleaned data-preparation utilities only if they are directly derived from the notebook logic
- checkpoints or evaluation artifacts if they are later added

The current evidence supports a notebook-first experimental workflow rather than a fully modular training repository.

## References

1. Zhang, L., Jiang, L., Ji, R., & Fan, H. *PIDray: A large-scale X-ray benchmark for real-world prohibited item detection*. International Journal of Computer Vision. DOI: <https://doi.org/10.1007/s11263-023-01855-1>
2. Official PIDray repository: <https://github.com/lutao2021/PIDray>
3. Official PIDray dataset download folder: <https://drive.google.com/drive/folders/1zvMIc1bqteRN9Z36hHYpoTGoZArsh4mE>
4. Optional PIDray mirror on Hugging Face: <https://huggingface.co/datasets/Voxel51/PIDray>
5. Ultralytics YOLO documentation: <https://docs.ultralytics.com/>
6. RT-DETRv2 checkpoint used in the notebook: <https://huggingface.co/PekingU/rtdetr_v2_r50vd>
7. Hugging Face Transformers documentation for RT-DETR: <https://huggingface.co/docs/transformers/>

## Citation / Acknowledgment

If you use PIDray in work derived from this repository, cite the original PIDray paper and acknowledge the dataset authors. This project builds on PIDray as the underlying benchmark and should not replace citation of the official dataset source.
