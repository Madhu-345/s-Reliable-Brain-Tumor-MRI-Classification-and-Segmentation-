# Dataset Download and Preparation

This directory contains the dataset loaders and documentation used by the
Uncertainty-Aware Multi-Task Evidential Learning framework for brain tumor
MRI classification and segmentation.

The actual medical imaging datasets are NOT included in this repository.
Users must download the datasets from their respective sources and place
them in the directory structure described below.

## 1. Classification Datasets

The classification pipeline supports four classes:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

### Dataset 1 — Brain Tumor MRI Dataset

Current release: approximately 7,200 MRI images.

Download:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Dataset DOI:
https://doi.org/10.34740/KAGGLE/DSV/2645886

Expected structure:

data/
└── classification/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/

If the downloaded dataset uses `notumor`, rename it to `no_tumor` or
update the dataset configuration.

### Dataset version note

An older release contains 7,023 images (5,712 training + 1,311 testing).
Do not mix the old and current releases within one experiment.

### Dataset 2 — Brain Tumor Classification (MRI)

Approximately 3,264 MRI images across four classes.

Download:
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

Original GitHub repository:
https://github.com/sartajbhuvaji/brain-tumor-classification-dataset

Expected structure:

data/
└── classification_dataset_2/
    ├── Training/
    │   ├── glioma_tumor/
    │   ├── meningioma_tumor/
    │   ├── no_tumor/
    │   └── pituitary_tumor/
    └── Testing/
        ├── glioma_tumor/
        ├── meningioma_tumor/
        ├── no_tumor/
        └── pituitary_tumor/

### Dataset 3 — 1,311-image classification set

The 1,311-image test set associated with the older Brain Tumor MRI Dataset
contains:

- Glioma: 300
- Meningioma: 306
- No tumor: 405
- Pituitary: 300

Total: 1,311.

Download:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Expected structure:

data/
└── classification_dataset_3/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/

## 2. BraTS Segmentation Dataset

The segmentation pipeline uses four MRI modalities:

- T1
- T1ce
- T2
- FLAIR

and three derived tumor regions:

- WT — Whole Tumor
- TC — Tumor Core
- ET — Enhancing Tumor

### Download

Current BraTS data access:
https://www.synapse.org/Synapse:syn51156910

Dataset access may require a Synapse account, registration, acceptance
of applicable terms, and authorization before download.

Do not redistribute BraTS data through this repository.

### Expected structure

data/
└── BraTS2020/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_t1.nii
    │   ├── BraTS20_Training_001_t1ce.nii
    │   ├── BraTS20_Training_001_t2.nii
    │   ├── BraTS20_Training_001_flair.nii
    │   └── BraTS20_Training_001_seg.nii
    ├── BraTS20_Training_002/
    └── ...

The loader also supports `.nii.gz`.

## 3. BraTS Regions

The loader converts the BraTS annotation into three binary regions:

WT = labels {1, 2, 4}
TC = labels {1, 4}
ET = label {4}

Result:

Image: [4, H, W]
    Channel 0 -> T1
    Channel 1 -> T1ce
    Channel 2 -> T2
    Channel 3 -> FLAIR

Mask: [3, H, W]
    Channel 0 -> WT
    Channel 1 -> TC
    Channel 2 -> ET

## 4. Dataset Loaders

Classification loader:

data/classification_dataset.py

BraTS loader:

data/brats_dataset.py

Package import:

```python
from data import (
    BrainTumorClassificationDataset,
    BraTSDataset
)
```

## 5. Dependencies

Install:

```bash
pip install numpy pillow nibabel torch torchvision
```

or:

```bash
pip install -r requirements.txt
```

## 6. Quick Verification

Classification:

```python
from data.classification_dataset import BrainTumorClassificationDataset

dataset = BrainTumorClassificationDataset(
    root_dir="data/classification"
)

print("Number of images:", len(dataset))
print("Number of classes:", dataset.num_classes)
print("Class distribution:", dataset.class_distribution())
```

BraTS:

```python
from data.brats_dataset import BraTSDataset

dataset = BraTSDataset(
    root_dir="data/BraTS2020",
    image_size=224
)

sample = dataset[0]

print("Image shape:", sample["image"].shape)
print("Mask shape:", sample["mask"].shape)
print("Patient:", sample["patient"])
print("Slice:", sample["slice"])
```

Expected BraTS format:

Image shape: [4, 224, 224]
Mask shape: [3, 224, 224]

## 7. Data Privacy and GitHub Policy

Do NOT commit medical imaging datasets to GitHub.

Do not upload:

- Raw MRI volumes
- NIfTI files
- DICOM files
- Patient images
- Segmentation masks
- Patient identifiers
- Dataset archives

Only the dataset loaders and documentation belong in this repository.

## 8. Recommended Local Structure

brain-tumor-uamtel/
├── data/
│   ├── __init__.py
│   ├── classification_dataset.py
│   ├── brats_dataset.py
│   ├── README.md
│   ├── classification/
│   ├── classification_dataset_2/
│   ├── classification_dataset_3/
│   └── BraTS2020/
├── models/
├── losses/
├── metrics/
├── configs/
├── scripts/
├── utils/
├── tests/
├── checkpoints/
└── outputs/

## 9. Dataset Reproducibility

Record the following for every experiment:

- Dataset name
- Dataset source
- Dataset URL
- Dataset version
- Download date
- Number of images/volumes
- Class distribution
- Train/validation/test split
- Preprocessing
- Augmentation
- Random seed
- Software/hardware environment

For Dataset 1, explicitly record whether the current 7,200-image release or
the older 7,023-image release was used.

## 10. Dataset Licensing

Users are responsible for complying with the license and terms of each
dataset. The repository does not grant additional rights to the underlying
medical images.

Consult the original dataset page before downloading or redistributing data.

## 11. Quick Start

1. Clone the repository.
2. Install the project requirements.
3. Download the required classification datasets.
4. Obtain authorized BraTS access.
5. Place the datasets under `data/`.
6. Run the dataset verification code above.
7. Start training using the configuration files in `configs/`.

Classification:

```bash
python scripts/train_classification.py --config configs/classification.yaml
```

Segmentation:

```bash
python scripts/train_segmentation.py --config configs/segmentation.yaml
```

Multi-task:

```bash
python scripts/train_multitask.py --config configs/multitask.yaml
```
