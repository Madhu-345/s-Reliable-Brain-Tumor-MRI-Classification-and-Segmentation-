# Brain Tumor UAMTEL

## Uncertainty-Aware Multi-Task Evidential Learning for Brain Tumor MRI Classification and Segmentation

This repository provides an implementation of an uncertainty-aware multi-task deep learning framework for brain tumor MRI analysis. The framework jointly addresses:

- Brain tumor MRI classification
- Brain tumor MRI segmentation
- Evidential uncertainty estimation
- Reliability-aware prediction
- Selective prediction with abstention
- Uncertainty-calibrated spatial attention
- Multi-objective multi-task learning

The implementation is organized to support reproducible experimentation, model evaluation, ablation studies, and reliability analysis.

---

## 1. Framework Overview

The proposed framework uses a shared ResNet-50 feature extraction backbone followed by uncertainty-calibrated spatial attention and two task-specific branches.

### Classification branch

The classification branch performs four-class brain tumor classification:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

An evidential classification head produces class evidence, Dirichlet parameters, class probabilities, predictive uncertainty, and a reliability score.

### Segmentation branch

The segmentation branch uses multi-scale features and an evidential segmentation head to produce pixel-wise predictions for:

- Whole Tumor (WT)
- Tumor Core (TC)
- Enhancing Tumor (ET)

Pixel-wise uncertainty is used to calculate a segmentation reliability score.

### Reliability-aware decision

Predictions are not treated as equally reliable. The framework uses the estimated uncertainty to calculate a reliability score and supports an accept/abstain decision:

```text
High Reliability  -> Accept Prediction
Low Reliability   -> Abstain
```

The default operating threshold used by the implementation is 0.80, unless changed through the experiment configuration.

---

## 2. Repository Structure

```text
brain-tumor-uamtel/
│
├── configs/
│   ├── __init__.py
│   ├── classification.yaml
│   ├── segmentation.yaml
│   ├── multitask.yaml
│   ├── ablations.yaml
│   ├── ablation_no_alignment.yaml
│   ├── ablation_no_calibration.yaml
│   └── ablation_no_kl.yaml
│
├── data/
│   ├── __init__.py
│   ├── brats_dataset.py
│   ├── classification_dataset.py
│   └── README.md
│
├── docs/
│   ├── reproducibility.md
│   └── manuscript_code_mapping.md
│
├── losses/
│   ├── __init__.py
│   ├── evidential.py
│   ├── dice.py
│   ├── calibration.py
│   ├── kl_regularization.py
│   └── attention_uncertainty.py
│
├── metrics/
│   ├── __init__.py
│   ├── classification.py
│   ├── segmentation.py
│   ├── uncertainty.py
│   ├── selective.py
│   └── reliability.py
│
├── models/
│   ├── __init__.py
│   ├── backbone.py
│   ├── ucsa.py
│   ├── evidential_heads.py
│   ├── decoder.py
│   └── multitask_model.py
│
├── scripts/
│   ├── __init__.py
│   ├── train_classification.py
│   ├── train_segmentation.py
│   ├── train_multitask.py
│   ├── evaluate.py
│   ├── selective_eval.py
│   ├── compute_selective_table.py
│   └── make_example_predictions.py
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   └── test_reliability.py
│
├── utils/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── config.py
│   ├── seed.py
│   ├── logger.py
│   └── visualization.py
│
├── checkpoints/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 3. Methodology

The overall processing pipeline is:

```text
Input Brain MRI
       |
       v
Preprocessing
       |
       v
ResNet-50 Feature Extraction
       |
       v
Multi-Scale Feature Maps
       |
       v
Uncertainty-Calibrated Spatial Attention
       |
       v
Uncertainty-Aware Multi-Scale Features
       |
       +------------------------+
       |                        |
       v                        v
Classification              Segmentation
Branch                      Branch
       |                        |
       v                        v
Evidential                  Multi-Scale
Classification Head         Decoder
       |                        |
       v                        v
Dirichlet Parameters       Evidential
       |                    Segmentation Head
       v                        |
Class Probability              v
       |                    Tumor Mask
       v                        |
Predictive Uncertainty         v
       |                    Pixel Uncertainty
       v                        |
Reliability Score              v
       |                    Reliability Score
       v                        |
Predict / Abstain              v
                            Accept / Abstain
```

---

## 4. Uncertainty-Calibrated Spatial Attention

For each multi-scale feature map, the framework generates an attention map and an uncertainty map.

The attention map is calibrated using uncertainty so that uncertain feature regions receive reduced attention.

The resulting refined representation is then used by both classification and segmentation branches.

---

## 5. Evidential Learning

Instead of directly producing only softmax probabilities, the evidential heads estimate non-negative evidence.

For a class `k`:

```text
Evidence -> Dirichlet Parameter -> Class Probability
                         |
                         v
                    Uncertainty
                         |
                         v
                    Reliability
```

The implementation uses:

```text
evidence = Softplus(logits)
alpha    = evidence + 1
```

The predictive uncertainty is calculated from the total Dirichlet strength.

For classification:

```text
U = K / S
R = 1 - U
```

where:

- `K` is the number of classes
- `S` is the total Dirichlet strength
- `U` is predictive uncertainty
- `R` is reliability

---

## 6. Multi-Objective Training

The framework combines the classification and segmentation objectives with regularization and uncertainty-related objectives.

The overall training objective is:

```text
L_Total =
    lambda_c   L_ECL
  + lambda_s   L_ESL
  + lambda_k   L_DKL
  + lambda_cal L_Cal
  + lambda_a   L_AU
```

where:

- `L_ECL` = evidential classification loss
- `L_ESL` = evidential segmentation loss
- `L_DKL` = Dirichlet KL regularization
- `L_Cal` = calibration loss
- `L_AU` = attention-uncertainty alignment loss

The segmentation objective additionally incorporates Dice loss.

---

## 7. Datasets

### Classification datasets

The manuscript configuration uses three brain MRI classification datasets containing four diagnostic categories:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

The reported dataset sizes are approximately:

| Dataset | Images |
|---|---:|
| Classification Dataset 1 | 7,200 |
| Classification Dataset 2 | 3,264 |
| Classification Dataset 3 | 1,311 |

The actual dataset files are not included in this repository.

### BraTS 2020

The segmentation branch is designed for BraTS 2020 multi-modal MRI data.

The input modalities are:

- T1
- T1ce
- T2
- FLAIR

The segmentation targets are represented as:

- WT
- TC
- ET

Dataset files must be obtained from their respective official or permitted sources and placed locally.

---

## 8. Data Organization

The classification dataset loader expects a directory structure organized by split and class.

A typical structure is:

```text
classification_dataset/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
│
├── val/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
│
└── test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/
```

For BraTS, use the directory organization expected by `data/brats_dataset.py`.

See:

```text
data/README.md
```

for the dataset-specific organization and preprocessing requirements.

---

## 9. Preprocessing

The classification pipeline includes:

- Skull stripping
- Image resizing to 224 × 224
- Intensity normalization
- Horizontal/vertical flipping as configured
- Rotation augmentation
- Mild intensity variation

The BraTS loader handles multi-modal input and WT/TC/ET segmentation targets.

The exact preprocessing applied during an experiment should be recorded in the corresponding configuration file.

---

## 10. Installation

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd brain-tumor-uamtel
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

### PyTorch

PyTorch installation may need to be adapted to the available CPU/GPU and CUDA environment. For GPU training, install the appropriate PyTorch build for the local CUDA version.

---

## 11. Configuration

Experiment settings are stored in:

```text
configs/
```

Main configurations:

```text
classification.yaml
segmentation.yaml
multitask.yaml
ablations.yaml
```

The configuration loader is implemented in:

```text
utils/config.py
```

Example:

```python
from utils.config import load_config

config = load_config("configs/multitask.yaml")

batch_size = config.get(
    "training.batch_size",
    16,
)

learning_rate = config.get(
    "training.learning_rate",
    1e-4,
)
```

---

## 12. Training

### Classification

```bash
python scripts/train_classification.py \
    --classification-root /path/to/classification_dataset
```

The classification training pipeline uses the evidential classification branch.

### Segmentation

```bash
python scripts/train_segmentation.py \
    --segmentation-root /path/to/BraTS
```

### Multi-task training

```bash
python scripts/train_multitask.py \
    --classification-root /path/to/classification_dataset \
    --segmentation-root /path/to/BraTS
```

The multi-task training procedure alternates classification and segmentation batches so that datasets with different input channel configurations can be used with the shared model.

---

## 13. Training Configuration

The main training configuration includes:

```text
Optimizer       : Adam
Learning rate   : 1e-4
Weight decay    : 1e-5
Batch size      : 16
Maximum epochs  : 100
Scheduler       : Cosine annealing
```

The training scripts also support:

- Random seed control
- Early stopping
- Automatic mixed precision where supported
- Model checkpointing
- Training history logging

---

## 14. Checkpoints

Model checkpoints are stored in:

```text
checkpoints/
```

Typical generated files include:

```text
classification_best.pt
classification_last.pt

segmentation_best.pt
segmentation_last.pt

multitask_best.pt
multitask_last.pt
```

Checkpoint management is implemented in:

```text
utils/checkpoint.py
```

Generated model weights are excluded from Git using `.gitignore`.

---

## 15. Evaluation

The main evaluation script is:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/multitask_best.pt
```

Evaluation includes classification, segmentation, uncertainty, and reliability measures supported by the selected checkpoint and dataset.

The evaluation utilities are located in:

```text
metrics/
```

---

## 16. Selective Prediction

Selective prediction evaluates model behavior at different reliability thresholds.

Run:

```bash
python scripts/selective_eval.py \
    --checkpoint checkpoints/classification_best.pt \
    --classification-root /path/to/classification_dataset
```

The default thresholds include:

```text
0.50
0.60
0.70
0.80
0.90
0.95
```

The framework reports:

- Coverage
- Selective accuracy
- Selective risk

Coverage is the percentage of samples retained after applying the reliability threshold.

Selective risk is the error rate among retained predictions.

---

## 17. Computing a Selective Evaluation Table

If a prediction CSV contains:

```text
correct,reliability
```

the selective table can be generated with:

```bash
python scripts/compute_selective_table.py \
    --input outputs/predictions.csv
```

The resulting table is written to:

```text
outputs/selective_table.csv
```

---

## 18. Example Predictions

To generate example classification and segmentation predictions:

```bash
python scripts/make_example_predictions.py \
    --checkpoint checkpoints/multitask_best.pt
```

Generated visualizations and prediction files are stored under the configured output directory.

---

## 19. Testing

Run all tests with:

```bash
pytest
```

Run model tests:

```bash
pytest tests/test_model.py
```

Run reliability tests:

```bash
pytest tests/test_reliability.py
```

The tests cover:

- Model construction
- Forward propagation
- Four-channel input handling
- Evidential outputs
- Probability normalization
- Uncertainty range
- Reliability range
- Reliability threshold decisions
- Segmentation reliability
- Empty-mask reliability fallback
- Gradient propagation

---

## 20. Reproducibility

The repository provides reproducibility utilities through:

```text
utils/seed.py
```

Set a fixed seed before training:

```python
from utils.seed import set_seed

set_seed(
    42,
    deterministic=True,
)
```

For repeated experiments, different seeds can be supplied while keeping the experimental configuration unchanged.

Additional reproducibility information is provided in:

```text
docs/reproducibility.md
```

---

## 21. Model Outputs

### Classification output

The evidential classification head provides:

```text
evidence
Dirichlet parameters
class probabilities
predicted class
predictive uncertainty
reliability score
```

### Segmentation output

The evidential segmentation head provides:

```text
pixel-wise evidence
Dirichlet parameters
pixel-wise probabilities
segmentation mask
pixel-wise uncertainty
segmentation reliability
```

---

## 22. Reliability and Abstention

For classification:

```text
Reliability = 1 - Predictive Uncertainty
```

A prediction can be accepted when:

```text
Reliability >= threshold
```

and rejected/abstained when:

```text
Reliability < threshold
```

This enables the framework to distinguish between confident and uncertain predictions instead of forcing a decision for every sample.

---

## 23. Ablation Studies

The repository includes configuration files for studying the contribution of major components, including:

- Dirichlet KL regularization
- Calibration loss
- Attention-uncertainty alignment

Ablation configurations are located in:

```text
configs/
```

The ablation experiments can be used to evaluate the effect of removing individual components from the complete framework.

---

## 24. Computational Considerations

Training brain MRI models can require substantial GPU memory and computation time, particularly for multi-task training and multi-modal segmentation.

Recommended practice:

- Use a CUDA-enabled GPU when available.
- Adjust batch size according to available GPU memory.
- Enable mixed precision when supported.
- Store datasets outside the Git repository.
- Store checkpoints and generated outputs outside version control.

---

## 25. Data and Model Weights

MRI datasets are **not included** in this repository.

Pre-trained model weights and generated checkpoints are also not included by default.

Users must obtain datasets from the appropriate source and comply with the corresponding dataset license, terms of use, privacy requirements, and redistribution restrictions.

---

## 26. Reproducible Experiment Workflow

A typical workflow is:

```text
1. Obtain permitted datasets
          |
          v
2. Organize datasets
          |
          v
3. Configure experiment YAML
          |
          v
4. Set random seed
          |
          v
5. Train model
          |
          v
6. Save best checkpoint
          |
          v
7. Evaluate classification
          |
          v
8. Evaluate segmentation
          |
          v
9. Evaluate uncertainty
          |
          v
10. Evaluate selective prediction
          |
          v
11. Run ablation experiments
          |
          v
12. Generate example predictions
```

---

## 27. Citation

If this repository is used in academic work, please cite the associated research publication when a formal citation is available.

---

## 28. License

This project is distributed under the MIT License.

See:

```text
LICENSE
```

for the complete license text.

---

