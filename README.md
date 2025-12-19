# Brain Tumor Classification

Comparing different CNN models for classifying brain tumors from MRI scans.

## What this does

Takes MRI brain scans and classifies them into 4 types:
- Glioma
- Meningioma  
- Pituitary tumor
- No tumor

## Models tested

I tried three different approaches:
- Simple custom CNN (baseline)
- ResNet-18 (trained from scratch)
- DenseNet-121 (pretrained on ImageNet)

## Results

| Model | Accuracy |
|-------|----------|
| ResNet-18 | 92.1% |
| DenseNet-121 | 79.6% |
| Baseline CNN | 75.7% |

ResNet-18 worked best. DenseNet-121 had some training issues due to small batch size and limited GPU memory.

## Dataset

Using BRISC 2025 dataset:
- 6,000 MRI images total
- 5,000 for training, 1,000 for testing
- Images resized to 512×512

## Files

- `notebooks/` - Jupyter notebooks for training and evaluation
- `report/` - LaTeX report with full details
- `brisc2025/` - Dataset folder
- `requirements.txt` - Python dependencies

## Setup

```bash
pip install -r requirements.txt
```

Main packages: PyTorch, scikit-learn, matplotlib

## How to use

1. Check `notebooks/rsna_train.ipynb` for training code
2. See `notebooks/rsna_eval.ipynb` for evaluation
3. Full report in `report/main.tex`
