# Copilot Instructions for Brain Tumor Classification Project

## Project Overview

This is a **comparative study** of CNN architectures for brain tumor classification from MRI images. The project includes PyTorch model training code in Jupyter notebooks, comprehensive evaluation with Grad-CAM interpretability, and an academic report written in LaTeX documenting results for the BRISC 2025 dataset.

## Project Architecture

### Core Components

**Data Pipeline** (`brisc2025/`):

- BRISC 2025 dataset: 6,000 T1-weighted MRI images (5,000 train, 1,000 test)
- Four classes: Glioma (22.9%), Meningioma (26.6%), Pituitary (29.1%), No Tumor (21.3%)
- Images resized to 512×512 during loading, stored as grayscale `.jpg` files
- Dataset has two tasks: `classification_task/` (this project) and `segmentation_task/` (not used)

**Training Notebooks** (`notebooks/`):

- `rsna_train.ipynb`: Main training loop with custom Dataset class, transforms, and model definitions
- `rsna_trainv2.ipynb`: Iterative experiments (similar structure)
- `rsna_eval.ipynb`: Model evaluation with Grad-CAM interpretability, confusion matrices, misclassification analysis
- `rsna_explore.ipynb`: Dataset exploration and visualization
- `rsna_train_results.ipynb`: Results analysis and figure generation

**Model Artifacts** (`notebooks/models/`):

- Three architectures tested: `baseline_cnn/`, `resnet18/`, `pretrained_densenet121/`
- Each directory contains: `best_<model>.pth` (best weights), `training_log.csv` (epoch metrics), `checkpoints/` (per-epoch `.pth` files)
- `evaluation_summary.csv`: Final test metrics for all models
- `figures/`: Generated plots (training curves, confusion matrices, Grad-CAM, misclassifications)

**Report** (`report/`):

- `main.tex`: Academic paper in standard article format (not IEEE template despite old instructions)
- `ref.bib`: BibTeX citations (ArXiv format for He2015 ResNet, Huang2018 DenseNet, Fateh2025 BRISC)
- Graphics path points to `../notebooks/models/figures/` for direct inclusion of generated plots
- Build artifacts go to `report/build/` (not committed)

### Data Flow

1. **Dataset Loading**: Custom `BrainTumorDataset` class in notebooks reads from `brisc2025/classification_task/train/` and `test/`
2. **Preprocessing**: OpenCV resize to 512×512 → PIL conversion → torchvision transforms (augmentation for train, normalize for val/test)
3. **Training**: PyTorch DataLoader with batch_size=16, Adam optimizer (lr=0.0005), 30 epochs, early stopping based on validation F1
4. **Evaluation**: Load best checkpoint → test set inference → metrics calculation → Grad-CAM generation → export figures to PDF
5. **Report**: LaTeX includes figures from `notebooks/models/figures/` → compile with `pdflatex` → `bibtex` → `pdflatex` (twice)

## Training Configuration Patterns

**Standard Hyperparameters** (from `rsna_train.ipynb`):

```python
IMG_SIZE = 512
BATCH_SIZE = 16  # Limited by GPU memory with 512×512 images
EPOCHS = 30
LEARNING_RATE = 0.0005
```

**Data Augmentation** (training only):

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(10)
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
transforms.ColorJitter(brightness=0.2, contrast=0.2)
transforms.Normalize(mean=[0.5], std=[0.5])  # Single-channel grayscale
```

**Model Architecture Convention**:

- Custom models inherit `nn.Module` with forward pass returning logits (no softmax)
- Loss: `nn.CrossEntropyLoss()` (handles softmax internally)
- Pretrained models: Replace final layer to output 4 classes: `model.classifier = nn.Linear(model.classifier.in_features, 4)`

**Checkpoint Saving Pattern**:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'val_f1': val_f1
}, f'models/{model_name}/checkpoints/{model_name}_epoch_{epoch}.pth')
```

## Critical Findings & Conventions

**ResNet-18 Outperforms Transfer Learning**:

- ResNet-18 (from scratch): 92.1% accuracy, 91.9% F1-score
- DenseNet-121 (pretrained): 79.6% accuracy (underperformed due to small batch size + limited dataset preventing effective use of ImageNet pretraining)
- Baseline CNN: 75.7% accuracy
- **Insight**: Small medical datasets with resource constraints (batch_size=16) prevent pretrained models from fine-tuning effectively

**Grad-CAM Implementation** (`rsna_eval.ipynb`):

- Visualizes model attention by computing gradients w.r.t. target layer activations
- Target layers: `model.layer4` (ResNet), `model.features[-1]` (DenseNet), custom final conv (Baseline)
- Generates 6-column visualizations: Original, Pred heatmap, Alt class heatmap, Pred overlay, Alt overlay, Colorbar
- Saved as PDF to `notebooks/models/figures/gradcam/` for LaTeX inclusion

**File Naming Follows Model Architecture**:

- Model directories: lowercase with underscores (e.g., `pretrained_densenet121/`)
- Checkpoint files: `{model_name}_epoch_{N}.pth` (e.g., `resnet18_epoch_25.pth`)
- Best model: `best_{model_name}.pth` (saved when validation F1 improves)

## LaTeX Report Structure

**Document Class**: `\documentclass[12pt, a4paper]{article}` (standard format, not IEEE)

**Graphics Integration**:

```latex
\graphicspath{{../notebooks/models/figures/}{../notebooks/models/}}
\includegraphics[width=0.8\textwidth]{metrics/training_curves.pdf}
```

**Section Structure** (from `report/main.tex`):

1. Introduction - Motivation and contributions
2. Dataset - BRISC 2025 description, class distribution, preprocessing
3. Methodology - Model architectures, training config, class imbalance handling
4. Evaluation Metrics - Accuracy, precision, recall, F1, confusion matrix
5. Results - Training curves, performance comparison, per-class metrics, prediction examples
6. Model Interpretability - Grad-CAM visualization analysis
7. Discussion - Why ResNet-18 worked best, DenseNet issues (batch size, training instability)
8. Conclusion - Summary and future work

**Citation Format**: BibTeX with `\cite{he2015deepresiduallearningimage}` for inline references

**Build Process**:

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex  # Second pass for cross-references
```

## Common Development Tasks

**Train a New Model**:

1. Open `notebooks/rsna_train.ipynb` or create similar notebook
2. Define model architecture inheriting `nn.Module`
3. Set hyperparameters (BATCH_SIZE=16 for 512×512 images)
4. Create DataLoaders with train/val transforms
5. Training loop: compute loss, backprop, log metrics to CSV
6. Save checkpoints to `notebooks/models/{model_name}/checkpoints/`
7. Save best model when validation F1 improves

**Evaluate Model Performance**:

1. Open `notebooks/rsna_eval.ipynb`
2. Load best checkpoint: `model.load_state_dict(torch.load('path/to/best_model.pth')['model_state_dict'])`
3. Run inference on test set with DataLoader
4. Generate confusion matrix, classification report (sklearn)
5. Apply Grad-CAM to sample images (correct predictions + misclassifications)
6. Export figures as PDF to `notebooks/models/figures/`

**Update Report with New Results**:

1. Regenerate figures in evaluation notebook (saves to `models/figures/`)
2. Update metrics in `report/main.tex` (accuracy, F1, error rates in tables)
3. Replace figure references if filenames changed
4. Recompile: `cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
5. Check `report/build/main.pdf` for output

**Add New Bibliography Entry**:

1. Add BibTeX entry to `report/ref.bib` (use ArXiv format for ML papers)
2. Cite in text: `\cite{bibkey}`
3. Run `bibtex main` before second LaTeX pass

## AI Writing Guidelines & Prevention Rules

When generating or editing text content, follow these strict rules to maintain academic quality:

**REQUIRED Style**:

- Use clear, simple language appropriate for academic papers
- Write spartan, informative sentences
- Use short, impactful sentences with active voice
- Focus on practical, actionable insights
- Support claims with data and examples when possible
- Address readers directly with "you" and "your" when appropriate

**PROHIBITED Patterns**:

- **CRITICAL: Never use em dashes (—) anywhere** - use commas, periods, or other standard punctuation only
- Avoid constructions like "not just this, but also this"
- Avoid metaphors and clichés in technical writing
- Avoid generalizations without supporting data
- Avoid common setup language: "in conclusion", "in closing", "in summary", etc.
- Avoid unnecessary adjectives and adverbs
- Avoid output warnings or notes unless specifically requested
- Avoid hashtags in academic writing
- Avoid semicolons in informal contexts (acceptable only in formal citations)
- Avoid passive voice constructions

**BANNED Words & Phrases** (never use these in generated text):

```
Filler words:
can, may, just, that, very, really, literally, actually, certainly, probably,
basically, could, maybe

AI-generated clichés:
delve, embark, enlightening, esteemed, shed light, craft, crafting, imagine,
realm, game-changer, unlock, discover, skyrocket, abyss, not alone,
in a world where, revolutionize, disruptive, utilize, utilizing, dive deep,
tapestry, illuminate, unveil, pivotal, intricate, elucidate, hence,
furthermore, however, harness, exciting, groundbreaking, cutting-edge,
remarkable, it remains to be seen, glimpse into, navigating, landscape,
stark, testament, in summary, in conclusion, moreover, boost, skyrocketing,
opened up, powerful, inquiries, ever-evolving
```

**Academic Alternatives** (use these instead of banned words):

- Instead of "delve into" → write "examine", "analyze", or "investigate"
- Instead of "shed light on" → write "clarify", "explain", or "demonstrate"
- Instead of "cutting-edge" → write "recent", "advanced", or "modern"
- Instead of "game-changer" → write "significant development" or "major advance"
- Instead of "revolutionize" → write "transform", "improve significantly", or "change"
- Instead of "utilize" → write "use"
- Instead of "realm" → write "field", "area", or "domain"
- Instead of "unveil" → write "present", "show", or "reveal"
- Instead of "navigate" → write "address", "handle", or "manage"
- Instead of "landscape" (metaphorical) → write "field", "environment", or be specific

**What Makes Text Sound AI-Generated** (avoid these patterns):

1. **Overuse of hedging language**: "may", "might", "could potentially", "it appears that"
2. **Excessive qualifiers**: "very", "really", "quite", "rather", "somewhat"
3. **Grandiose opening statements**: "In today's rapidly evolving world..."
4. **Metaphorical abstractions**: "journey", "tapestry", "landscape", "realm"
5. **Compound emphasis**: "not only...but also", "not just...but"
6. **Vague intensifiers**: "incredibly", "extremely", "remarkably", "truly"
7. **Setup phrases**: "It is important to note that...", "It should be mentioned..."
8. **Lists of three**: AI often defaults to three-item lists unnecessarily

**How to Sound Human in Academic Writing**:

- Start with specific, concrete statements
- Use precise technical terms instead of vague descriptors
- Cite data and measurements directly
- Write shorter sentences (15-25 words optimal)
- One idea per sentence
- Use transitions sparingly and only when necessary
- Be direct: "This system achieves 96% accuracy" not "It could be argued that this system potentially achieves..."

**Review Requirement**: Before finalizing any text generation, scan output to ensure:

1. No em dashes present (CRITICAL - check every line)
2. No banned words used (scan the complete list above)
3. Active voice predominates (passive voice <10% of sentences)
4. Sentences are direct and clear
5. No unnecessary qualifiers or hedging language
6. No metaphorical abstractions
7. Concrete, specific statements with data when possible
