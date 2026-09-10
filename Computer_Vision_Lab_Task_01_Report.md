# Computer Vision Lab Task 01: Skin Lesion Classification Using Transfer Learning

## 1. Objective

This lab task investigates the classification of skin lesion images into nine diagnostic categories using deep learning. Three complementary experiments were conducted:

1. **End-to-end transfer learning** — fine-tuning eight pretrained CNN architectures directly on the skin lesion dataset.
2. **Deep feature extraction + classical machine learning** — using a pretrained CNN (DenseNet121) purely as a feature extractor and training traditional ML classifiers on the extracted features.
3. **Computational efficiency analysis** — comparing the trained CNNs on parameter count, model size, FLOPs, and inference latency, alongside their accuracy.

## 2. Dataset

- **Source:** ISIC skin cancer dataset (`nodoubttome/skin-cancer9-classesisic`, retrieved via `kagglehub`).
- **Classes:** 9 skin lesion categories (folder names used as class labels).
- **Split:** Stratified split into training (80%), validation (10%), and test (10%) sets using `train_test_split` with `random_state=42`.
- **Image size:** All images resized to 224 × 224 pixels.

## 3. Methodology

### 3.1 Preprocessing and Augmentation
- **Training set:** resize → random horizontal flip → random vertical flip → random affine transform (rotation ±15°, translation ±20%, scaling 0.8–1.2×, shear ±10°) → tensor conversion → ImageNet normalization.
- **Validation/test sets:** resize → tensor conversion → ImageNet normalization (no augmentation).
- **Batch size:** 32, loaded via PyTorch `DataLoader`.

### 3.2 Transfer Learning (Table 1)
Eight ImageNet-pretrained architectures were fine-tuned end-to-end for 5 epochs each, with the final classification layer replaced to output 9 classes:

`AlexNet, VGG16, VGG19, ResNet18, ResNet50, ResNet101, DenseNet121, EfficientNet-B0`

- **Optimizer:** Adam, learning rate = 1e-4
- **Loss:** Cross-entropy
- **Scheduler:** `ReduceLROnPlateau` (factor = 0.1, patience = 2)
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-Score, and AUC (weighted, one-vs-rest) computed on the held-out test set.

### 3.3 Deep Features + Classical Classifiers (Table 2)
- **Feature extractor:** DenseNet121 (pretrained, classification head replaced with `Identity`), used purely to generate deep feature vectors for train and test sets.
- **Feature scaling:** `StandardScaler` fit on training features, applied to both sets.
- **Classifiers evaluated:** Logistic Regression, Decision Tree, Random Forest (200 trees), K-Nearest Neighbors (k = 5), Linear SVM, RBF-SVM, and XGBoost (200 estimators, max depth 6, learning rate 0.05).
- Each classifier was trained on the scaled DenseNet121 features and evaluated on the test set with the same five metrics as above.

### 3.4 Computational Efficiency Analysis (Table 3)
For each trained CNN (excluding ResNet101), the following were measured:
- **Parameters (M):** total trainable parameter count.
- **Model Size (MB):** size of the saved `state_dict` on disk.
- **FLOPs (G):** computed via the `thop` profiler on a single 224×224 input.
- **Inference Time (ms):** average latency over 50 repetitions (after a 10-iteration GPU warm-up), single-image batch.
- **Accuracy (%):** carried over from Table 1 for direct comparison.

## 4. Results

### Table 1 — Comparison of Transfer Learning Models

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---|---|---|---|
| AlexNet | 66.53 | 67.69 | 66.53 | 64.27 | 94.22 |
| VGG16 | 65.25 | 64.13 | 65.25 | 62.50 | 93.64 |
| VGG19 | 61.86 | 56.04 | 61.86 | 55.71 | 92.53 |
| **ResNet18** | **78.39** | **76.47** | **78.39** | **77.16** | **96.00** |
| ResNet50 | 76.69 | 74.73 | 76.69 | 75.43 | 95.71 |
| ResNet101 | 70.76 | 71.03 | 70.76 | 69.79 | 95.03 |
| DenseNet121 | 73.31 | 71.01 | 73.31 | 70.93 | 96.14 |
| EfficientNet-B0 | 75.00 | 73.43 | 75.00 | 73.03 | 96.21 |

### Table 2 — Comparison of Different Classifiers (DenseNet121 Deep Features)

| Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---|---|---|---|---|
| Deep Features | Logistic Regression | 50.42 | 50.58 | 50.42 | 49.82 | 86.16 |
| Deep Features | Decision Tree | 35.59 | 35.70 | 35.59 | 35.18 | 62.06 |
| Deep Features | Random Forest | 56.36 | 54.84 | 56.36 | 50.57 | 87.12 |
| Deep Features | K-Nearest Neighbors (KNN) | 42.37 | 42.35 | 42.37 | 40.25 | 75.19 |
| Deep Features | Linear SVM | 53.81 | 55.06 | 53.81 | 53.34 | 90.88 |
| **Deep Features** | **RBF-SVM** | **58.47** | 54.67 | 58.47 | 52.09 | 90.53 |
| Deep Features | XGBoost | 57.20 | 56.17 | 57.20 | 51.99 | 89.54 |

### Table 3 — Computational Efficiency Comparison

| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
|---|---|---|---|---|---|
| AlexNet | 57.04 | 217.60 | 0.71 | 2.11 | 66.53 |
| VGG16 | 134.30 | 512.32 | 15.47 | 9.67 | 65.25 |
| VGG19 | 139.61 | 532.57 | 19.63 | 11.56 | 61.86 |
| ResNet18 | 11.18 | 42.73 | 1.82 | 2.77 | 78.39 |
| ResNet50 | 23.53 | 90.05 | 4.13 | 5.72 | 76.69 |
| DenseNet121 | 6.96 | 27.14 | 2.90 | 14.81 | 73.31 |
| EfficientNet-B0 | 4.02 | 15.62 | 0.41 | 8.13 | 75.00 |

## 5. Analysis and Discussion

**Best overall CNN — ResNet18.** Despite being one of the smallest and fastest networks tested (11.18M parameters, 2.77 ms inference), ResNet18 achieved the highest accuracy (78.39%), precision, recall, and F1-score among all eight transfer learning models, and the second-highest AUC (96.00%). This suggests that, for this dataset size and 5-epoch fine-tuning budget, a moderately deep residual network generalizes better than both very shallow (AlexNet) and very large/deep (VGG19, ResNet101) architectures — likely due to a combination of residual connections easing optimization and a parameter count well matched to the amount of training data available.

**Diminishing returns from depth/size.** VGG16 and VGG19 — despite having by far the most parameters (134–140M) and the largest model sizes (>500 MB) — produced the *worst* accuracy and F1-scores of the CNN group. This is a classic case of over-parameterized classical CNNs struggling to fine-tune efficiently within a short training schedule, and it reinforces that parameter count alone is a poor proxy for downstream accuracy.

**Efficiency leaders.** EfficientNet-B0 is the most parameter- and memory-efficient model (4.02M params, 15.62 MB) and still reaches a strong 96.21% AUC — the best AUC in Table 1 — while AlexNet remains the fastest at inference (2.11 ms) due to its shallow architecture, despite its comparatively weak accuracy. ResNet18 offers the best overall trade-off, combining top accuracy with near-fastest inference and a small footprint (42.73 MB), making it the most practical choice if a single deployment model must be chosen.

**Deep features + classical ML underperforms end-to-end fine-tuning.** Every classical classifier trained on frozen DenseNet121 features (Table 2) scored well below the fine-tuned CNNs in Table 1 — the best classical result (RBF-SVM, 58.47% accuracy) is roughly 20 points behind ResNet18's fine-tuned accuracy (78.39%). This gap indicates that end-to-end fine-tuning, which adapts the convolutional backbone itself to the target domain, captures lesion-specific discriminative patterns that a frozen, generic ImageNet feature space cannot. Among the classical classifiers, tree-based and margin-based methods (Random Forest, RBF-SVM, XGBoost) outperformed simpler linear/instance-based methods (Logistic Regression, KNN), and the Decision Tree classifier performed worst by a wide margin (35.59% accuracy, 62.06% AUC), consistent with its tendency to overfit high-dimensional deep feature vectors without ensembling.

**AUC vs. accuracy divergence.** Across both tables, AUC values remain comparatively high even when accuracy is mediocre (e.g., Logistic Regression: 50.42% accuracy but 86.16% AUC). This reflects the multi-class, weighted one-vs-rest AUC's sensitivity to class-probability ranking rather than hard classification correctness, and is a useful reminder that AUC alone can overstate practical classification performance on an imbalanced, multi-class dataset.

## 6. Conclusions

- **ResNet18** is the recommended model for this skin lesion classification task, offering the best balance of accuracy (78.39%), AUC (96.00%), model size, and inference speed.
- Fine-tuning the full network end-to-end substantially outperforms using a frozen pretrained network purely as a feature extractor for classical ML classifiers.
- Larger architectures (VGG16/VGG19) do not translate into better performance here, highlighting the importance of matching model capacity to dataset size and training budget rather than defaulting to the largest available network.
- If deployment constraints (memory, latency) are critical rather than absolute accuracy, EfficientNet-B0 and AlexNet are strong lightweight alternatives, with EfficientNet-B0 preferred for its markedly better AUC and accuracy at a similar computational cost.
