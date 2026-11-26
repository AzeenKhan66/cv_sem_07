# CNN Feature Visualization, Parameter Audit & Critical Analysis

Deep Learning & Computer Vision – Practical Laboratory Assignment

A Google Colab notebook that shows how a CNN turns raw pixels into feature maps. It applies five 3×3 filters, traces ReLU and pooling, audits parameter counts for two architectures, and provides evidence code for five viva questions.

## Files

| File | Description |
|---|---|
| `CNN_Visualization_Lab.ipynb` | The runnable Colab notebook |
| `CNN_Visualization_Lab.md` | All notebook code and notes in Markdown |
| `cnn_5x4_grid.png` | The 5×4 grid image (created when you run the notebook) |

## Requirements

Google Colab already includes everything needed. To run locally:

```bash
pip install numpy scipy matplotlib pandas tensorflow
```

- Python 3.9 or newer
- No GPU needed

## How to run

1. Upload `CNN_Visualization_Lab.ipynb` to [Google Colab](https://colab.research.google.com).
2. Choose `Runtime → Run all`.
3. To use your own image, set `USE_UPLOAD = True` in the input image cell and upload a photo when prompted. By default the notebook generates a synthetic building facade, so no upload is required.

## What the notebook does

### Part I – Filter pipeline
- Builds five 3×3 kernels: horizontal edge (Sobel-Y), vertical edge (Sobel-X), sharpen, box blur and emboss.
- Runs each kernel through four stages:
  - **A:** raw convolution
  - **B:** ReLU
  - **C:** 2×2 max pooling (stride 2)
  - **D:** 2×2 average pooling (stride 2)
- Plots a 5×4 grid with colorbars. Stages B, C and D share one intensity scale per row. The grid is saved as `cnn_5x4_grid.png`.

### Part II – Parameter audit
- Builds Model A (3×3 kernels, max pooling) and Model B (5×5 kernels, average pooling) in Keras.
- Both models use `Input(32,32,3) → Conv(5) → Pool → Conv(10) → Pool → GlobalAvgPool → Dense(10)`.
- Prints the hand calculations using `(Kh × Kw × Cin + 1) × Cout`, then `model.summary()`.
- Checks that the manual counts match Keras.

| | Model A | Model B |
|---|---|---|
| Conv1 | 140 | 380 |
| Conv2 | 460 | 1260 |
| Dense | 110 | 110 |
| **Total** | **710** | **1750** |

### Part III – Viva evidence
| Question | Evidence in the notebook |
|---|---|
| Q1 | Percent of pixels zeroed by ReLU, with ReLU vs Leaky ReLU vs Sigmoid maps |
| Q2 | Max vs average pooling peaks, plus a dark-ink-on-white demo |
| Q3 | Parameter increase (146.48% total) and two 3×3 layers vs one 5×5 |
| Q4 | Numerical proof that ReLU → MaxPool equals MaxPool → ReLU |
| Q5 | Cosine-similarity heatmap of filters, healthy vs collapsed initialization |

## Notes

- **Kernel naming:** the horizontal-edge kernel is Sobel-Y and the vertical-edge kernel is Sobel-X. Sobel-X responds to vertical edges, so the assignment's "Horizontal Edge (Sobel-X)" wording is slightly off.
- **Convolution:** it is implemented as cross-correlation (the kernel is not flipped), with zero padding, the same as in deep learning frameworks.
- **Numbers vary by image:** the values quoted in the answers (percent zeroed, pooling peaks) come from the synthetic facade. Rerun on your own image and use your own figures.
- **Seeds:** NumPy and TensorFlow seeds are fixed at 42, so runs are repeatable.

## Submission checklist

- [ ] Screenshot of the 5×4 grid
- [ ] Screenshots of both `model.summary()` outputs
- [ ] Hand-written parameter calculations
- [ ] Written answers to Questions 1–5 using your own image's numbers
- [ ] Your name, roll number, course and date on the report

## Academic integrity

The assignment allows AI help with the code, but the report and the oral defense must be your own work. Make sure you can explain every feature map, calculation and trade-off.
