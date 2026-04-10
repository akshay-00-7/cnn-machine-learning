# cnn-machine-learning

# CNN Image Classification — Deep Learning Assignment

This is my second deep learning assignment where I built a CNN (Convolutional Neural Network) from scratch for image classification. The task is the same as before — 10 class classification on 28x28 pixel images — but this time instead of a plain fully connected network, I used convolutional layers to actually learn spatial features from the images.

I ran 5 experiments on the same base CNN architecture, each time changing one thing to see how it affects the result. I also visualized the learned filters and used guided backpropagation to understand what the network is actually paying attention to.

---

## Dataset

IITM DL18 PA3 — same dataset as the previous assignment.

| Split | Samples |
|-------|---------|
| Train | 55,000 |
| Validation | 5,000 |
| Test | 10,000 |

Images are 28x28 grayscale, reshaped to `(28, 28, 1)` before feeding into the CNN. Pixel values normalized to [0, 1].

---

## Base Architecture

All 5 experiments use the same CNN. Only the training settings or initialization changes.

```
Input: 28x28x1

Conv2D(64 filters, 3x3, same padding, relu)
MaxPooling2D(2x2)

Conv2D(128 filters, 3x3, same padding, relu)
MaxPooling2D(2x2)

Conv2D(256 filters, 3x3, same padding, relu)
Conv2D(256 filters, 3x3, same padding, relu)
MaxPooling2D(2x2)

Flatten

Dense(1024, relu)
Dense(1024, relu)
BatchNormalization

Dense(10, softmax)
```

Early stopping was used in every experiment with `patience=5` monitoring validation loss.

---

## Experiments

### Experiment 1 — Baseline
Adam (lr=0.001) | Batch size: 50

**Val Accuracy: 92.24%**

Standard setup. Default Adam, batch size 50, Keras default weight initialization. This ended up being one of the best results across all experiments.

---

### Experiment 2 — Lower Learning Rate
Adam (lr=0.0005) | Batch size: 50

**Val Accuracy: 91.32%**

Lowered the learning rate to half. The model was slower to converge and early stopping ended training before it could fully catch up. Actually did worse than baseline — smaller lr doesn't always help.

---

### Experiment 3 — Smaller Batch Size
Adam (lr=0.001) | Batch size: 25

**Val Accuracy: 92.30%**

Halved the batch size. Tiny improvement over baseline (92.30% vs 92.24%). Smaller batches introduce more noise in gradient updates which can sometimes nudge the model into slightly better generalization.

---

### Experiment 4 — Xavier Initialization
Adam (lr=0.001) | Batch size: 50 | GlorotUniform initializer

**Val Accuracy: 91.42%**

Explicitly used Xavier (Glorot Uniform) initialization. Since this is already Keras's default, results were very similar to Experiment 1 — just marginally lower. No real difference.

---

### Experiment 5 — He Initialization
Adam (lr=0.001) | Batch size: 50 | HeNormal initializer

**Val Accuracy: 91.92%**

Switched to He initialization, which is the recommended choice for ReLU activations. Results were better than Xavier but still just below the default baseline. He init is theoretically better suited here, so the baseline might've just gotten lucky with the random seed.

---

## Results Summary

| Experiment | What Changed | Val Accuracy |
|------------|--------------|--------------|
| 1 — Baseline | Default Adam, batch=50 | 92.24% |
| 2 — Lower LR | lr = 0.0005 | 91.32% |
| 3 — Smaller Batch | batch = 25 | 92.30% |
| 4 — Xavier Init | GlorotUniform | 91.42% |
| 5 — He Init | HeNormal | 91.92% |

The default baseline and smaller batch size gave the best results. Lower learning rate was actually the worst — early stopping cut it off before it could converge properly.

---

## Visualization

Two types of visualization were done after training:

**Filter Visualization** — Plotted all 64 filters from the first conv layer. You can clearly see the network learned basic edge detectors and simple patterns, which is typical for early conv layers.

**Guided Backpropagation** — For a sample image from the validation set, guided backprop was used to highlight which pixels activated each filter the most. Ran it for 10 different filters to see the difference between what each one focuses on.

---

## How to Run

Trained on Google Colab with a T4 GPU.

1. Upload `cnnassignment.ipynb` to [Google Colab](https://colab.research.google.com)
2. Put the dataset zip in your Google Drive
3. Mount Drive and run cells in order

For running locally:
```bash
pip install tensorflow pandas numpy matplotlib jupyter
jupyter notebook cnnassignment.ipynb
```
Remove the Drive mount cell and update file paths if running locally.

---

## Files

```
cnnassignment.ipynb    main notebook
train.csv              training data (55k samples)
val.csv                validation data (5k samples)
test.csv               test data, no labels (10k samples)
sample_sub.csv         submission format
```
