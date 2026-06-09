# 🌸 Flower Species Recognition using InceptionV3

## Project Overview

This project builds a flower classification system using deep learning on the Oxford 102 Flower Categories dataset. The goal is to correctly identify flower species from images — all 102 of them — using transfer learning on top of InceptionV3. Along the way, we explored things like custom loss functions, data augmentation strategies, and a two-phase fine-tuning approach to squeeze out better performance.

---

## Dataset

**Dataset:** Oxford 102 Flower Categories

* Total Classes: 102
* Total Images: 8,189
* Image Type: RGB Flower Images
* Input Size: 299 × 299 × 3

### Dataset Split

| Split      | Images |
| ---------- | -----: |
| Training   |   6149 |
| Validation |   1020 |
| Testing    |   1020 |

We reshuffled the original dataset split so more images went into training. The default split wasn't ideal for learning, and this change made a noticeable difference in how well the model generalized.

---

# Team Contribution and Workflow

We split the work into three main areas so each person could go deep on their part while keeping everything connected.

---

Member A — Data Preparation & Augmentation

This member handled everything before the model sees a single image. That meant downloading and extracting the Oxford dataset, parsing labels and split files, and organizing images into proper train/val/test folders. We also did a round of exploratory analysis — checking class distributions, making sure nothing was corrupted, and visualizing what the data actually looked like.

On the augmentation side, we applied a fairly aggressive set of transforms: rotation, shifting, zoom, shear, flips (both horizontal and vertical), brightness variation, and channel shifting. The idea was to make the model robust to the kinds of variation you'd expect in real flower photos — different angles, lighting, cropping, etc.

Key outputs: preprocessing module, data visualizations, augmentation pipeline, loading framework

---

Member B — Model Architecture & Loss Function
This member took InceptionV3 (pretrained on ImageNet) and adapted it for our task. The original classification head was swapped out for a custom stack: Global Average Pooling → Batch Normalization → Dense(512) → Dropout → Dense(256) → 102-class softmax. Pretty standard transfer learning setup, but the details matter.
The more interesting part was the custom loss function. Rather than plain cross-entropy, we designed a hierarchical loss with two components:

Label smoothing — softens the one-hot targets a bit, which helped prevent the model from becoming overconfident on training data.
Family penalty — adds extra cost when the model predicts the wrong family of flower, not just the wrong species. The intuition is that misclassifying a tulip as a rose is worse than confusing two tulip subspecies.

We also enabled mixed precision (float16) training to cut memory usage and speed things up on GPU.
Key outputs: modified InceptionV3, custom loss, fine-tuning setup
---
Member C — Training, Evaluation & Reporting

Training happened in two phases. In Phase 1, the InceptionV3 backbone was frozen and only the new classification head was trained. Once that converged, Phase 2 unfroze the upper layers of the backbone for fine-tuning, using cosine learning rate decay and gradient clipping to keep things stable.

We ran several experiments tweaking learning rates, dropout rates, how many layers to unfreeze, and training length. Not everything helped — some combinations made 
things worse — but that's kind of the point of running experiments.

For evaluation, we looked at accuracy, precision, recall, F1, a full classification report, and a confusion matrix. We also did per-class accuracy breakdowns because aggregate metrics can hide a lot with 102 classes.

Key outputs: trained model, evaluation metrics, experiment logs, final report, presentation materials

---

# Technologies Used

* Python
* TensorFlow / Keras
* InceptionV3
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* SciPy

---

#Evaluation

We evaluated the final model on the held-out test set using accuracy, precision, recall, F1, confusion matrix, and per-class accuracy. With 102 classes, per-class analysis was especially useful for spotting where the model struggled — some species are genuinely hard to distinguish visually.

#Takeaways

Transfer learning works well here, but the details really matter. Naive fine-tuning without careful learning rate scheduling tended to degrade performance. The hierarchical loss didn't produce dramatic improvements on its own, but combined with the other techniques it contributed to a more robust model. If we were to extend this, better class-balanced sampling and ensembling would be natural next steps.
