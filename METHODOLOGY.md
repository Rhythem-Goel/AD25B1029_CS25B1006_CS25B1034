# Methodology: Hierarchical Cost-Sensitive Loss Function with Label Smoothing

## 1. Introduction

Traditional multi-class image classification models typically use the Categorical Cross-Entropy (CCE) loss function. While effective, CCE treats all classification errors equally. In the Oxford 102 Flower Dataset, some flower species are visually similar and may belong to related botanical groups, whereas others are significantly different.

To improve classification robustness and reduce overconfident predictions, a custom loss function named **Hierarchical Cost-Sensitive Cross-Entropy with Label Smoothing** was developed and integrated into the InceptionV3 model.

The proposed loss function combines:

1. Label Smoothing
2. Hierarchical Family Penalty
3. Standard Cross-Entropy Loss

This hybrid approach encourages better generalization while penalizing biologically distant classification errors more heavily.

---

## 2. Motivation

Standard Cross-Entropy Loss is defined as:

$$
L=-\sum_{i=1}^{K} y_i\log(p_i)
$$

where:

* K = number of classes
* yi = ground truth label
* pi = predicted probability

A limitation of this formulation is that all incorrect predictions receive the same penalty regardless of semantic similarity.

For example:

* Predicting Daisy as Sunflower
* Predicting Daisy as Orchid

Both errors are treated equally despite differing levels of similarity.

To address this limitation, an additional hierarchical penalty term was introduced.

---

## 3. Label Smoothing

### Problem

Traditional one-hot encoding assigns:

```text
True Class = 1
All Other Classes = 0
```

This encourages the network to become extremely confident, which often causes:

* Overfitting
* Poor calibration
* Reduced generalization

### Solution

Label smoothing redistributes a small portion of the probability mass across all classes.

The smoothed target is computed as:

y_{smooth}=y(1-\epsilon)+\frac{\epsilon}{K}

where:

* ε = smoothing factor (0.1)
* K = total number of classes (102)

For example:

Original target:

```text
[0,0,1,0,0]
```

Smoothed target:

```text
[0.02,0.02,0.92,0.02,0.02]
```

This prevents the model from assigning absolute confidence to a single class and improves robustness.

---

## 4. Smoothed Cross-Entropy Loss

Using the smoothed targets, the loss becomes:

L_{CE}=-\sum_{i=1}^{K} y_{smooth,i}\log(p_i)

The predicted probabilities are clipped to avoid numerical instability:

```python
y_pred_clipped = tf.clip_by_value(
    y_pred,
    1e-7,
    1.0 - 1e-7
)
```

This prevents undefined logarithm operations.

---

## 5. Hierarchical Family Penalty

### Concept

Flower species often belong to broader taxonomic families.

To incorporate hierarchical information, classes were grouped using:

```python
family_id = class_index // 10
```

Example:

| Classes | Family   |
| ------- | -------- |
| 0 – 9   | Family 0 |
| 10 – 19 | Family 1 |
| 20 – 29 | Family 2 |

This grouping serves as a proxy botanical hierarchy.

### Penalty Rule

If the predicted class belongs to a different family than the true class:

```python
is_wrong_family = 1
```

Otherwise:

```python
is_wrong_family = 0
```

The penalty term becomes:

P=\lambda\cdot I(f_{true}\neq f_{pred})

where:

* λ = family penalty coefficient
* I(.) = indicator function

For this project:

```text
λ = 0.5
```

---

## 6. Final Proposed Loss Function

The final loss combines:

1. Smoothed Cross-Entropy
2. Family Penalty

The overall objective function is:

L_{total}=L_{CE}+\lambda\cdot I(f_{true}\neq f_{pred})

where:

* LCE = Label-Smoothed Cross-Entropy
* λ = Family Penalty Weight
* I = Indicator Function

The loss is averaged across the batch during training.

---

## 7. Algorithm

### Step 1

Receive:

* Ground truth labels
* Predicted probabilities

### Step 2

Apply label smoothing:

```text
One-Hot → Smoothed Labels
```

### Step 3

Compute cross-entropy loss using smoothed labels.

### Step 4

Determine:

```text
True Family ID
Predicted Family ID
```

### Step 5

Apply family penalty if the families differ.

### Step 6

Add penalty to the base loss.

### Step 7

Compute mean batch loss and perform backpropagation.

---

## 8. Advantages of the Proposed Loss Function

### Improved Generalization

Label smoothing reduces overfitting by preventing extreme confidence.

### Better Calibration

Prediction probabilities become more realistic and reliable.

### Hierarchical Awareness

Misclassifications across different flower families receive stronger penalties.

### Robust Learning

The model learns finer distinctions between visually similar flower species.

### Stable Optimization

Probability clipping prevents numerical instability during training.

---

## 9. Parameters Used

| Parameter            | Value       |
| -------------------- | ----------- |
| Number of Classes    | 102         |
| Label Smoothing (ε)  | 0.1         |
| Family Penalty (λ)   | 0.5         |
| Probability Clipping | 1e-7        |
| Backbone Network     | InceptionV3 |

---

## 10. Conclusion

A custom Hierarchical Cost-Sensitive Cross-Entropy Loss with Label Smoothing was developed to improve flower classification performance. By combining label smoothing with a family-based penalty mechanism, the loss function encourages better generalization, reduces overconfidence, and incorporates hierarchical relationships between flower categories. This approach provides a more informative optimization objective than standard categorical cross-entropy and contributes to improved classification performance on the Oxford 102 Flower Dataset.
