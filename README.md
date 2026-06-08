# 🌸 Flower Species Recognition using InceptionV3

## Project Overview

This project develops a deep learning–based flower classification system using the Oxford 102 Flower Categories dataset. The objective is to accurately classify flower images into one of 102 species using transfer learning with InceptionV3. The model incorporates advanced techniques such as custom hierarchical loss, data augmentation, fine-tuning, and performance evaluation using multiple classification metrics.

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

The original dataset split was modified so that the larger image pool is used for training, improving model generalization and classification performance.

---

# Team Contribution and Workflow

The project was developed collaboratively by dividing the workflow into three major modules.

---

## Member A: Dataset Preparation, Augmentation and Baseline Development

### Responsibilities

#### 1. Dataset Preparation

* Loaded Oxford 102 Flower Dataset
* Extracted image archive
* Processed image labels and split information
* Created structured train, validation and test directories
* Generated dataset statistics and visualizations

#### 2. Exploratory Data Analysis (EDA)

* Analyzed class distribution
* Visualized train/validation/test split
* Displayed sample flower images
* Verified dataset integrity

#### 3. Data Augmentation

Implemented image augmentation techniques including:

* Rotation
* Width and height shifting
* Zooming
* Shearing
* Horizontal and vertical flipping
* Brightness adjustment
* Channel shifting

#### 4. Baseline Pipeline

* Created ImageDataGenerator pipelines
* Configured batch loading
* Established preprocessing workflow
* Prepared baseline model training environment

### Deliverables

* Dataset preprocessing module
* Dataset visualization graphs
* Augmentation pipeline
* Data loading framework

---

## Member B: InceptionV3 Architecture Modifications and Custom Loss Function

### Responsibilities

#### 1. Transfer Learning Architecture

Implemented InceptionV3 using ImageNet pretrained weights.

### Architecture Modifications

* Removed original classification head
* Added Global Average Pooling layer
* Added Batch Normalization
* Added Dense(512) layer
* Added Dropout regularization
* Added Dense(256) layer
* Added final 102-class softmax classifier

#### 2. Mixed Precision Training

* Enabled float16 computation
* Reduced GPU memory consumption
* Improved training speed

#### 3. Custom Hierarchical Loss Function

Designed a cost-sensitive loss function incorporating:

##### Label Smoothing

* Reduced model overconfidence
* Improved generalization

##### Family Penalty

* Penalized predictions belonging to incorrect flower families
* Encouraged taxonomically meaningful predictions

#### 4. Fine-Tuning Strategy

* Controlled layer freezing
* Unfroze top InceptionV3 layers
* Optimized transfer learning process

### Deliverables

* Modified InceptionV3 architecture
* Custom hierarchical loss implementation
* Fine-tuning strategy
* Transfer learning framework

---

## Member C: Training Experiments, Evaluation and Documentation

### Responsibilities

#### 1. Model Training

Conducted two-stage training:

### Phase 1

* Trained custom classification head
* Kept InceptionV3 backbone frozen

### Phase 2

* Fine-tuned upper InceptionV3 layers
* Applied cosine learning rate decay
* Used gradient clipping for stable optimization

#### 2. Hyperparameter Experiments

Evaluated:

* Learning rates
* Epoch configurations
* Fine-tuning depth
* Dropout settings
* Optimization strategies

#### 3. Model Evaluation

Generated performance metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix
* Per-Class Accuracy Analysis

#### 4. Visualization and Reporting

Produced:

* Training loss curves
* Accuracy curves
* Dataset split graphs
* Confusion matrix visualizations
* Sample prediction results

#### 5. Documentation

* Compiled project findings
* Summarized experimental results
* Prepared final report and presentation materials

### Deliverables

* Trained model
* Evaluation metrics
* Experimental analysis
* Final report
* Presentation materials

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

# Evaluation Metrics

The final model performance was assessed using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Per-Class Accuracy

These metrics provide a comprehensive evaluation of classification performance across all 102 flower categories.

---

# Conclusion

This project demonstrates the effectiveness of transfer learning using InceptionV3 for large-scale flower classification. Through extensive data augmentation, custom loss design, and systematic fine-tuning, the model achieves strong performance while maintaining robust generalization across diverse flower species.
