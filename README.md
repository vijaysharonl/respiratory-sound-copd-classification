# Respiratory Sound Classification

This project classifies respiratory sounds into **COPD** and **Non-COPD** using Mel spectrograms and a Convolutional Neural Network (CNN) enhanced with Efficient Channel Attention (ECA).

---

## 📂 Dataset
- **Source:** [ICBHI Respiratory Sound Database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)
- **Input:** `.wav` audio files and corresponding `.txt` annotations
- **Labels:** Extracted from `patient_diagnosis.csv`

---

## 🔄 Pipeline
1. **Segment Audio:** Extract labeled segments from `.wav` files using annotation files.
2. **Convert to Mel Spectrograms:** Generate visual representations of audio segments.
3. **Organize Dataset:** Separate into COPD and Non-COPD based on patient diagnosis.
4. **Train/Test Split:** Create train/test sets for model training.
5. **Model Training:** Train CNN with dual-path convolutions and ECA attention.
6. **Evaluation:** Accuracy, loss, confusion matrix, and classification report.

---

## ✅ Requirements
- Python 3.8+
- PyTorch
- Librosa
- scikit-learn
- Matplotlib
- Seaborn

---

## 🧠 Training

The model is trained using the following techniques for improved robustness and performance:

- **Dynamic Sample Weighting:** Loss is adaptively scaled based on per-sample difficulty, improving generalization on imbalanced data.
- **Gradient Clipping:** Prevents exploding gradients by capping maximum gradient norms during backpropagation.
- **Early Stopping:** Stops training when validation loss stops improving, avoiding overfitting.

---

## 📊 Results & Plots
Visualizations include:

Training Curves: Plots of accuracy and loss over epochs for both training and validation sets.
Confusion Matrix: Shows prediction performance across COPD and Non-COPD classes.
Classification Report: Precision, recall, and F1-score metrics to evaluate the model.

These plots help analyze learning behavior, identify overfitting, and assess classification performance visually.
